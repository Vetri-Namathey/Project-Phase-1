"""Experiment B: train TwinGuard with 3 independently-seeded OOD heads on
Cityscapes + CutMix outlier exposure, evaluate on Fishyscapes each epoch,
log to MLflow, save the best checkpoint by val AUROC.

Fishyscapes is split into val/test halves. Checkpoint selection reads the
val half only; the test half is reported. Training never touches either.
"""

import contextlib
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config
from data.cityscapes_dataset import CityscapesDataset
from data.cutmix import CutMixAugmentedDataset
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from losses import build_seg_criterion, compute_total_loss
from metrics import ConfusionMatrix, ScoreHistogram
from model.twinguard_model import (TwinGuardModel, verify_encoder_frozen,
                                   verify_heads_independent)
from utils import get_device


def amp_context(device):
    """bfloat16 autocast on CUDA, a no-op everywhere else.

    Used for both training and evaluation. Scores come out of a sigmoid in
    [0,1], where bf16's ~3 decimal digits of precision are far finer than the
    200k-bin metric histogram can resolve, so this does not move the reported
    numbers.
    """
    if config.USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_ood(model, pairs, device, num_heads):
    """Streams each image into histogram accumulators -- no 170M-element
    arrays held in memory, no repeated full sorts."""
    model.eval()
    fused_hist = ScoreHistogram()
    head_hists = [ScoreHistogram() for _ in range(num_heads)]
    disagreement_hist = ScoreHistogram()
    raw_sums, raw_sqsums, raw_counts = np.zeros(num_heads), np.zeros(num_heads), 0
    pos_sums, neg_sums = np.zeros(num_heads), np.zeros(num_heads)
    pos_counts, neg_counts = 0, 0

    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid = label_map != 255
        labels = label_map[valid]

        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        out = {k: v.float() for k, v in out.items()}

        fused = F.interpolate(out["ood_fused"].unsqueeze(1), size=label_map.shape,
                              mode="bilinear", align_corners=False)
        fused_hist.update(fused.squeeze().cpu().numpy()[valid], labels)

        per_head = F.interpolate(out["ood_scores"], size=label_map.shape,
                                 mode="bilinear", align_corners=False).squeeze(0)
        # (num_heads, n_valid) -- flattened to the valid pixels once, so the
        # histograms and the separation stats index the same thing.
        per_head_valid = per_head.cpu().numpy()[:, valid]
        for h in range(num_heads):
            head_hists[h].update(per_head_valid[h], labels)

        disagree = F.interpolate(out["ood_disagreement"].unsqueeze(1),
                                 size=label_map.shape, mode="bilinear",
                                 align_corners=False)
        disagreement_hist.update(disagree.squeeze().cpu().numpy()[valid], labels)

        # Collapse diagnostic, accumulated inline so it is always available
        # rather than needing a separate script run after the fact.
        flat = out["ood_scores"].flatten(2)
        raw_sums += flat.sum(dim=(0, 2)).cpu().numpy()
        raw_sqsums += (flat ** 2).sum(dim=(0, 2)).cpu().numpy()
        raw_counts += flat.shape[0] * flat.shape[2]

        # Separation between anomalous and normal pixels, per head. This --
        # not the global std -- is what "are the heads discriminating?"
        # actually means. At a 0.24% positive rate a HEALTHY detector has a
        # low global std, because almost every pixel is a correctly-near-zero
        # negative, so std alone flags a working model as collapsed.
        is_pos = labels == 1
        if is_pos.any():
            pos_sums += per_head_valid[:, is_pos].sum(axis=1)
            pos_counts += int(is_pos.sum())
        if (~is_pos).any():
            neg_sums += per_head_valid[:, ~is_pos].sum(axis=1)
            neg_counts += int((~is_pos).sum())

    metrics = fused_hist.summary()
    for h in range(num_heads):
        metrics[f"auroc_head{h}"] = head_hists[h].auroc()

    # Does head disagreement on its own separate anomalies? This is the
    # project's core uncertainty claim, so it gets measured, not assumed.
    metrics["auroc_disagreement"] = disagreement_hist.auroc()

    means = raw_sums / max(raw_counts, 1)
    stds = np.sqrt(np.maximum(raw_sqsums / max(raw_counts, 1) - means ** 2, 0))
    pos_means = pos_sums / max(pos_counts, 1)
    neg_means = neg_sums / max(neg_counts, 1)
    separations = pos_means - neg_means

    for h in range(num_heads):
        metrics[f"head{h}_out_mean"] = float(means[h])
        metrics[f"head{h}_out_std"] = float(stds[h])
        metrics[f"head{h}_pos_mean"] = float(pos_means[h])
        metrics[f"head{h}_neg_mean"] = float(neg_means[h])
        metrics[f"head{h}_separation"] = float(separations[h])
    metrics["min_head_std"] = float(stds.min())
    metrics["min_head_separation"] = float(separations.min())

    return metrics


def is_collapsed(metrics, num_heads):
    """Collapse means the heads have stopped telling anomalous pixels apart
    from normal ones -- NOT merely that their outputs are concentrated.

    An earlier version tested global output std alone, and fired on a model
    scoring 0.99 AUROC. At a 0.24% positive rate almost every pixel is a
    correctly-near-zero negative, so a healthy detector necessarily has a low
    global std. The test that means something is whether the score
    distribution differs between the two classes at all, cross-checked
    against ranking performance.
    """
    separated = metrics["min_head_separation"] >= 0.01
    ranks = metrics["auroc"] >= 0.70
    return not (separated or ranks)


@torch.no_grad()
def evaluate_miou(model, dataset, device, max_images):
    """Cityscapes-val mIoU -- the 'did the OOD heads break normal
    segmentation?' check named in the plan but previously not computed."""
    model.eval()
    cm = ConfusionMatrix(num_classes=config.NUM_SEG_CLASSES, ignore_index=255)
    n = min(max_images, len(dataset))
    stride = max(1, len(dataset) // n)

    for i in range(0, stride * n, stride):
        image_t, label_t = dataset[i]
        with amp_context(device):
            out = model(image_t.unsqueeze(0).to(device))
        pred = out["seg_logits"].float().argmax(dim=1).squeeze(0).cpu().numpy()
        cm.update(pred, label_t.numpy())
    return cm.miou()


def main():
    torch.manual_seed(config.GLOBAL_SEED)
    np.random.seed(config.GLOBAL_SEED)
    device = get_device()

    num_heads = 3
    model = TwinGuardModel(num_ood_heads=num_heads,
                           ood_seeds=config.OOD_HEAD_SEEDS_3HEAD).to(device)
    assert verify_encoder_frozen(model), "encoder must be fully frozen"
    assert verify_heads_independent(model), "OOD heads must be independently initialised"
    print(f"pre-flight: encoder frozen, heads independent, encoder={config.ENCODER_NAME}")

    # normalize=False: CutMix composites in pixel space and normalises once
    # at the end, so the base loader must not normalise first.
    train_base = CityscapesDataset(split="train", normalize=False)
    train_dataset = CutMixAugmentedDataset(train_base, p=config.CUTMIX_PROB)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
        pin_memory=device.type == "cuda")
    print(f"dataloader: {len(train_loader)} steps/epoch, "
          f"num_workers={config.NUM_WORKERS}")

    val_dataset = CityscapesDataset(split="val", normalize=True)

    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    fishy_val, fishy_test = split_fishyscapes_pairs(all_pairs)
    print(f"fishyscapes: {len(fishy_val)} val (selection) / {len(fishy_test)} test (reported)")

    seg_criterion = build_seg_criterion().to(device)
    ood_head_lr = (config.OOD_HEAD_LEARNING_RATE if config.USE_OOD_HEAD_LR_SPLIT
                   else config.LEARNING_RATE)
    optimizer = torch.optim.AdamW([
        {"params": model.seg_head.parameters(), "lr": config.LEARNING_RATE},
        {"params": model.ood_heads.parameters(), "lr": ood_head_lr},
    ])

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    metric_key = config.SELECTION_METRIC
    best_val_score = -1.0
    best_epoch = -1
    selected_test = {}
    print(f"checkpoint selection: best val {metric_key} (val half only)")

    with mlflow.start_run(run_name="experiment_b_3head"):
        mlflow.log_params({
            "num_ood_heads": num_heads,
            "seeds": config.OOD_HEAD_SEEDS_3HEAD,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "num_workers": config.NUM_WORKERS,
            "lr": config.LEARNING_RATE,
            "ood_head_lr": ood_head_lr,
            "use_ood_head_lr_split": config.USE_OOD_HEAD_LR_SPLIT,
            "encoder": config.ENCODER_NAME,
            "anomaly_source": config.ANOMALY_SOURCE,
            "cutmix_prob": config.CUTMIX_PROB,
            "cutmix_scale_min": config.CUTMIX_SCALE_MIN,
            "cutmix_scale_max": config.CUTMIX_SCALE_MAX,
            "cutmix_max_objects": config.CUTMIX_MAX_OBJECTS,
            "cutmix_surface_constrained": True,
            "cutmix_harmonize": config.CUTMIX_HARMONIZE,
            "ood_pos_weight": config.OOD_POS_WEIGHT,
            "ood_loss": "bce_with_logits",
            "input_normalized": True,
            "alpha_ood": config.ALPHA_OOD,
            "precalibration_auroc_gate": config.PRECALIBRATION_AUROC_GATE,
            "postcalibration_auroc_target": config.POSTCALIBRATION_AUROC_TARGET,
            "selection_metric": metric_key,
        })
        mlflow.set_tag("calibration_note", config.CALIBRATION_TRADEOFF_NOTE)
        mlflow.set_tag("selection_protocol",
                       "checkpoint selected on fishyscapes val half only; "
                       "test half reported and never used for selection")

        for epoch in range(config.EPOCHS):
            model.train()
            model.encoder.eval()  # frozen encoder stays deterministic always

            running_loss, running_seg, running_ood = 0.0, 0.0, 0.0
            running_pos_rate = 0.0
            t0 = time.time()
            n_steps = len(train_loader)

            for step, (images, seg_labels, ood_target) in enumerate(train_loader):
                images = images.to(device)
                seg_labels = seg_labels.to(device)
                ood_target = ood_target.to(device)

                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    out = model(images)
                    loss, l_seg, l_ood = compute_total_loss(
                        out["seg_logits"].float(), seg_labels,
                        out["ood_logits"].float(), ood_target, seg_criterion)
                # No GradScaler: bf16 keeps fp32's exponent range, so the
                # underflow that fp16 needs scaling for does not occur.
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_seg += l_seg.item()
                running_ood += l_ood.item()
                running_pos_rate += ood_target.mean().item()

                if step % 100 == 0:
                    print(f"epoch {epoch + 1}/{config.EPOCHS} step {step}/{n_steps} "
                          f"loss={loss.item():.4f} (seg={l_seg.item():.4f} "
                          f"ood={l_ood.item():.4f})")

            train_time = time.time() - t0

            val_metrics = evaluate_ood(model, fishy_val, device, num_heads)
            miou = evaluate_miou(model, val_dataset, device, config.MIOU_EVAL_IMAGES)

            improved = val_metrics[metric_key] > best_val_score
            last_epoch = epoch == config.EPOCHS - 1
            # Scoring the test half every epoch costs ~25% of epoch time and
            # only the selected checkpoint's numbers are ever reported.
            want_test = (improved or last_epoch
                         or not config.EVAL_TEST_ON_IMPROVEMENT_ONLY)
            test_metrics = (evaluate_ood(model, fishy_test, device, num_heads)
                            if want_test else None)

            line = (f"epoch {epoch + 1}/{config.EPOCHS} done in {train_time:.1f}s -- "
                    f"loss={running_loss / n_steps:.4f} | "
                    f"VAL {metric_key}={val_metrics[metric_key]:.4f} "
                    f"auroc={val_metrics['auroc']:.4f}")
            if test_metrics:
                line += (f" | TEST auroc={test_metrics['auroc']:.4f} "
                         f"ap={test_metrics['ap']:.4f} ece={test_metrics['ece']:.4f} "
                         f"fpr95={test_metrics['fpr95']:.4f}")
            line += f" | mIoU={miou:.4f}"
            print(line)

            print("  head separation (mean score on anomalous - on normal): " +
                  "  ".join(f"h{h}={val_metrics[f'head{h}_separation']:+.4f}"
                            for h in range(num_heads)))
            if is_collapsed(val_metrics, num_heads):
                print("  WARNING: heads are NOT separating anomalous from normal "
                      "pixels (separation < 0.01 and AUROC < 0.70) -- collapse. "
                      "More epochs will not fix this.")

            mlflow.log_metrics({
                "train_loss": running_loss / n_steps,
                "train_l_seg": running_seg / n_steps,
                "train_l_ood": running_ood / n_steps,
                "train_ood_pos_rate": running_pos_rate / n_steps,
                "miou": miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **({f"test_{k}": v for k, v in test_metrics.items()}
                   if test_metrics else {}),
                # Unprefixed aliases so existing dashboards keep resolving;
                # these are the REPORTED (test-half) numbers.
                **({"auroc": test_metrics["auroc"], "ap": test_metrics["ap"],
                    "ece": test_metrics["ece"], "fpr95": test_metrics["fpr95"]}
                   if test_metrics else {}),
            }, step=epoch)

            if improved:
                best_val_score = val_metrics[metric_key]
                best_epoch = epoch
                selected_test = dict(test_metrics) if test_metrics else {}
                torch.save(model.state_dict(), config.CHECKPOINT_3HEAD)
                suffix = (f" (test {metric_key} {test_metrics[metric_key]:.4f})"
                          if test_metrics else "")
                print(f"  -> new best VAL {metric_key} {best_val_score:.4f}"
                      f"{suffix}, checkpoint saved")

        mlflow.log_artifact(config.CHECKPOINT_3HEAD)
        # The headline numbers must describe the checkpoint that was actually
        # saved, not whatever the last epoch happened to score. Logging only
        # last-epoch values means the summary table and the .pth file on disk
        # describe different models.
        mlflow.log_metrics({
            f"best_val_{metric_key}": best_val_score,
            "best_epoch": best_epoch,
            **{f"selected_{k}": v for k, v in selected_test.items()},
        })

    print(f"\ntraining done. selected epoch {best_epoch + 1} "
          f"(best val {metric_key}={best_val_score:.4f})")
    if selected_test:
        print(f"  SELECTED CHECKPOINT on the test half:")
        print(f"    AUROC={selected_test['auroc']:.4f}  AP={selected_test['ap']:.4f}  "
              f"FPR@95={selected_test['fpr95']:.4f}  ECE={selected_test['ece']:.4f}")
        print(f"    head-disagreement AUROC={selected_test['auroc_disagreement']:.4f} "
              f"(the ensemble uncertainty signal scored on its own)")
        print(f"  These are the numbers to quote -- they describe the .pth that "
              f"was saved, not the last epoch.")
    print(f"gate: {config.PRECALIBRATION_AUROC_GATE} pre-calibration, "
          f"{config.POSTCALIBRATION_AUROC_TARGET} post-calibration target")


if __name__ == "__main__":
    main()

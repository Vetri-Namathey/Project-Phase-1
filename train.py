"""Experiment B (Section 12): train TwinGuard with 3 independently-seeded
OOD heads on Cityscapes + CutMix, evaluate on the same Fishyscapes set as
Experiment A each epoch, log to MLflow, save the best checkpoint by AUROC.
"""

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
from data.fishyscapes_dataset import list_fishyscapes_pairs
from losses import build_seg_criterion, compute_total_loss
from metrics import compute_auroc, compute_ece, compute_fpr_at_tpr
from model.twinguard_model import TwinGuardModel, verify_encoder_frozen, verify_heads_independent


def evaluate(model, fishy_pairs, device):
    model.eval()
    all_scores, all_labels = [], []
    per_head_scores = [[] for _ in range(len(model.ood_heads))]

    with torch.no_grad():
        for image_path, label_path in fishy_pairs:
            image = Image.open(image_path).convert("RGB")
            label_map = np.array(Image.open(label_path))

            image_resized = image.resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
            image_t = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_t = image_t.to(device)

            out = model(image_t)

            fused = F.interpolate(
                out["ood_fused"].unsqueeze(1), size=label_map.shape, mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

            valid = label_map != 255
            all_scores.append(fused[valid])
            all_labels.append(label_map[valid])

            per_head = F.interpolate(
                out["ood_scores"], size=label_map.shape, mode="bilinear", align_corners=False
            ).squeeze(0).cpu().numpy()  # (num_heads, H, W)
            for h in range(per_head.shape[0]):
                per_head_scores[h].append(per_head[h][valid])

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    metrics = {
        "auroc": compute_auroc(scores, labels),
        "ece": compute_ece(scores, labels),
        "fpr95": compute_fpr_at_tpr(scores, labels),
    }
    for h in range(len(per_head_scores)):
        metrics[f"auroc_head{h}"] = compute_auroc(np.concatenate(per_head_scores[h]), labels)

    return metrics


def main():
    torch.manual_seed(config.GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD).to(device)
    assert verify_encoder_frozen(model), "encoder must be fully frozen (Section 02)"
    assert verify_heads_independent(model), "OOD heads must be independently initialised (Section 02)"
    print("pre-flight checks passed: encoder frozen, heads independent")

    train_base = CityscapesDataset(split="train")
    train_dataset = CutMixAugmentedDataset(train_base, p=config.CUTMIX_PROB)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    fishy_pairs = list_fishyscapes_pairs()
    assert len(fishy_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(fishy_pairs)}"

    seg_criterion = build_seg_criterion().to(device)
    # USE_OOD_HEAD_LR_SPLIT gates whether the OOD heads get a separate, 10x
    # lower LR (Checkpoint B's variable) or share the flat LEARNING_RATE with
    # seg_head (Checkpoint A -- isolates the backbone swap alone). Toggle in
    # config.py, not here, so the two checkpoints stay a clean, reportable
    # ablation instead of silently blurring together.
    ood_head_lr = config.OOD_HEAD_LEARNING_RATE if config.USE_OOD_HEAD_LR_SPLIT else config.LEARNING_RATE
    optimizer = torch.optim.AdamW([
        {"params": model.seg_head.parameters(), "lr": config.LEARNING_RATE},
        {"params": model.ood_heads.parameters(), "lr": ood_head_lr},
    ])

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    best_auroc = -1.0

    with mlflow.start_run(run_name="experiment_b_3head"):
        mlflow.log_params({
            "num_ood_heads": 3,
            "seeds": config.OOD_HEAD_SEEDS_3HEAD,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "ood_head_lr": ood_head_lr,
            "use_ood_head_lr_split": config.USE_OOD_HEAD_LR_SPLIT,
            "encoder": config.ENCODER_NAME,
            "cutmix_prob": config.CUTMIX_PROB,
            "cutmix_scale_min": config.CUTMIX_SCALE_MIN,
            "cutmix_scale_max": config.CUTMIX_SCALE_MAX,
            "alpha_ood": config.ALPHA_OOD,
            "precalibration_auroc_gate": config.PRECALIBRATION_AUROC_GATE,
            "postcalibration_auroc_target": config.POSTCALIBRATION_AUROC_TARGET,
        })
        # Baked into every run's metadata, not just source comments -- so the
        # 0.75/0.83 distinction survives even if someone only ever looks at
        # the MLflow UI and never opens config.py.
        mlflow.set_tag("calibration_note", config.CALIBRATION_TRADEOFF_NOTE)

        for epoch in range(config.EPOCHS):
            model.train()
            model.encoder.eval()  # frozen encoder stays deterministic always

            running_loss, running_seg, running_ood = 0.0, 0.0, 0.0
            t0 = time.time()
            n_steps = len(train_loader)

            for step, (images, seg_labels, ood_target) in enumerate(train_loader):
                images = images.to(device)
                seg_labels = seg_labels.to(device)
                ood_target = ood_target.to(device)

                optimizer.zero_grad()
                out = model(images)
                loss, l_seg, l_ood = compute_total_loss(
                    out["seg_logits"], seg_labels, out["ood_scores"], ood_target, seg_criterion
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_seg += l_seg.item()
                running_ood += l_ood.item()

                if step % 100 == 0:
                    print(f"epoch {epoch + 1}/{config.EPOCHS} step {step}/{n_steps} loss={loss.item():.4f}")

            train_time = time.time() - t0
            eval_metrics = evaluate(model, fishy_pairs, device)

            print(
                f"epoch {epoch + 1}/{config.EPOCHS} done in {train_time:.1f}s -- "
                f"train_loss={running_loss / n_steps:.4f} auroc={eval_metrics['auroc']:.4f} "
                f"ece={eval_metrics['ece']:.4f} fpr95={eval_metrics['fpr95']:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": running_loss / n_steps,
                "train_l_seg": running_seg / n_steps,
                "train_l_ood": running_ood / n_steps,
                **eval_metrics,
            }, step=epoch)

            if eval_metrics["auroc"] > best_auroc:
                best_auroc = eval_metrics["auroc"]
                torch.save(model.state_dict(), config.CHECKPOINT_3HEAD)
                mlflow.log_artifact(config.CHECKPOINT_3HEAD)
                print(f"  -> new best AUROC {best_auroc:.4f}, checkpoint saved")

    print(f"\ntraining done. best AUROC={best_auroc:.4f}")


if __name__ == "__main__":
    main()

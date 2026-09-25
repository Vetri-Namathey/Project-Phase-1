"""Phase 2b: L_calib.

Three things this script does, in order, matching PLAN.md's "Calibration
(L_calib) gate" and "Proof-of-change outputs" sections:

1. Temperature scaling -- a single scalar T fit on the Fishyscapes val half.
   This is the BASELINE L_calib has to beat, not a step towards L_calib. A
   scalar rescale cannot reshape calibration spatially, which is the whole
   point (see losses.py's SoftECELoss docstring).
2. L_calib joint fine-tune -- continues training an already-converged
   checkpoint with a differentiable ECE-surrogate loss added to the existing
   seg+OOD loss. Selects on val ECE, but restarts if val AUROC drops more
   than config.CALIB_AUROC_DROP_LIMIT (a SMALL, accounted-for cost is
   expected per CALIBRATION_TRADEOFF_NOTE -- an open-ended one is not).
3. Comparison table + reliability diagram -- raw vs. temp-scaled vs.
   L_calib, on AUROC/AP/FPR@95/ECE (Fishyscapes test half), logged to MLflow
   and saved as a plot.

Whole-image ECE only. Boundary-only ECE (Novelty 6) and UBQ are separate,
still-unbuilt metrics (see PLAN.md's UBQ section) -- this comparison alone
cannot prove L_calib beats temperature scaling on the actual spatial claim
that is the point of building it. Said explicitly in the table's own output,
not left implicit.

Usage:
    python calibrate.py               # all three steps
    python calibrate.py --temp-only   # step 1 only, quick sanity check
    python calibrate.py --checkpoint checkpoints/some_other.pth
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from losses import SoftECELoss, build_seg_criterion, compute_total_loss, ood_bce_loss
from metrics import ScoreHistogram
from train import amp_context
from utils import get_device, load_trained_model


def _cache_logits(model, pairs, device):
    """One forward pass per image, cached -- reused across every temperature
    -fit epoch without re-running the (frozen) encoder each time."""
    cached = []
    model.eval()
    with torch.no_grad():
        for image_path, label_path in pairs:
            label_map = np.array(Image.open(label_path))
            valid = label_map != 255
            with amp_context(device):
                out = model(load_image_tensor(image_path, device))
            logits = F.interpolate(out["ood_logits"].float(), size=label_map.shape,
                                   mode="bilinear", align_corners=False)
            num_heads = logits.shape[1]
            logits_flat = logits.squeeze(0).reshape(num_heads, -1).cpu()
            valid_flat = torch.from_numpy(valid.reshape(-1))
            target = torch.from_numpy((label_map[valid] == 1).astype(np.float32))
            cached.append((logits_flat[:, valid_flat], target))
    return cached


def fit_temperature(model, val_pairs, device):
    """Single scalar T minimizing the SAME per-head BCE the model was
    trained with, on logit/T instead of raw logits, over the Fishyscapes val
    half. This is the temperature-scaling BASELINE -- see this file's
    module docstring for why it is not L_calib itself.
    """
    print("fitting temperature scaling baseline...")
    cached = _cache_logits(model, val_pairs, device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.Adam([temperature], lr=config.TEMPERATURE_LEARNING_RATE)

    for epoch in range(config.TEMPERATURE_EPOCHS):
        total_loss = 0.0
        for logits_valid, target in cached:
            logits_valid = logits_valid.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            num_heads = logits_valid.shape[0]
            loss = torch.stack([
                ood_bce_loss(logits_valid[h] / temperature, target)
                for h in range(num_heads)
            ]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{config.TEMPERATURE_EPOCHS}: "
              f"T={temperature.item():.4f}  loss={total_loss / len(cached):.4f}")
    return float(temperature.detach().item())


@torch.no_grad()
def evaluate_fused(model, pairs, device, temperature=1.0):
    """Fused-score AUROC/AP/FPR@95/ECE on a Fishyscapes split, with an
    optional post-hoc temperature applied to each head's logit BEFORE fusing
    -- the same mean-of-sigmoids fusion the model itself uses, just fed
    logit/T instead of the raw logit when temperature != 1.0.
    """
    model.eval()
    hist = ScoreHistogram()
    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid = label_map != 255
        labels = label_map[valid]

        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=label_map.shape,
                               mode="bilinear", align_corners=False)
        scores = torch.sigmoid(logits / temperature).mean(dim=1)
        hist.update(scores.squeeze(0).cpu().numpy()[valid], labels)
    return hist


def run_calib_finetune(base_checkpoint, device):
    """Continues training an already-converged checkpoint with SoftECELoss
    added to the existing loss. Selection reads val ECE (what this fine-tune
    actually exists to fix), guarded by the AUROC-drop limit -- restart, do
    not save, if the drop exceeds what CALIBRATION_TRADEOFF_NOTE accounts for.
    """
    model = load_trained_model(base_checkpoint, device)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)  # frozen invariant holds through Phase 2b too

    train_base = CityscapesDataset(split="train", normalize=False)
    train_dataset = CutMixAugmentedDataset(train_base, p=config.CUTMIX_PROB)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
        pin_memory=device.type == "cuda")
    print(f"calib dataloader: {len(train_loader)} steps/epoch, "
          f"num_workers={config.NUM_WORKERS}")

    all_pairs = list_fishyscapes_pairs()
    fishy_val, _ = split_fishyscapes_pairs(all_pairs)

    seg_criterion = build_seg_criterion().to(device)
    soft_ece = SoftECELoss().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.seg_head.parameters(), "lr": config.CALIB_LEARNING_RATE},
        {"params": model.ood_heads.parameters(), "lr": config.CALIB_LEARNING_RATE},
    ])

    start_hist = evaluate_fused(model, fishy_val, device)
    start_auroc = start_hist.auroc()
    print(f"base checkpoint val AUROC: {start_auroc:.4f} (fine-tune stops if "
          f"this drops by more than {config.CALIB_AUROC_DROP_LIMIT})")

    best_score, best_epoch = -1.0, -1
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.CALIB_EPOCHS):
        model.train()
        model.encoder.eval()
        running_total, running_calib = 0.0, 0.0
        n_steps = len(train_loader)

        for step, (images, seg_labels, ood_target) in enumerate(train_loader):
            images = images.to(device)
            seg_labels = seg_labels.to(device)
            ood_target = ood_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                out = model(images)
                base_loss, l_seg, l_ood = compute_total_loss(
                    out["seg_logits"].float(), seg_labels,
                    out["ood_logits"].float(), ood_target, seg_criterion)
                l_calib = soft_ece(out["ood_fused"].float(), ood_target)
                loss = base_loss + config.BETA_CALIB * l_calib
            loss.backward()
            optimizer.step()

            running_total += loss.item()
            running_calib += l_calib.item()
            if step % 100 == 0:
                print(f"  calib epoch {epoch + 1}/{config.CALIB_EPOCHS} "
                      f"step {step}/{n_steps} loss={loss.item():.4f} "
                      f"(base={base_loss.item():.4f} calib={l_calib.item():.4f})")

        val_hist = evaluate_fused(model, fishy_val, device)
        val_auroc, val_ece = val_hist.auroc(), val_hist.ece()
        auroc_drop = start_auroc - val_auroc
        print(f"calib epoch {epoch + 1}/{config.CALIB_EPOCHS} done -- "
              f"val AUROC={val_auroc:.4f} (drop {auroc_drop:+.4f}) "
              f"val ECE={val_ece:.4f} avg_calib_loss={running_calib / n_steps:.4f}")

        if auroc_drop > config.CALIB_AUROC_DROP_LIMIT:
            print(f"  STOPPING: AUROC dropped {auroc_drop:.4f} > "
                  f"{config.CALIB_AUROC_DROP_LIMIT} limit -- per "
                  f"CALIBRATION_TRADEOFF_NOTE this is no longer an "
                  f"accounted-for cost. This epoch's weights are NOT saved.")
            break

        score = -val_ece
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), config.CHECKPOINT_3HEAD_CALIB)
            print(f"  -> new best val ECE {val_ece:.4f}, checkpoint saved")

    if best_epoch == -1:
        raise SystemExit(
            "no epoch improved val ECE within the AUROC-drop budget -- no "
            "calibrated checkpoint was saved. See the CALIB_* settings in "
            "config.py (BETA_CALIB, CALIB_EPOCHS, CALIB_AUROC_DROP_LIMIT)."
        )
    print(f"L_calib fine-tune done. selected epoch {best_epoch + 1}")
    return config.CHECKPOINT_3HEAD_CALIB


def comparison_table(raw_model, temperature, calib_model, fishy_test, device):
    """PLAN.md's 'Comparison table' proof-of-change artifact."""
    raw_hist = evaluate_fused(raw_model, fishy_test, device, temperature=1.0)
    temp_hist = evaluate_fused(raw_model, fishy_test, device, temperature=temperature)
    calib_hist = evaluate_fused(calib_model, fishy_test, device, temperature=1.0)
    rows = [("raw", raw_hist), ("temp-scaled", temp_hist), ("L_calib", calib_hist)]

    header = f"{'model':<14} {'AUROC':>7} {'AP':>7} {'FPR@95':>7} {'ECE':>7}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, hist in rows:
        m = hist.summary()
        print(f"{name:<14} {m['auroc']:>7.4f} {m['ap']:>7.4f} "
              f"{m['fpr95']:>7.4f} {m['ece']:>7.4f}")
        results[name] = m

    print(f"\ntemperature (fitted): {temperature:.4f}")
    print(config.CALIBRATION_TRADEOFF_NOTE)
    print("Note: whole-image ECE only. Boundary-only ECE (Novelty 6) and UBQ "
          "are separate, still-unbuilt metrics -- this table alone cannot "
          "prove L_calib beats temperature scaling on the spatial claim that "
          "is the actual point of building it (see PLAN.md).")
    return results, dict(rows)


def reliability_diagram(hists, out_path="calibration_reliability.png"):
    """Three curves -- raw, temp-scaled, L_calib -- confidence vs. actual
    accuracy, binned. Built from the same per-bin data ScoreHistogram.ece()
    uses internally, so it can never show something the reported ECE numbers
    disagree with.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    colors = {"raw": "#3D5AFE", "temp-scaled": "#4472C4", "L_calib": "#2E8B57"}
    for name, hist in hists.items():
        conf, acc, weight = hist.reliability_curve()
        mask = weight > 0
        ax.plot(conf[mask], acc[mask], marker="o", label=name,
               color=colors.get(name))
    ax.set_xlabel("mean predicted score (confidence)")
    ax.set_ylabel("empirical positive rate (accuracy)")
    ax.set_title("Reliability diagram -- Fishyscapes test half")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-only", action="store_true",
                        help="fit + evaluate temperature scaling only, skip the L_calib fine-tune")
    parser.add_argument("--checkpoint", default=None,
                        help="base checkpoint to calibrate (default: config.CHECKPOINT_3HEAD)")
    args = parser.parse_args()

    device = get_device()
    base_checkpoint = args.checkpoint or config.CHECKPOINT_3HEAD
    raw_model = load_trained_model(base_checkpoint, device)

    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    fishy_val, fishy_test = split_fishyscapes_pairs(all_pairs)

    temperature = fit_temperature(raw_model, fishy_val, device)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    if args.temp_only:
        with mlflow.start_run(run_name="temperature_scaling_only"):
            mlflow.log_params({"base_checkpoint": base_checkpoint,
                               "temperature": temperature})
            hist = evaluate_fused(raw_model, fishy_test, device, temperature=temperature)
            m = hist.summary()
            mlflow.log_metrics(m)
            print(f"temperature-scaled test: AUROC={m['auroc']:.4f} "
                  f"AP={m['ap']:.4f} FPR@95={m['fpr95']:.4f} ECE={m['ece']:.4f}")
        return

    with mlflow.start_run(run_name="l_calib_finetune"):
        mlflow.log_params({
            "base_checkpoint": base_checkpoint,
            "beta_calib": config.BETA_CALIB,
            "calib_epochs": config.CALIB_EPOCHS,
            "calib_lr": config.CALIB_LEARNING_RATE,
            "auroc_drop_limit": config.CALIB_AUROC_DROP_LIMIT,
            "fitted_temperature": temperature,
        })
        calib_checkpoint = run_calib_finetune(base_checkpoint, device)
        calib_model = load_trained_model(calib_checkpoint, device)

        results, hists = comparison_table(raw_model, temperature, calib_model,
                                          fishy_test, device)
        mlflow.log_metrics({f"raw_{k}": v for k, v in results["raw"].items()})
        mlflow.log_metrics({f"temp_{k}": v for k, v in results["temp-scaled"].items()})
        mlflow.log_metrics({f"calib_{k}": v for k, v in results["L_calib"].items()})

        reliability_diagram(hists)
        mlflow.log_artifact("calibration_reliability.png")
        mlflow.log_artifact(calib_checkpoint)


if __name__ == "__main__":
    main()

"""Phase 2b: L_calib, and the temperature-scaling baseline it has to beat.

Three steps, in order:

1. Temperature scaling -- one scalar T, fitted on the Fishyscapes VAL half
   by minimising plain binary NLL. Fitted twice: on all valid pixels
   ("whole") and on the boundary band only ("band"). This is the BASELINE,
   not a step towards L_calib: a scalar rescale stretches every score by the
   same factor and cannot reshape calibration near object edges.
2. L_calib joint fine-tune -- continues training a converged checkpoint
   with beta * SoftECELoss added to the existing seg + OOD loss, computed on
   the band around each pasted object's edge (config.CALIB_LOSS_REGION),
   with sensor degradations switched on. An epoch is saved only if it beats
   the STARTING checkpoint's val ECE and stays within the AUROC and AP drop
   limits -- "the fine-tune ran" is not the same as "it helped".
3. Comparison table + reliability diagram on the Fishyscapes TEST half:
   raw vs. temp(whole) vs. temp(band) vs. L_calib, whole-image and band.

eval_spatial.py adds r=4/8/16 bands, UBQ and paired bootstrap CIs on top.

Usage:
    python calibrate.py --checkpoint <raw.pth>              # all three steps
    python calibrate.py --checkpoint <raw.pth> --temp-only  # step 1 + test eval
    python calibrate.py --checkpoint <raw.pth> --out <calib.pth>
"""

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

import config
from data.cityscapes_dataset import CityscapesDataset
from data.cutmix import CutMixAugmentedDataset
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from losses import SoftECELoss, boundary_band_torch, build_seg_criterion, compute_total_loss
from metrics import ScoreHistogram, boundary_band
from train import amp_context
from utils import get_device, load_trained_model

REGIONS = ("whole", "band")


def _load_fishyscapes_label(label_path, radius):
    """-> (anomaly bool, valid bool, band bool), all (H,W)."""
    label_map = np.array(Image.open(label_path))
    valid = label_map != 255
    anomaly = label_map == 1
    return anomaly, valid, boundary_band(anomaly, radius_px=radius, valid=valid)


@torch.no_grad()
def _cache_val_logits(model, pairs, device, radius=config.CALIB_BAND_RADIUS_PX):
    """One forward pass per val image. Keeps per-head logits over valid
    pixels plus which of them fall in the band, so every temperature the
    1-D search tries is scored without re-running the model."""
    cached = []
    model.eval()
    for image_path, label_path in pairs:
        anomaly, valid, band = _load_fishyscapes_label(label_path, radius)
        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=anomaly.shape,
                               mode="bilinear", align_corners=False)
        logits = logits.squeeze(0)[:, torch.from_numpy(valid).to(device)].cpu()
        cached.append((logits,
                       torch.from_numpy(anomaly[valid].astype(np.float32)),
                       torch.from_numpy(band[valid])))
    return cached


def _mean_nll(cached, temperature, region):
    """Plain (unweighted) binary NLL of sigmoid(logit / T), averaged over
    every head and every pixel in the region."""
    total, count = 0.0, 0
    for logits, target, band in cached:
        if region == "band":
            logits, target = logits[:, band], target[band]
        if target.numel() == 0:
            continue
        total += F.binary_cross_entropy_with_logits(
            logits / temperature, target.expand_as(logits), reduction="sum").item()
        count += logits.numel()
    return total / count


def fit_temperatures(model, val_pairs, device):
    """Fits T separately for each region in REGIONS. Returns {region: T}.

    Plain NLL, NOT the training loss: the training loss carries
    pos_weight=20, which rewards pushing scores UP -- fitting T against it
    makes calibration worse on purpose (measured: 10x worse ECE, see the
    TEMPERATURE_LOG_BOUNDS comment in config.py).
    """
    print("fitting temperature scaling baselines (plain NLL, Fishyscapes val half)...")
    cached = _cache_val_logits(model, val_pairs, device)
    lo, hi = config.TEMPERATURE_LOG_BOUNDS
    temperatures = {}
    for region in REGIONS:
        res = minimize_scalar(lambda log_t: _mean_nll(cached, float(np.exp(log_t)), region),
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        t = float(np.exp(res.x))
        if abs(res.x - lo) < 1e-3 or abs(res.x - hi) < 1e-3:
            print(f"  WARNING: {region} T={t:.4f} hit the search bound -- widen "
                  f"config.TEMPERATURE_LOG_BOUNDS")
        print(f"  {region:<5}  T={t:.4f}  NLL {_mean_nll(cached, 1.0, region):.6f} "
              f"(T=1) -> {res.fun:.6f}  [{res.nfev} evaluations]")
        temperatures[region] = t
    return temperatures


@torch.no_grad()
def evaluate_fused(model, pairs, device, temperature=1.0,
                   radius=config.CALIB_BAND_RADIUS_PX):
    """Fused-score histograms on a Fishyscapes split -> {"whole", "band"}.

    Same order the model uses: upsample logits, per-head sigmoid(logit/T),
    mean across heads.
    """
    model.eval()
    hists = {region: ScoreHistogram() for region in REGIONS}
    for image_path, label_path in pairs:
        anomaly, valid, band = _load_fishyscapes_label(label_path, radius)
        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=anomaly.shape,
                               mode="bilinear", align_corners=False)
        scores = torch.sigmoid(logits / temperature).mean(dim=1).squeeze(0).cpu().numpy()
        labels = anomaly.astype(np.int64)
        hists["whole"].update(scores[valid], labels[valid])
        hists["band"].update(scores[band], labels[band])
    return hists


def _calib_scores_and_targets(ood_logits, ood_target):
    """Pixels L_calib is computed on, per config.CALIB_LOSS_REGION.

    Fused score rebuilt from fp32 logits rather than read from the model's
    bf16 ood_fused, so the loss sees the same numbers the metrics do.
    """
    fused = torch.sigmoid(ood_logits.float()).mean(dim=1)
    if config.CALIB_LOSS_REGION == "whole":
        return fused, ood_target
    band = boundary_band_torch(ood_target, config.CALIB_BAND_RADIUS_PX)
    return fused[band], ood_target[band]


def _band_mean_anchor(ood_logits, base_logits, ood_target):
    """(mean edge-band score now - mean edge-band score of the frozen base)^2.

    Why it exists (measured, runs/RESULTS.md): at object edges the model is
    on average OVER-confident on synthetic pastes but UNDER-confident on real
    anomalies. Unanchored L_calib fixes the pastes by lowering edge scores,
    which moves real edges the wrong way (val band-ECE 0.153 -> 0.29). What
    both domains share is scores that are too extreme. Holding the band's
    average score at the base model's level leaves L_calib only that part to
    change.
    """
    band = boundary_band_torch(ood_target, config.CALIB_BAND_RADIUS_PX)
    if not band.any():
        return ood_logits.new_zeros(())
    cur = torch.sigmoid(ood_logits.float()).mean(dim=1)[band].mean()
    ref = torch.sigmoid(base_logits.float()).mean(dim=1)[band].mean()
    return (cur - ref) ** 2


def run_calib_finetune(base_checkpoint, out_checkpoint, device):
    """Continues training base_checkpoint with beta * L_calib added.

    Saves an epoch only if its val ECE (config.CALIB_SELECTION_REGION) beats
    the best so far, starting from the BASE checkpoint's own val ECE -- so a
    fine-tune that never improves on where it started saves nothing and says
    so. Stops outright if val AUROC or AP fall past their drop limits.
    """
    model = load_trained_model(base_checkpoint, device)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)  # frozen invariant holds through Phase 2b too

    # Frozen copy of the starting heads, for the anchor. The encoder is shared
    # (and frozen), so this costs the heads' ~4M parameters, not a second model.
    base_heads = copy.deepcopy(model.ood_heads).eval()
    for p in base_heads.parameters():
        p.requires_grad_(False)

    train_base = CityscapesDataset(split="train", normalize=False)
    train_dataset = CutMixAugmentedDataset(
        train_base, p=config.CUTMIX_PROB,
        degradation_prob=config.CALIB_DEGRADATION_PROB)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
        pin_memory=device.type == "cuda")
    print(f"calib dataloader: {len(train_loader)} steps/epoch, "
          f"num_workers={config.NUM_WORKERS}, "
          f"degradation p={config.CALIB_DEGRADATION_PROB}, "
          f"L_calib region={config.CALIB_LOSS_REGION}"
          f"{f' (r={config.CALIB_BAND_RADIUS_PX})' if config.CALIB_LOSS_REGION == 'band' else ''}")

    fishy_val, _ = split_fishyscapes_pairs(list_fishyscapes_pairs())

    seg_criterion = build_seg_criterion().to(device)
    soft_ece = SoftECELoss().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.seg_head.parameters(), "lr": config.CALIB_LEARNING_RATE},
        {"params": model.ood_heads.parameters(), "lr": config.CALIB_LEARNING_RATE},
    ])

    sel = config.CALIB_SELECTION_REGION
    start = evaluate_fused(model, fishy_val, device)
    start_auroc = start["whole"].auroc()
    start_ap = start["whole"].average_precision()
    best_ece = start[sel].ece()
    print(f"base checkpoint val: AUROC={start_auroc:.4f} AP={start_ap:.4f} "
          f"ECE={start['whole'].ece():.5f} band-ECE={start['band'].ece():.4f}")
    print(f"an epoch is saved only if val {sel}-ECE < {best_ece:.5f}, with "
          f"AUROC drop <= {config.CALIB_AUROC_DROP_LIMIT} and AP drop <= "
          f"{config.CALIB_AP_DROP_LIMIT}")
    mlflow.log_metrics({"base_val_auroc": start_auroc, "base_val_ap": start_ap,
                        "base_val_ece": start["whole"].ece(),
                        "base_val_band_ece": start["band"].ece()})

    best_epoch = -1
    os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)

    for epoch in range(config.CALIB_EPOCHS):
        model.train()
        model.encoder.eval()
        running = {"total": 0.0, "calib": 0.0, "degraded": 0.0}
        n_steps = len(train_loader)

        accum = config.GRAD_ACCUM_STEPS
        optimizer.zero_grad(set_to_none=True)
        for step, (images, seg_labels, ood_target, degraded) in enumerate(train_loader):
            images = images.to(device)
            seg_labels = seg_labels.to(device)
            ood_target = ood_target.to(device)

            size = images.shape[-2:]
            with amp_context(device):
                hidden = model.encode(images)          # frozen, no grad
                seg_logits = model.seg_head(hidden, size)
                ood_logits = torch.stack([h(hidden, size) for h in model.ood_heads], dim=1)
                with torch.no_grad():
                    base_logits = torch.stack([h(hidden, size) for h in base_heads], dim=1)
            base_loss, _, _ = compute_total_loss(
                seg_logits.float(), seg_labels,
                ood_logits.float(), ood_target, seg_criterion)
            l_calib = soft_ece(*_calib_scores_and_targets(ood_logits, ood_target))
            l_anchor = _band_mean_anchor(ood_logits, base_logits, ood_target)
            loss = (base_loss + config.BETA_CALIB * l_calib
                    + config.CALIB_ANCHOR_WEIGHT * l_anchor)
            (loss / accum).backward()   # see train.py: effective batch = BATCH_SIZE * accum
            if (step + 1) % accum == 0 or step + 1 == n_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running["total"] += loss.item()
            running["calib"] += l_calib.item()
            running["degraded"] += degraded.mean().item()
            if step % 100 == 0:
                print(f"  calib epoch {epoch + 1}/{config.CALIB_EPOCHS} "
                      f"step {step}/{n_steps} loss={loss.item():.4f} "
                      f"(base={base_loss.item():.4f} calib={l_calib.item():.5f} "
                      f"anchor={l_anchor.item():.6f})")

        val = evaluate_fused(model, fishy_val, device)
        val_auroc = val["whole"].auroc()
        val_ap = val["whole"].average_precision()
        val_ece = val[sel].ece()
        auroc_drop, ap_drop = start_auroc - val_auroc, start_ap - val_ap
        print(f"calib epoch {epoch + 1}/{config.CALIB_EPOCHS} done -- "
              f"val AUROC={val_auroc:.4f} ({-auroc_drop:+.4f}) "
              f"AP={val_ap:.4f} ({-ap_drop:+.4f}) "
              f"ECE={val['whole'].ece():.5f} band-ECE={val['band'].ece():.4f} "
              f"avg L_calib={running['calib'] / n_steps:.5f} "
              f"degraded={running['degraded'] / n_steps:.0%}")
        mlflow.log_metrics({
            "calib_train_loss": running["total"] / n_steps,
            "calib_train_l_calib": running["calib"] / n_steps,
            "calib_train_degraded_frac": running["degraded"] / n_steps,
            "calib_val_auroc": val_auroc, "calib_val_ap": val_ap,
            "calib_val_ece": val["whole"].ece(),
            "calib_val_band_ece": val["band"].ece(),
        }, step=epoch)

        if auroc_drop > config.CALIB_AUROC_DROP_LIMIT or ap_drop > config.CALIB_AP_DROP_LIMIT:
            print(f"  STOPPING: detection fell past its limit (AUROC -{auroc_drop:.4f}, "
                  f"AP -{ap_drop:.4f}). Per CALIBRATION_TRADEOFF_NOTE this is no "
                  f"longer an accounted-for cost. This epoch is NOT saved.")
            break

        if val_ece < best_ece:
            best_ece, best_epoch = val_ece, epoch
            torch.save(model.state_dict(), out_checkpoint)
            print(f"  -> new best val {sel}-ECE {val_ece:.5f}, saved {out_checkpoint}")

    if best_epoch == -1:
        raise SystemExit(
            f"no epoch beat the base checkpoint's val {sel}-ECE within the "
            f"AUROC/AP limits -- nothing saved. That is a result, not a crash: "
            f"L_calib did not improve on the starting point with these settings "
            f"(see the CALIB_* block in config.py).")
    print(f"L_calib fine-tune done. selected epoch {best_epoch + 1}")
    return out_checkpoint


def comparison_table(raw_model, temperatures, calib_model, fishy_test, device):
    """Test-half table. calib_model=None skips the L_calib row."""
    rows = [("raw", evaluate_fused(raw_model, fishy_test, device, 1.0))]
    for region in REGIONS:
        rows.append((f"temp({region})",
                     evaluate_fused(raw_model, fishy_test, device, temperatures[region])))
    if calib_model is not None:
        rows.append(("L_calib", evaluate_fused(calib_model, fishy_test, device, 1.0)))

    r = config.CALIB_BAND_RADIUS_PX
    header = (f"{'model':<12} {'AUROC':>7} {'AP':>7} {'FPR@95':>7} "
              f"{'ECE':>8} {f'band-ECE r={r}':>14}")
    print("\n" + header)
    print("-" * len(header))
    results = {}
    for name, hists in rows:
        m = hists["whole"].summary()
        m["band_ece"] = hists["band"].ece()
        print(f"{name:<12} {m['auroc']:>7.4f} {m['ap']:>7.4f} {m['fpr95']:>7.4f} "
              f"{m['ece']:>8.5f} {m['band_ece']:>14.4f}")
        results[name] = m

    print("\nfitted temperatures: " +
          ", ".join(f"{k}={v:.4f}" for k, v in temperatures.items()))
    print(config.CALIBRATION_TRADEOFF_NOTE)
    print("The band column is the Novelty-6 claim; temp(band) is the fair "
          "baseline for it. Whole-image ECE is dominated by easy background "
          "pixels. For r=4/8/16, UBQ and confidence intervals run eval_spatial.py.")
    return results, dict(rows)


def reliability_diagram(hists, out_path="calibration_reliability.png"):
    """Whole-image and band panels, one curve per model. Built from the same
    per-bin data ScoreHistogram.ece() uses, so it cannot disagree with the
    table."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, region in zip(axes, REGIONS):
        ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
        for name, per_region in hists.items():
            conf, acc, weight = per_region[region].reliability_curve()
            mask = weight > 0
            ax.plot(conf[mask], acc[mask], marker="o", label=name)
        title = ("whole image" if region == "whole"
                 else f"boundary band r={config.CALIB_BAND_RADIUS_PX}px")
        ax.set_title(f"Reliability -- {title} (Fishyscapes test half)")
        ax.set_xlabel("mean predicted score (confidence)")
        ax.set_ylabel("empirical positive rate (accuracy)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="raw (Phase 2a) checkpoint to calibrate -- explicit, "
                             "since several checkpoints from different data coexist")
    parser.add_argument("--out", default=config.CHECKPOINT_3HEAD_CALIB,
                        help="where the L_calib checkpoint is written")
    parser.add_argument("--temp-only", action="store_true",
                        help="fit + evaluate temperature scaling only, skip the fine-tune")
    args = parser.parse_args()

    device = get_device()
    raw_model = load_trained_model(args.checkpoint, device)

    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    fishy_val, fishy_test = split_fishyscapes_pairs(all_pairs)

    temperatures = fit_temperatures(raw_model, fishy_val, device)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    run_name = "temperature_scaling_only" if args.temp_only else "l_calib_finetune"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "base_checkpoint": args.checkpoint,
            "temperature_whole": temperatures["whole"],
            "temperature_band": temperatures["band"],
            "band_radius_px": config.CALIB_BAND_RADIUS_PX,
        })
        calib_model = None
        if not args.temp_only:
            mlflow.log_params({
                "beta_calib": config.BETA_CALIB,
                "calib_epochs": config.CALIB_EPOCHS,
                "calib_lr": config.CALIB_LEARNING_RATE,
                "calib_anchor_weight": config.CALIB_ANCHOR_WEIGHT,
                "batch_size": config.BATCH_SIZE,
                "grad_accum_steps": config.GRAD_ACCUM_STEPS,
                "calib_loss_region": config.CALIB_LOSS_REGION,
                "calib_selection_region": config.CALIB_SELECTION_REGION,
                "calib_degradation_prob": config.CALIB_DEGRADATION_PROB,
                "auroc_drop_limit": config.CALIB_AUROC_DROP_LIMIT,
                "ap_drop_limit": config.CALIB_AP_DROP_LIMIT,
                "anomaly_source": config.ANOMALY_SOURCE,
            })
            calib_model = load_trained_model(
                run_calib_finetune(args.checkpoint, args.out, device), device)

        results, hists = comparison_table(raw_model, temperatures, calib_model,
                                          fishy_test, device)
        for name, m in results.items():
            key = name.replace("(", "_").replace(")", "")
            mlflow.log_metrics({f"test_{key}_{k}": v for k, v in m.items()})

        reliability_diagram(hists)
        mlflow.log_artifact("calibration_reliability.png")


if __name__ == "__main__":
    main()

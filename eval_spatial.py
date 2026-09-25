"""Boundary-only ECE (Novelty 6) + UBQ, eval-only.

PLAN.md's "Boundary-only ECE (Novelty 6) + UBQ" section, design point 2:
this script is intentionally separate from train.py's evaluate_ood (drives
checkpoint selection, must not change) and calibrate.py's evaluate_fused
(used by fit/finetune/comparison_table). It never trains and never writes
a checkpoint -- forward passes only.

Checkpoint path gotcha (see PLAN.md): config.CHECKPOINT_3HEAD points to a
stale checkpoint on this machine. This script takes explicit --raw/--calib
paths instead of trusting config defaults for exactly that reason.

UBQ threshold: Fishyscapes fits threshold_at_tpr(0.95) on the VAL half and
applies it unchanged to test (PLAN.md Step 4). CARLA has no val/test split,
so it self-fits on the same 45 frames -- acceptable only because Step 3 is a
pipeline check, never a reported result.

Usage:
    python eval_spatial.py --dataset carla
    python eval_spatial.py --dataset fishyscapes
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from metrics import ScoreHistogram, boundary_band, ubq
from train import amp_context
from utils import get_device, load_trained_model

RADII = (4, 8, 16)
# 1500 = 15 x 100, so ece()'s 15 bins line up exactly with per-image bins.
PER_IMAGE_BINS = 1500
N_BOOTSTRAP = 1000


def carla_pairs():
    images = sorted(glob.glob(os.path.join(config.CARLA_IMAGES_DIR, "*.png")))
    masks = sorted(glob.glob(os.path.join(config.CARLA_MASKS_DIR, "*.npy")))
    assert len(images) == len(masks), (
        f"CARLA images/masks must pair 1:1 -- found {len(images)} images, "
        f"{len(masks)} masks.")
    return list(zip(images, masks))


def load_label(mask_path, dataset):
    """-> (anomaly_mask bool (H,W), valid bool (H,W))."""
    if dataset == "carla":
        mask = np.load(mask_path).astype(bool)
        return mask, np.ones_like(mask)  # CLAUDE.md: no 255/void pixels in CARLA masks
    label_map = np.array(Image.open(mask_path))
    return (label_map == 1), (label_map != 255)


@torch.no_grad()
def cache_logits(model, pairs, device, dataset):
    """One forward pass per image, logits kept at the model's native output
    resolution (tiny) and upsampled on demand -- caching full-resolution
    Fishyscapes logits would cost ~25MB per image per model."""
    cached = []
    model.eval()
    for image_path, mask_path in pairs:
        anomaly_mask, valid = load_label(mask_path, dataset)
        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        cached.append((out["ood_logits"].float().squeeze(0).cpu(), anomaly_mask, valid))
    return cached


def fused_scores(logits, shape, temperature=1.0):
    """Same order as calibrate.py's evaluate_fused: upsample logits, then
    per-head sigmoid(logit/T), then mean across heads."""
    up = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(up / temperature).mean(dim=1).squeeze(0).numpy()


def compute_bands(cached, radii=RADII):
    """Bands depend only on ground truth, so compute once and share across models."""
    return [{r: boundary_band(mask, radius_px=r, valid=valid) for r in radii}
            for _, mask, valid in cached]


def whole_histogram(cached, temperature):
    hist = ScoreHistogram()
    for logits, mask, valid in cached:
        scores = fused_scores(logits, mask.shape, temperature)
        hist.update(scores[valid], mask[valid].astype(np.int64))
    return hist


def histogram_from_arrays(pos, neg, cps, cpn):
    hist = ScoreHistogram(n_bins=PER_IMAGE_BINS)
    hist.pos, hist.neg, hist.conf_sum_pos, hist.conf_sum_neg = pos, neg, cps, cpn
    return hist


def evaluate_model(name, cached, bands, temperature, threshold_source=None, radii=RADII):
    whole = ScoreHistogram()
    band_hists = {r: ScoreHistogram() for r in radii}
    band_counts = {r: [0, 0] for r in radii}
    per_image = {r: [] for r in radii}
    score_maps = []

    for (logits, mask, valid), img_bands in zip(cached, bands):
        scores = fused_scores(logits, mask.shape, temperature)
        score_maps.append(scores)
        whole.update(scores[valid], mask[valid].astype(np.int64))
        for r in radii:
            band = img_bands[r]
            s, y = scores[band], mask[band].astype(np.int64)
            band_hists[r].update(s, y)
            small = ScoreHistogram(n_bins=PER_IMAGE_BINS)
            small.update(s, y)
            per_image[r].append(small)
            band_counts[r][0] += int(band.sum())
            band_counts[r][1] += int(y.sum())

    threshold_hist = threshold_source if threshold_source is not None else whole
    threshold = threshold_hist.threshold_at_tpr(0.95)

    fields = {"pred_to_gt_px": [], "gt_to_pred_px": [], "pred_to_gt_p95": []}
    pred_fracs, skipped = [], 0
    for scores, (_, mask, valid) in zip(score_maps, cached):
        pred_mask = (scores >= threshold) & valid
        pred_fracs.append(pred_mask.sum() / max(valid.sum(), 1))
        result = ubq(pred_mask, mask, valid=valid)
        if any(np.isnan(v) for v in result.values()):
            skipped += 1
            continue
        for k in fields:
            fields[k].append(result[k])
    ubq_means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in fields.items()}

    print(f"\n=== {name} (temperature={temperature:.4f}) ===")
    m = whole.summary()
    print(f"  whole-image  AUROC={m['auroc']:.4f} AP={m['ap']:.4f} "
          f"FPR@95={m['fpr95']:.4f} ECE={m['ece']:.4f}")
    for r in radii:
        bm = band_hists[r].summary()
        total, pos = band_counts[r]
        print(f"  band r={r:<2} AUROC={bm['auroc']:.4f} AP={bm['ap']:.4f} "
              f"FPR@95={bm['fpr95']:.4f} ECE={bm['ece']:.4f}  "
              f"(n_px={total}, pos_rate={pos / total if total else float('nan'):.4f})")
    src = "val half" if threshold_source is not None else "same set (self-fit)"
    print(f"  UBQ @ threshold_at_tpr(0.95)={threshold:.5f} [fit on {src}]  "
          f"pred_to_gt_px={ubq_means['pred_to_gt_px']:.2f}  "
          f"gt_to_pred_px={ubq_means['gt_to_pred_px']:.2f}  "
          f"pred_to_gt_p95={ubq_means['pred_to_gt_p95']:.2f}  "
          f"mean predicted-positive fraction={np.mean(pred_fracs):.4f}  "
          f"(skipped {skipped}/{len(cached)} images with an empty mask)")
    if np.mean(pred_fracs) > 0.2:
        print("  WARNING: threshold marks >20% of the frame as anomalous -- UBQ is "
              "saturated at this threshold and does not measure boundary quality.")
    return per_image


def paired_bootstrap(per_image_a, per_image_b, label, radii=RADII, seed=config.GLOBAL_SEED):
    """95% CI of band-ECE(a) - band-ECE(b), resampling the SAME image indices
    for both models each round (paired)."""
    rng = np.random.default_rng(seed)
    print(f"\npaired bootstrap, {N_BOOTSTRAP} resamples: band-ECE({label})")
    for r in radii:
        stacks = []
        for hists in (per_image_a[r], per_image_b[r]):
            stacks.append(tuple(np.stack([getattr(h, f) for h in hists])
                                for f in ("pos", "neg", "conf_sum_pos", "conf_sum_neg")))
        n = len(per_image_a[r])
        full = [histogram_from_arrays(*(a.sum(0) for a in s)).ece() for s in stacks]
        diffs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            eces = [histogram_from_arrays(*(a[idx].sum(0) for a in s)).ece() for s in stacks]
            diffs[i] = eces[0] - eces[1]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"  r={r:<2} diff={full[0] - full[1]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["carla", "fishyscapes"], required=True)
    parser.add_argument("--raw", default="model_3head_best.pth",
                        help="raw checkpoint path (repo root, NOT config.CHECKPOINT_3HEAD -- see PLAN.md)")
    parser.add_argument("--calib", default="model_3head_calib_best.pth",
                        help="L_calib checkpoint path (repo root)")
    parser.add_argument("--temperature", type=float, default=1.7142,
                        help="fitted temperature from calibrate.log -- do not re-fit, per PLAN.md")
    args = parser.parse_args()

    device = get_device()
    raw_model = load_trained_model(args.raw, device)
    calib_model = load_trained_model(args.calib, device)

    thresholds = {"raw": None, "temp": None, "calib": None}
    if args.dataset == "carla":
        pairs = carla_pairs()
        print(f"CARLA: {len(pairs)} frames. NOTE per PLAN.md Step 3 caveats -- these "
              f"objects are in the training CutMix bank and CARLA backgrounds are "
              f"out-of-domain for a Cityscapes-trained model. Pipeline validation only, "
              f"not generalization evidence.")
    else:
        all_pairs = list_fishyscapes_pairs()
        assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
        fishy_val, pairs = split_fishyscapes_pairs(all_pairs)
        print(f"Fishyscapes: fitting UBQ thresholds on {len(fishy_val)} val images, "
              f"evaluating on {len(pairs)} test images.")
        raw_val = cache_logits(raw_model, fishy_val, device, "fishyscapes")
        calib_val = cache_logits(calib_model, fishy_val, device, "fishyscapes")
        thresholds = {
            "raw": whole_histogram(raw_val, 1.0),
            "temp": whole_histogram(raw_val, args.temperature),
            "calib": whole_histogram(calib_val, 1.0),
        }
        del raw_val, calib_val

    raw_cached = cache_logits(raw_model, pairs, device, args.dataset)
    calib_cached = cache_logits(calib_model, pairs, device, args.dataset)
    bands = compute_bands(raw_cached)

    per_raw = evaluate_model("raw", raw_cached, bands, 1.0, thresholds["raw"])
    per_temp = evaluate_model("temp-scaled", raw_cached, bands, args.temperature, thresholds["temp"])
    per_calib = evaluate_model("L_calib", calib_cached, bands, 1.0, thresholds["calib"])

    paired_bootstrap(per_calib, per_temp, "L_calib) - band-ECE(temp")
    paired_bootstrap(per_calib, per_raw, "L_calib) - band-ECE(raw")


if __name__ == "__main__":
    main()

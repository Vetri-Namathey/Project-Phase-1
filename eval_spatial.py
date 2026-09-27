"""Boundary-only ECE (Novelty 6) + UBQ, eval-only.

PLAN.md's "Boundary-only ECE (Novelty 6) + UBQ" section, design point 2:
this script is intentionally separate from train.py's evaluate_ood (drives
checkpoint selection, must not change) and calibrate.py's evaluate_fused
(used by fit/finetune/comparison_table). It never trains and never writes
a checkpoint -- forward passes only.

Checkpoints are explicit (--raw required, --calib optional): checkpoints
trained on different anomaly data coexist, and a config default silently
picking the wrong one is how a result gets misattributed.

Temperatures are FITTED here, on the Fishyscapes val half, with
calibrate.fit_temperatures (plain NLL, whole and band). The value that used
to be hardcoded (1.7142) came from main's pos_weight=20 fit and is invalid --
see config.TEMPERATURE_LOG_BOUNDS. --temp-whole/--temp-band skip the fit
when the values are already known for this exact checkpoint.

UBQ: local UBQ (metrics.ubq_local) at the max-F1 threshold fitted on the
Fishyscapes VAL half and applied unchanged to test, then swept over fixed
thresholds. The old threshold_at_tpr(0.95) operating point flagged ~3% of
every frame and the frame-wide Hausdorff distance was ~1080px at every
setting -- it measured false alarms, not outlines. CARLA has no val/test split, so it self-fits
on the same 45 frames -- a pipeline check only. The current CARLA masks are
also misaligned (asynchronous capture in generate_anomalies.py: 23 of 45
blend into the road), so CARLA numbers are not evidence of anything until
the frames are re-rendered.

Usage:
    python eval_spatial.py --dataset fishyscapes --raw <raw.pth>
    python eval_spatial.py --dataset fishyscapes --raw <raw.pth> --calib <calib.pth>
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
from metrics import ScoreHistogram, boundary_band, ubq_local
from train import amp_context
from utils import get_device, load_trained_model

RADII = (4, 8, 16)
# UBQ is measured only around the true objects (see metrics.ubq_local); false
# alarms elsewhere are reported separately as far-FP. 32px ~ the median real
# object's longest side at Fishyscapes' native resolution.
UBQ_ROI_PX = 32
UBQ_TOL_PX = 4
UBQ_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)
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
    ubq_rows = ubq_sweep(score_maps, cached, threshold_hist.threshold_at_max_f1())

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
    print(f"  UBQ (local, roi={UBQ_ROI_PX}px, boundary-F1 tolerance={UBQ_TOL_PX}px); "
          f"max-F1 threshold fit on {src}")
    print(f"    {'threshold':<16} {'objects found':>13} {'spill px':>9} {'miss px':>8} "
          f"{'bnd-F1':>7} {'far-FP':>7} {'flagged':>8}")
    for r in ubq_rows:
        found = f"{r['n_found']}/{r['n_objects']}"
        print(f"    {r['label']:<16} {found:>13} {r['spill_px']:>9.2f} {r['miss_px']:>8.2f} "
              f"{r['boundary_f1']:>7.3f} {r['far_fp_frac']:>7.1%} {r['flagged']:>8.3%}")
    print("    (spill / miss / bnd-F1 are averaged over FOUND objects only; far-FP = share "
          "of flagged pixels > roi from any object)")
    return per_image


def ubq_sweep(score_maps, cached, max_f1_threshold):
    """Local UBQ at the val-fitted max-F1 operating point, then across fixed
    thresholds -- the plan's Novelty-4 stability check. Every true object
    counts: found ones are scored, missed ones lower object recall."""
    thresholds = [(f"maxF1 t={max_f1_threshold:.3f}", max_f1_threshold)] + \
        [(f"t={t:.1f}", t) for t in UBQ_SWEEP]
    rows = []
    for label, t in thresholds:
        objects, far_px, pred_px, flagged = [], 0, 0, []
        for scores, (_, mask, valid) in zip(score_maps, cached):
            pred = (scores >= t) & valid
            flagged.append(pred.sum() / max(valid.sum(), 1))
            objs, f, p = ubq_local(pred, mask, valid=valid,
                                   roi_px=UBQ_ROI_PX, tolerance_px=UBQ_TOL_PX)
            objects += objs
            far_px += f
            pred_px += p
        found = [o for o in objects if o["detected"]]
        row = {"label": label, "n_objects": len(objects), "n_found": len(found),
               "flagged": float(np.mean(flagged)),
               "far_fp_frac": far_px / pred_px if pred_px else float("nan")}
        for k in ("spill_px", "miss_px", "boundary_f1"):
            row[k] = float(np.mean([o[k] for o in found])) if found else float("nan")
        rows.append(row)
    return rows


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
    parser.add_argument("--raw", required=True, help="raw (Phase 2a) checkpoint path")
    parser.add_argument("--calib", default=None,
                        help="L_calib checkpoint path; omit to evaluate raw + temperature only")
    parser.add_argument("--temp-whole", type=float, default=None,
                        help="skip the fit and use this whole-image T (must be for --raw)")
    parser.add_argument("--temp-band", type=float, default=None,
                        help="skip the fit and use this band T (must be for --raw)")
    args = parser.parse_args()

    device = get_device()
    raw_model = load_trained_model(args.raw, device)
    calib_model = load_trained_model(args.calib, device) if args.calib else None

    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    fishy_val, fishy_test = split_fishyscapes_pairs(all_pairs)

    if args.temp_whole is not None and args.temp_band is not None:
        temps = {"whole": args.temp_whole, "band": args.temp_band}
        print(f"using given temperatures {temps}")
    else:
        from calibrate import fit_temperatures
        temps = fit_temperatures(raw_model, fishy_val, device)

    # (name, model, temperature). Threshold for UBQ is fitted per row.
    rows = [("raw", raw_model, 1.0),
            ("temp(whole)", raw_model, temps["whole"]),
            ("temp(band)", raw_model, temps["band"])]
    if calib_model is not None:
        rows.append(("L_calib", calib_model, 1.0))

    thresholds = {name: None for name, _, _ in rows}
    if args.dataset == "carla":
        pairs = carla_pairs()
        print(f"CARLA: {len(pairs)} frames. Pipeline validation only -- these objects "
              f"are in the training bank, the backgrounds are out-of-domain, and the "
              f"current masks are misaligned (see this file's docstring).")
    else:
        pairs = fishy_test
        print(f"Fishyscapes: fitting UBQ thresholds on {len(fishy_val)} val images, "
              f"evaluating on {len(pairs)} test images.")
        val_cache = {"raw": cache_logits(raw_model, fishy_val, device, "fishyscapes")}
        if calib_model is not None:
            val_cache["L_calib"] = cache_logits(calib_model, fishy_val, device, "fishyscapes")
        for name, model, t in rows:
            thresholds[name] = whole_histogram(
                val_cache["L_calib" if model is calib_model else "raw"], t)
        del val_cache

    test_cache = {"raw": cache_logits(raw_model, pairs, device, args.dataset)}
    if calib_model is not None:
        test_cache["L_calib"] = cache_logits(calib_model, pairs, device, args.dataset)
    bands = compute_bands(test_cache["raw"])

    per_image = {}
    for name, model, t in rows:
        cached = test_cache["L_calib" if model is calib_model else "raw"]
        per_image[name] = evaluate_model(name, cached, bands, t, thresholds[name])

    # The comparison that matters most comes first: L_calib against the
    # strongest temperature baseline for a band claim.
    if calib_model is not None:
        paired_bootstrap(per_image["L_calib"], per_image["temp(band)"],
                         "L_calib) - band-ECE(temp(band)")
        paired_bootstrap(per_image["L_calib"], per_image["raw"],
                         "L_calib) - band-ECE(raw")
    paired_bootstrap(per_image["temp(band)"], per_image["raw"],
                     "temp(band)) - band-ECE(raw")


if __name__ == "__main__":
    main()

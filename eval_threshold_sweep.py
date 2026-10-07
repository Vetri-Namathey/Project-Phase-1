"""Threshold-sensitivity sweep for the outline result (PLAN.md "yes" item 9). Eval only.

Question: is C5's outline gain over raw (per-image boundary F1 +0.0950 [+0.0349,
+0.1543] at each row's own val max-F1 threshold, eval_fishyscapes_T122.log) a
property of C5, or of the one operating point it was measured at?

Thresholds are matched by OPERATING POINT, never by value (MISTAKES.md M11: C5's
scores live on a different scale): for each target fraction of valid pixels
flagged, each row gets the threshold that flags that fraction of the Fishyscapes
VAL half, and that threshold is applied to the TEST half. C5 is fitted on val
exactly as in eval_spatial.py.

Pre-registered rule (written before the first run):
  - sanity: at each row's own val max-F1 threshold this script must reproduce
    eval_fishyscapes_T122.log's objects found (raw 40/85, C5 59/85); else STOP.
  - "robust": the paired per-image boundary-F1 difference C5 - raw has a 95% CI
    above 0 at >= 4 of the 6 operating points; "threshold-dependent" otherwise.
  - objects found and far-FP are reported at every point, so the cost of C5
    (more far false alarms) is visible next to any gain.

    python eval_threshold_sweep.py --raw model_3head_best.pth | Tee-Object -FilePath eval_threshold_sweep.log
"""

import argparse

import numpy as np

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from eval_spatial import (UBQ_ROI_PX, UBQ_TOL_PX, cache_logits, compute_bands,
                          fit_context_calibrator, fused_scores, whole_histogram)
from metrics import ubq_local
from utils import get_device, load_trained_model

# Fractions of valid pixels flagged on val. The signed max-F1 points flag
# 0.144% (raw) and 0.170% (C5) of test pixels, so the grid brackets them.
FLAG_FRACTIONS = (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005)
SAMPLE_PER_IMAGE = 200_000
N_BOOT = 1000


def score_maps(cached, calibrator):
    return [fused_scores(logits, mask.shape, calibrator) for logits, mask, _ in cached]


def threshold_for_fraction(maps, cached, frac, rng):
    """Score above which `frac` of the val valid pixels lie (uniform pixel sample)."""
    sample = []
    for scores, (_, _, valid) in zip(maps, cached):
        v = scores[valid]
        sample.append(rng.choice(v, size=min(SAMPLE_PER_IMAGE, v.size), replace=False))
    return float(np.quantile(np.concatenate(sample), 1.0 - frac))


def outline(maps, cached, threshold):
    """-> (found, n_objects, far-FP share, flagged share, per-image mean BF1 (NaN if none found))."""
    found = n_obj = far = pred = 0
    flagged, bf1 = [], []
    for scores, (_, mask, valid) in zip(maps, cached):
        p = (scores >= threshold) & valid
        flagged.append(p.sum() / max(valid.sum(), 1))
        objs, f, n = ubq_local(p, mask, valid=valid, roi_px=UBQ_ROI_PX, tolerance_px=UBQ_TOL_PX)
        hits = [o["boundary_f1"] for o in objs if o["detected"]]
        found += len(hits)
        n_obj += len(objs)
        far += f
        pred += n
        bf1.append(float(np.mean(hits)) if hits else float("nan"))
    return found, n_obj, (far / pred if pred else float("nan")), float(np.mean(flagged)), np.array(bf1)


def paired_ci(a, b, rng):
    both = ~np.isnan(a) & ~np.isnan(b)
    d = (b - a)[both]
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan"), len(d)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="e.g. model_3head_best.pth (rule 11)")
    args = parser.parse_args()

    device = get_device()
    model = load_trained_model(args.raw, device)
    val_pairs, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    print(f"threshold sweep: {args.raw}, scale 1, {len(val_pairs)} val / {len(test_pairs)} test images")
    val = cache_logits(model, val_pairs, device, "fishyscapes")
    test = cache_logits(model, test_pairs, device, "fishyscapes")
    c5 = fit_context_calibrator(val, compute_bands(val, radii=(8,)))
    print(f"C5 refit on val: {c5!r}  (eval_fishyscapes_T122.log: w=[+1.448, +2.090, -0.609, -1.068, -1.711])")

    rows = {"raw": 1.0, "C5": c5}
    val_maps = {k: score_maps(val, t) for k, t in rows.items()}
    test_maps = {k: score_maps(test, t) for k, t in rows.items()}

    print("\nSANITY: each row at its own val max-F1 threshold (must match eval_fishyscapes_T122.log)")
    expected = {"raw": 40, "C5": 59}
    ok = True
    for k, t in rows.items():
        thr = whole_histogram(val, t).threshold_at_max_f1()
        found, n_obj, far, flag, _ = outline(test_maps[k], test, thr)
        match = found == expected[k]
        ok &= match
        print(f"  {k:4s} t={thr:.4f}  found {found}/{n_obj} (expected {expected[k]})  far-FP {far:.1%}  "
              f"flagged {flag:.3%}  {'PASS' if match else 'FAIL'}")
    if not ok:
        raise SystemExit("sanity FAILED: the sweep does not reproduce the signed table -- STOP (rule)")

    rng = np.random.default_rng(config.GLOBAL_SEED)
    print("\nSWEEP: thresholds matched by the fraction of VAL pixels flagged, applied to TEST")
    print(f"  {'val flag':>8s} | {'raw t':>7s} found  BF1   far-FP | {'C5 t':>7s} found  BF1   far-FP | "
          f"BF1 C5-raw [95% CI] (n images)")
    robust = 0
    for frac in FLAG_FRACTIONS:
        res = {}
        for k in rows:
            thr = threshold_for_fraction(val_maps[k], val, frac, rng)
            res[k] = (thr,) + outline(test_maps[k], test, thr)
        m, lo, hi, n = paired_ci(res["raw"][5], res["C5"][5], rng)
        robust += lo > 0
        cells = " | ".join(f"{r[0]:7.4f} {r[1]:3d}/{r[2]:<2d} {np.nanmean(r[5]):.3f} {r[3]:6.1%}"
                           for r in (res["raw"], res["C5"]))
        print(f"  {frac:8.3%} | {cells} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})")
    verdict = "ROBUST" if robust >= 4 else "THRESHOLD-DEPENDENT"
    print(f"\nRULE: CI above 0 at {robust}/{len(FLAG_FRACTIONS)} operating points -> {verdict} "
          f"(robust needs >= 4)")


if __name__ == "__main__":
    main()

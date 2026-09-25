"""Validate the histogram metric estimators against sklearn.

metrics.py replaced sklearn's sorted-array path with a histogram
accumulator, because evaluation touches 170.6M pixels and the old code
sorted all of them four times per epoch. Binning is an approximation, so
that swap is only safe if it is checked -- this is the check.

Run: python validate_metrics.py
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve)

from metrics import ConfusionMatrix, ScoreHistogram, boundary_band, ubq

TOLERANCE = 1e-4


def sklearn_fpr_at_tpr(scores, labels, target=0.95):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, target, side="left")
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def case(name, scores, labels):
    hist = ScoreHistogram()
    # Feed in chunks, to also exercise multi-image accumulation.
    for chunk in np.array_split(np.arange(len(scores)), 7):
        hist.update(scores[chunk], labels[chunk])

    rows = [
        ("AUROC", hist.auroc(), roc_auc_score(labels, scores)),
        ("AP", hist.average_precision(), average_precision_score(labels, scores)),
        ("FPR@95", hist.fpr_at_tpr(0.95), sklearn_fpr_at_tpr(scores, labels)),
    ]

    print(f"\n{name}  (n={len(scores):,}  pos_rate={labels.mean():.4%})")
    ok = True
    for metric, ours, theirs in rows:
        delta = abs(ours - theirs)
        flag = "OK " if delta < TOLERANCE else "FAIL"
        if delta >= TOLERANCE:
            ok = False
        print(f"  {flag} {metric:<8} ours={ours:.6f}  sklearn={theirs:.6f}  d={delta:.2e}")
    return ok


def main():
    rng = np.random.default_rng(0)
    all_ok = True

    # 1. Balanced, well separated.
    n = 200_000
    labels = (rng.random(n) < 0.5).astype(np.int64)
    scores = np.clip(rng.normal(np.where(labels == 1, 0.7, 0.3), 0.15), 0, 1)
    all_ok &= case("separable, balanced", scores, labels)

    # 2. Fishyscapes-like: 0.28% positive rate, weak separation.
    n = 500_000
    labels = (rng.random(n) < 0.0028).astype(np.int64)
    scores = np.clip(rng.normal(np.where(labels == 1, 0.55, 0.45), 0.2), 0, 1)
    all_ok &= case("realistic imbalance (0.28% pos)", scores, labels)

    # 3. Collapsed model: everything crushed into a narrow band near zero.
    #    This is the regime the old runs were actually in, so the estimator
    #    has to stay correct there too.
    n = 300_000
    labels = (rng.random(n) < 0.003).astype(np.int64)
    scores = np.clip(rng.normal(0.001, 0.01, n), 0, 1)
    all_ok &= case("collapsed model (mean~0.001, std~0.01)", scores, labels)

    # 4. Heavy ties -- discrete scores, the worst case for a histogram.
    n = 100_000
    labels = (rng.random(n) < 0.1).astype(np.int64)
    scores = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n)
    scores = np.clip(scores + labels * 0.25, 0, 1)
    all_ok &= case("heavy ties (5 discrete values)", scores, labels)

    # 5. mIoU against a hand-computed case.
    print("\nmIoU")
    cm = ConfusionMatrix(num_classes=3, ignore_index=255)
    target = np.array([0, 0, 1, 1, 2, 2, 255, 255])
    pred = np.array([0, 1, 1, 1, 2, 0, 0, 1])
    cm.update(pred, target)
    # class0: TP=1, FP=1(from t=2,p=0), FN=1 -> IoU 1/3
    # class1: TP=2, FP=1(from t=0,p=1), FN=0 -> IoU 2/3
    # class2: TP=1, FP=0, FN=1            -> IoU 1/2
    expected = float(np.mean([1 / 3, 2 / 3, 1 / 2]))
    got = cm.miou()
    delta = abs(got - expected)
    flag = "OK " if delta < 1e-9 else "FAIL"
    if delta >= 1e-9:
        all_ok = False
    print(f"  {flag} mIoU     ours={got:.6f}  expected={expected:.6f}  d={delta:.2e}")
    print(f"  ignore_index respected: {cm.matrix.sum()} == 6 valid pixels "
          f"-> {'OK' if cm.matrix.sum() == 6 else 'FAIL'}")
    if cm.matrix.sum() != 6:
        all_ok = False

    # 6. Boundary-only ECE (Novelty 6) + UBQ primitives -- PLAN.md's
    #    "Boundary-only ECE (Novelty 6) + UBQ" section, Step 1.
    print("\nBoundary band + UBQ primitives")

    # 6a. Band pixel count vs. the analytic outer-offset area of a square.
    #     Minkowski sum of an L-side square with a radius-r disk adds a
    #     ring of area perimeter*r + pi*r^2 outside the square -- compare
    #     boundary_band(side="outer") pixel count against that continuous
    #     formula. Pixel-grid discretization gives a few-percent gap, not
    #     an exact match, so this uses a relative tolerance.
    canvas = np.zeros((200, 200), dtype=bool)
    canvas[50:150, 50:150] = True  # 100x100 square, well clear of the canvas edge
    L, r = 100, 8
    band = boundary_band(canvas, radius_px=r, side="outer")
    analytic_area = 4 * L * r + np.pi * r**2
    got_area = int(band.sum())
    rel_err = abs(got_area - analytic_area) / analytic_area
    flag = "OK " if rel_err < 0.05 else "FAIL"
    if rel_err >= 0.05:
        all_ok = False
    print(f"  {flag} band area (outer, r={r})  ours={got_area}  analytic={analytic_area:.1f}  "
          f"rel_err={rel_err:.4f}")

    # 6b. ubq(gt, gt) is exactly 0/0 -- every pixel is its own nearest match.
    result = ubq(canvas, canvas)
    same = (result["pred_to_gt_px"] == 0.0 and result["gt_to_pred_px"] == 0.0
            and result["pred_to_gt_p95"] == 0.0)
    flag = "OK " if same else "FAIL"
    if not same:
        all_ok = False
    print(f"  {flag} ubq(gt, gt)  {result}")

    # 6c. ubq agrees with scipy's directed_hausdorff (the reference
    #     implementation this is checked against, not the production path)
    #     to within 1px on random small masks.
    rng = np.random.default_rng(1)
    for trial in range(5):
        h, w = 30, 30
        pred = rng.random((h, w)) < 0.15
        gt = rng.random((h, w)) < 0.15
        if not pred.any() or not gt.any():
            continue
        got = ubq(pred, gt)
        pred_pts = np.argwhere(pred)
        gt_pts = np.argwhere(gt)
        ref_pred_to_gt = directed_hausdorff(pred_pts, gt_pts)[0]
        ref_gt_to_pred = directed_hausdorff(gt_pts, pred_pts)[0]
        d1 = abs(got["pred_to_gt_px"] - ref_pred_to_gt)
        d2 = abs(got["gt_to_pred_px"] - ref_gt_to_pred)
        ok = d1 < 1.0 and d2 < 1.0
        flag = "OK " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {flag} ubq vs directed_hausdorff (trial {trial})  "
              f"pred_to_gt d={d1:.3f}  gt_to_pred d={d2:.3f}")

    # 6d. threshold_at_tpr is consistent with fpr_at_tpr -- re-scoring with
    #     `scores >= threshold_at_tpr(t)` should reproduce fpr_at_tpr(t).
    n = 200_000
    labels = (rng.random(n) < 0.05).astype(np.int64)
    scores = np.clip(rng.normal(np.where(labels == 1, 0.7, 0.3), 0.2), 0, 1)
    hist = ScoreHistogram()
    hist.update(scores, labels)
    target = 0.95
    thresh = hist.threshold_at_tpr(target)
    expected_fpr = hist.fpr_at_tpr(target)
    pred_positive = scores >= thresh
    P, N = labels.sum(), (labels == 0).sum()
    reproduced_tpr = (pred_positive & (labels == 1)).sum() / P
    reproduced_fpr = (pred_positive & (labels == 0)).sum() / N
    tpr_ok = reproduced_tpr >= target - 0.01
    fpr_delta = abs(reproduced_fpr - expected_fpr)
    ok = tpr_ok and fpr_delta < 1e-3
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag} threshold_at_tpr  thresh={thresh:.5f}  reproduced_tpr={reproduced_tpr:.4f}  "
          f"fpr_at_tpr={expected_fpr:.5f}  reproduced_fpr={reproduced_fpr:.5f}  d={fpr_delta:.2e}")

    print("\n" + ("ALL METRICS MATCH SKLEARN" if all_ok else "MISMATCH -- DO NOT TRAIN"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

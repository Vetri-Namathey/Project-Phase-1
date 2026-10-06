"""Validate the histogram metric estimators against sklearn.

metrics.py replaced sklearn's sorted-array path with a histogram
accumulator, because evaluation touches 170.6M pixels and the old code
sorted all of them four times per epoch. Binning is an approximation, so
that swap is only safe if it is checked -- this is the check.

Run: python validate_metrics.py
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from metrics import (ConfusionMatrix, ScoreHistogram, boundary_band, paired_bootstrap_ap,
                     ubq, ubq_local)

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

    # 6e. threshold_at_max_f1 agrees with sklearn's precision-recall curve.
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    ref_best_f1 = float(f1.max())
    t = hist.threshold_at_max_f1()
    pred_positive = scores >= t
    tp = (pred_positive & (labels == 1)).sum()
    got_f1 = 2 * tp / (pred_positive.sum() + labels.sum())
    ok = abs(got_f1 - ref_best_f1) < 1e-3
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag} threshold_at_max_f1  t={t:.5f}  F1 at t={got_f1:.5f}  "
          f"sklearn best F1={ref_best_f1:.5f}  d={abs(got_f1 - ref_best_f1):.2e}")

    # 6f. ubq_local: exact on a hand-built case, and against brute force.
    #     A 20x20 square object; prediction = the square shifted 3px right.
    gt = np.zeros((80, 80), bool)
    gt[30:50, 30:50] = True
    pred = np.zeros_like(gt)
    pred[30:50, 33:53] = True
    objs, far_px, pred_px = ubq_local(pred, gt, roi_px=10, tolerance_px=2)
    got = objs[0]
    # Spill: columns 50,51,52 at distance 1,2,3 -> mean 2. Miss: columns
    # 30,31,32 at distance 3,2,1 -> 6 px per row / 400 object pixels.
    exp_spill, exp_miss = 2.0, (3 + 2 + 1) * 20 / 400
    ok = (len(objs) == 1 and got["detected"] and abs(got["spill_px"] - exp_spill) < 1e-9
          and abs(got["miss_px"] - exp_miss) < 1e-9 and far_px == 0
          and 0.0 < got["boundary_f1"] < 1.0)
    same = ubq_local(gt, gt)[0][0]
    ok = ok and same["spill_px"] == 0.0 and same["miss_px"] == 0.0 and same["boundary_f1"] == 1.0
    far = pred.copy()
    far[0:2, 0:5] = True                                   # 10 px far false alarm
    _, far_px, far_pred_px = ubq_local(far, gt, roi_px=10)
    ok = ok and far_px == 10 and far_pred_px == far.sum()
    # A second object that is completely missed must be counted as missed and
    # must not change the first object's scores -- the leak this version fixes.
    gt2 = gt.copy()
    gt2[5:12, 65:75] = True
    objs2, _, _ = ubq_local(pred, gt2, roi_px=10, tolerance_px=2)
    first = [o for o in objs2 if o["detected"]]
    ok = ok and len(objs2) == 2 and len(first) == 1 and first[0] == got
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag} ubq_local exact case  {got}  far_fp_px={far_px}  "
          f"missed 2nd object isolated={len(first) == 1 and first[0] == got}")

    for trial in range(3):
        g = np.zeros((40, 40), bool)
        g[10 + trial:25, 12:28 - trial] = True
        p = rng.random((40, 40)) < 0.08
        p[12:24, 14:26] = True
        objs, far_px, pred_px = ubq_local(p, g, roi_px=6, tolerance_px=2)
        got = objs[0]
        gpts, ppts = np.argwhere(g), np.argwhere(p)
        d_to_g = np.sqrt(((ppts[:, None, :] - gpts[None, :, :]) ** 2).sum(-1)).min(1)
        near = ppts[d_to_g <= 6]
        d_to_p = np.sqrt(((gpts[:, None, :] - near[None, :, :]) ** 2).sum(-1)).min(1)
        outside = (d_to_g > 0) & (d_to_g <= 6)
        ref_spill = d_to_g[outside].mean() if outside.any() else 0.0
        ref_miss = d_to_p.mean()
        d = max(abs(got["spill_px"] - ref_spill), abs(got["miss_px"] - ref_miss),
                abs(far_px - int((d_to_g > 6).sum())), abs(pred_px - len(ppts)))
        ok = d < 1e-6
        flag = "OK " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {flag} ubq_local vs brute force (trial {trial})  max d={d:.2e}")

    # 7. Edge cases (MISTAKES.md standing rule 5) for the metrics PLAN.md P1
    #    depends on: ubq_local on degenerate masks, and the paired AP
    #    bootstrap on identical inputs.
    print("\nubq_local edge cases + paired_bootstrap_ap")
    gt = np.zeros((40, 40), bool)
    gt[10:20, 10:20] = True
    # 7a. Empty GT: no objects to score; every predicted pixel counts as far.
    pred = np.zeros_like(gt)
    pred[0:3, 0:3] = True
    objs, far_px, pred_px = ubq_local(pred, np.zeros_like(gt))
    ok_a = objs == [] and far_px == 9 and pred_px == 9
    # 7b. Empty prediction: the object is counted, as missed.
    objs, far_px, pred_px = ubq_local(np.zeros_like(gt), gt)
    ok_b = objs == [{"detected": False}] and far_px == 0 and pred_px == 0
    # 7c. Single-pixel object predicted exactly: zero spill/miss, F1 1.
    one = np.zeros_like(gt)
    one[25, 25] = True
    o = ubq_local(one, one)[0][0]
    ok_c = (o["detected"] and o["spill_px"] == 0.0 and o["miss_px"] == 0.0
            and o["boundary_f1"] == 1.0)
    # 7d. Object touching the image border: the crop window is clipped, and
    #     an exact prediction still scores F1 1 -- no crash, no NaN.
    border = np.zeros_like(gt)
    border[0:8, 30:40] = True
    o = ubq_local(border, border)[0][0]
    ok_d = o["detected"] and o["boundary_f1"] == 1.0 and o["miss_px"] == 0.0
    vals = [v for o in (ubq_local(pred, gt)[0] + ubq_local(one, one)[0]
                        + ubq_local(border, border)[0]) for v in o.values()]
    ok_nan = not any(isinstance(v, float) and np.isnan(v) for v in vals)
    ok = ok_a and ok_b and ok_c and ok_d and ok_nan
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag} ubq_local edge cases: empty GT={ok_a} empty pred={ok_b} "
          f"single pixel={ok_c} border object={ok_d} no NaN={ok_nan}")
    print(f"       3x3 pred ~10 px off the object (inside roi=32, so detected, F1 0): "
          f"{ubq_local(pred, gt)}")

    # 7e. paired_bootstrap_ap(a, a): difference exactly 0, CI exactly [0, 0].
    per_image = []
    for _ in range(6):
        y = (rng.random(5000) < 0.05).astype(np.int64)
        s = np.clip(rng.normal(np.where(y == 1, 0.7, 0.3), 0.2), 0, 1)
        h = ScoreHistogram()
        h.update(s, y)
        per_image.append((h.pos, h.neg))
    res = paired_bootstrap_ap(per_image, per_image, n_resamples=200)
    pooled = ScoreHistogram()
    pooled.pos = sum(p for p, _ in per_image)
    pooled.neg = sum(n for _, n in per_image)
    ok = (res["diff"] == 0.0 and res["lo"] == 0.0 and res["hi"] == 0.0
          and abs(res["ap_a"] - pooled.average_precision()) < 1e-12)
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag} paired_bootstrap_ap(a, a)  diff={res['diff']}  "
          f"CI [{res['lo']}, {res['hi']}]  AP={res['ap_a']:.6f} (= pooled AP)")

    print("\n" + ("ALL METRICS MATCH SKLEARN" if all_ok else "MISMATCH -- DO NOT TRAIN"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

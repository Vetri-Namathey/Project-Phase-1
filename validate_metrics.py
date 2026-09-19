"""Validate the histogram metric estimators against sklearn.

metrics.py replaced sklearn's sorted-array path with a histogram
accumulator, because evaluation touches 170.6M pixels and the old code
sorted all of them four times per epoch. Binning is an approximation, so
that swap is only safe if it is checked -- this is the check.

Run: python validate_metrics.py
"""

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve)

from metrics import ConfusionMatrix, ScoreHistogram

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

    print("\n" + ("ALL METRICS MATCH SKLEARN" if all_ok else "MISMATCH -- DO NOT TRAIN"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

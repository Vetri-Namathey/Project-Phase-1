"""Evaluation metrics shared by Experiment A and Experiment B.

AUROC / AP / FPR@95 / ECE are computed from a shared score histogram in a
single pass rather than from sklearn's sorted-array path. Evaluating the
100 Fishyscapes images touches 170.6M valid pixels, and the previous code
ran a full sort over all of them four separate times per epoch (fused score
plus each of three heads). Histogram accumulation is O(n) with no sort and
no 1.4GB temporary.

Binning is the only approximation, and it is bounded: with 200k bins over
[0,1] every score is placed within 2.5e-6 of its true value, at a fixed
6.4MB cost regardless of pixel count. 20k bins was tried first and failed
validation (1.7e-4 AUROC error) in exactly the regime that matters -- a
collapsed model crushes every score into a narrow band near zero, so most
bins sit empty and the few that are occupied carry everything. Validated
against sklearn including that case -- see validate_metrics.py.
"""

import numpy as np

import config

DEFAULT_BINS = 200000


class ScoreHistogram:
    """Accumulates (score, label) pairs across images without keeping them.

    Scores must lie in [0,1]; labels must be binary {0,1}. Feed each image
    as it is evaluated, then read the metrics off at the end.
    """

    def __init__(self, n_bins=DEFAULT_BINS):
        self.n_bins = n_bins
        self.pos = np.zeros(n_bins, dtype=np.int64)
        self.neg = np.zeros(n_bins, dtype=np.int64)
        self.conf_sum_pos = np.zeros(n_bins, dtype=np.float64)
        self.conf_sum_neg = np.zeros(n_bins, dtype=np.float64)

    def update(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels).ravel()
        if scores.size == 0:
            return

        idx = np.clip((scores * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        is_pos = labels == 1

        self.pos += np.bincount(idx[is_pos], minlength=self.n_bins)
        self.neg += np.bincount(idx[~is_pos], minlength=self.n_bins)
        self.conf_sum_pos += np.bincount(
            idx[is_pos], weights=scores[is_pos], minlength=self.n_bins)
        self.conf_sum_neg += np.bincount(
            idx[~is_pos], weights=scores[~is_pos], minlength=self.n_bins)

    @property
    def n_pos(self):
        return int(self.pos.sum())

    @property
    def n_neg(self):
        return int(self.neg.sum())

    def auroc(self):
        """Mann-Whitney form with mid-rank tie handling, matching sklearn."""
        P, N = self.n_pos, self.n_neg
        if P == 0 or N == 0:
            return float("nan")
        # Walk bins from highest score to lowest.
        pos_desc = self.pos[::-1].astype(np.float64)
        neg_desc = self.neg[::-1].astype(np.float64)
        pos_strictly_above = np.concatenate(([0.0], np.cumsum(pos_desc)[:-1]))
        concordant = (neg_desc * (pos_strictly_above + 0.5 * pos_desc)).sum()
        return float(concordant / (P * N))

    def average_precision(self):
        """AP = sum (R_n - R_{n-1}) * P_n, sklearn's step-wise definition."""
        P = self.n_pos
        if P == 0:
            return float("nan")
        pos_desc = self.pos[::-1].astype(np.float64)
        neg_desc = self.neg[::-1].astype(np.float64)

        tp = np.cumsum(pos_desc)
        fp = np.cumsum(neg_desc)
        keep = (pos_desc + neg_desc) > 0
        tp, fp = tp[keep], fp[keep]

        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / P
        recall_delta = np.diff(np.concatenate(([0.0], recall)))
        return float((recall_delta * precision).sum())

    def fpr_at_tpr(self, target_tpr=0.95):
        """False-positive rate at the lowest threshold reaching target TPR."""
        P, N = self.n_pos, self.n_neg
        if P == 0 or N == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)
        reached = np.nonzero((tp / P) >= target_tpr)[0]
        if len(reached) == 0:
            return 1.0
        return float(fp[reached[0]] / N)

    def ece(self, n_bins=15):
        """Expected Calibration Error over the accumulated histogram.

        REPORTING ONLY -- never import this as a training loss. The
        differentiable soft-binning surrogate used as L_calib in Phase 2b is
        a separate implementation and belongs in losses.py.

        Interpretation warning: the OOD positive rate on Fishyscapes is
        0.28%, so a model that outputs ~0.003 everywhere scores a near-
        perfect ECE while being completely non-discriminative. A low ECE is
        only meaningful alongside an AUROC that shows the model discriminates
        at all -- report the two together, never ECE on its own.
        """
        total = self.n_pos + self.n_neg
        if total == 0:
            return float("nan")

        edges = np.linspace(0, self.n_bins, n_bins + 1).astype(np.int64)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            count = self.pos[lo:hi].sum() + self.neg[lo:hi].sum()
            if count == 0:
                continue
            conf = (self.conf_sum_pos[lo:hi].sum() + self.conf_sum_neg[lo:hi].sum()) / count
            acc = self.pos[lo:hi].sum() / count
            ece += (count / total) * abs(conf - acc)
        return float(ece)

    def summary(self, prefix=""):
        return {
            f"{prefix}auroc": self.auroc(),
            f"{prefix}ap": self.average_precision(),
            f"{prefix}fpr95": self.fpr_at_tpr(0.95),
            f"{prefix}ece": self.ece(),
        }


# ---------------------------------------------------------------------------
# Convenience wrappers -- same call signatures the old module exposed, so
# existing scripts keep working on in-memory arrays.
# ---------------------------------------------------------------------------

def _histogram_of(scores, labels):
    hist = ScoreHistogram()
    hist.update(scores, labels)
    return hist


def compute_auroc(scores, labels):
    return _histogram_of(scores, labels).auroc()


def compute_average_precision(scores, labels):
    return _histogram_of(scores, labels).average_precision()


def compute_fpr_at_tpr(scores, labels, target_tpr=0.95):
    return _histogram_of(scores, labels).fpr_at_tpr(target_tpr)


def compute_ece(confidences, labels, n_bins=15):
    return _histogram_of(confidences, labels).ece(n_bins=n_bins)


# ---------------------------------------------------------------------------
# Segmentation quality -- the "did the OOD heads break normal perception?"
# check. Accumulated as a confusion matrix so it costs one pass and no
# per-image storage.
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    def __init__(self, num_classes=config.NUM_SEG_CLASSES, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        if pred.size == 0:
            return
        flat = target.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        self.matrix += np.bincount(
            flat, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def per_class_iou(self):
        intersection = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(1) + self.matrix.sum(0) - np.diag(self.matrix)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = intersection / union
        return iou

    def miou(self):
        iou = self.per_class_iou()
        return float(np.nanmean(iou[~np.isnan(iou)])) if np.any(~np.isnan(iou)) else float("nan")

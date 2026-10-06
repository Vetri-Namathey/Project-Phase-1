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
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as label_components

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

    def threshold_at_tpr(self, target_tpr=0.95):
        """Fused-score threshold reaching target_tpr, same cutoff bin fpr_at_tpr
        uses -- so a mask built as `scores >= threshold_at_tpr(t)` reproduces
        fpr_at_tpr(t) when re-scored. Fixed on a val half, then applied
        unchanged elsewhere (never tuned on the set it's evaluated against).
        """
        P = self.n_pos
        if P == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        reached = np.nonzero((tp / P) >= target_tpr)[0]
        if len(reached) == 0:
            return 0.0
        original_bin = self.n_bins - 1 - reached[0]
        return float(original_bin / self.n_bins)

    def threshold_at_max_f1(self):
        """Score threshold maximising pixel-level F1 of `scores >= t`.

        The operating point for ubq_local (ported from exp_v2).
        threshold_at_tpr(0.95) forces 95% recall, which on Fishyscapes flags
        ~7% of every frame (eval_fishyscapes_c5.log) -- a region that big has
        no meaningful outline to measure. Max-F1 balances missed object
        pixels against false alarms instead. Fit on a val half and applied
        unchanged elsewhere, like threshold_at_tpr.
        """
        if self.n_pos == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)
        f1 = 2 * tp / (tp + fp + self.n_pos)
        best = int(np.argmax(f1))
        return float((self.n_bins - 1 - best) / self.n_bins)

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

    def reliability_curve(self, n_bins=15):
        """Per-bin (confidence, accuracy, weight) for a reliability diagram.

        Same binning as ece() above -- this is that computation's per-bin
        detail instead of the single reduced scalar, so the diagram and the
        reported ECE number can never silently disagree.
        """
        total = self.n_pos + self.n_neg
        edges = np.linspace(0, self.n_bins, n_bins + 1).astype(np.int64)
        confs, accs, weights = [], [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            count = self.pos[lo:hi].sum() + self.neg[lo:hi].sum()
            if count == 0:
                confs.append(float("nan"))
                accs.append(float("nan"))
                weights.append(0.0)
                continue
            conf = (self.conf_sum_pos[lo:hi].sum() + self.conf_sum_neg[lo:hi].sum()) / count
            acc = self.pos[lo:hi].sum() / count
            confs.append(float(conf))
            accs.append(float(acc))
            weights.append(float(count) / total if total else 0.0)
        return np.array(confs), np.array(accs), np.array(weights)

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


# ---------------------------------------------------------------------------
# Boundary-only ECE (Novelty 6) + UBQ shared primitives -- see PLAN.md's
# "Boundary-only ECE (Novelty 6) + UBQ" section for the design this
# implements. Plain functions, not a class/module, per this codebase's
# convention of not adding abstractions beyond what's needed.
# ---------------------------------------------------------------------------

def gt_signed_distance(anomaly_mask):
    """Euclidean distance (px) to the ground-truth anomaly edge, negative
    inside the anomaly, positive outside. All-NaN if the mask is empty --
    callers must check and skip that image.
    """
    mask = np.asarray(anomaly_mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    dist_outside = distance_transform_edt(~mask)
    dist_inside = distance_transform_edt(mask)
    signed = np.where(mask, -dist_inside, dist_outside)
    return signed.astype(np.float32)


def boundary_band(anomaly_mask, radius_px, valid=None, side="both"):
    """Pixels within radius_px (Euclidean) of the ground-truth anomaly edge.

    side="both" (default) is symmetric across the edge; "inner"/"outer"
    restrict to just inside / just outside. `valid` (e.g. label != 255)
    is ANDed in afterwards, matching evaluate_fused's existing convention.
    """
    mask = np.asarray(anomaly_mask, dtype=bool)
    dist = gt_signed_distance(mask)
    if np.isnan(dist).all():
        band = np.zeros(mask.shape, dtype=bool)
    elif side == "both":
        band = np.abs(dist) <= radius_px
    elif side == "inner":
        band = (dist <= 0) & (dist >= -radius_px)
    elif side == "outer":
        band = (dist >= 0) & (dist <= radius_px)
    else:
        raise ValueError(f"unknown side {side!r}")
    if valid is not None:
        band = band & np.asarray(valid, dtype=bool)
    return band


def ubq(pred_mask, anomaly_mask, valid=None):
    """Uncertainty Boundary Quality: directed-Hausdorff-style distances
    between a binary predicted region and the ground-truth anomaly extent.

    pred_to_gt_px: how far predicted pixels spill outside the true extent
      (looseness). gt_to_pred_px: how much true extent the prediction
      misses. pred_to_gt_p95 is the 95th percentile of the same distances,
      since the max alone is decided by a single stray false-positive
      pixel anywhere in the frame.

    All distances come from EDT lookups, not
    scipy.spatial.distance.directed_hausdorff -- that function is the
    reference implementation validate_metrics.py's unit check compares
    this against, not the production path (an O(H*W) EDT beats the
    O(n*m) point-set search this would otherwise require per frame).

    Returns NaN for every field if pred_mask or anomaly_mask is empty
    (after `valid` is applied).
    """
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(anomaly_mask, dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        pred = pred & v
        gt = gt & v

    nan_result = {
        "pred_to_gt_px": float("nan"),
        "gt_to_pred_px": float("nan"),
        "pred_to_gt_p95": float("nan"),
    }
    if not pred.any() or not gt.any():
        return nan_result

    dist_from_gt = distance_transform_edt(~gt)
    pred_to_gt_vals = dist_from_gt[pred]

    dist_from_pred = distance_transform_edt(~pred)
    gt_to_pred_vals = dist_from_pred[gt]

    return {
        "pred_to_gt_px": float(pred_to_gt_vals.max()),
        "gt_to_pred_px": float(gt_to_pred_vals.max()),
        "pred_to_gt_p95": float(np.percentile(pred_to_gt_vals, 95)),
    }


def _outline(mask):
    """Pixels of `mask` that touch a pixel outside it (4-neighbourhood)."""
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, border_value=0)


def ubq_local(pred_mask, anomaly_mask, valid=None, roi_px=32, tolerance_px=4):
    """UBQ measured where boundary quality actually lives: object by object.
    Ported from exp_v2.

    ubq() above takes the worst distance anywhere in the frame, so one false
    alarm 1000px from any object decides the score -- that is a detection
    error, not a boundary one. On our model it reads ~1152px for every
    calibrator (eval_fishyscapes_c5.log), i.e. it measures nothing about
    outlines.

    Each ground-truth object (connected component) is scored on its own,
    using only predictions within roi_px of THAT object, so a missed second
    object or a false alarm elsewhere cannot leak into an outline score:

      detected     any prediction within roi_px of the object. Missed
                   objects are counted (object recall), not scored.
      spill_px     mean distance (px) of predicted pixels OUTSIDE the object
                   -- "how far does the flagged region bleed past the edge?"
      miss_px      mean distance (px) from object pixels to the nearest
                   predicted pixel -- "how much of the object is left out?"
      boundary_f1  standard boundary F-score at tolerance_px: share of each
                   outline lying within tolerance_px of the other.

    Returns (objects, far_fp_px, pred_px): a list of per-object dicts
    ({"detected": False} for a missed one), the number of predicted pixels
    further than roi_px from every object, and the total predicted pixels --
    so callers can pool false alarms across images by pixel count. Empty
    ground truth returns ([], pred_px, pred_px): every prediction is far.
    """
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(anomaly_mask, dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        pred, gt = pred & v, gt & v
    if not gt.any():
        return [], int(pred.sum()), int(pred.sum())

    far_fp_px = int((pred & (distance_transform_edt(~gt) > roi_px)).sum())
    labelled, n = label_components(gt)
    margin = roi_px + tolerance_px + 2
    h, w = gt.shape
    objects = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labelled == k)
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, h)
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, w)
        obj = labelled[y0:y1, x0:x1] == k
        dist = distance_transform_edt(~obj)
        local = pred[y0:y1, x0:x1] & (dist <= roi_px)
        if not local.any():
            objects.append({"detected": False})
            continue
        outside = local & ~gt[y0:y1, x0:x1]
        obj_edge, pred_edge = _outline(obj), _outline(local)
        precision = float((distance_transform_edt(~obj_edge)[pred_edge] <= tolerance_px).mean())
        recall = float((distance_transform_edt(~pred_edge)[obj_edge] <= tolerance_px).mean())
        objects.append({
            "detected": True,
            "spill_px": float(dist[outside].mean()) if outside.any() else 0.0,
            "miss_px": float(distance_transform_edt(~local)[obj].mean()),
            "boundary_f1": 0.0 if precision + recall == 0 else
                           2 * precision * recall / (precision + recall),
        })
    return objects, far_fp_px, int(pred.sum())


# ---------------------------------------------------------------------------
# Paired image bootstrap of AP (PLAN.md P1 rule f, P3). AP is not a mean of
# per-image values, so each resample re-pools per-image score histograms and
# recomputes AP from the pooled counts -- the same estimator as the reported
# number, just on resampled images.
# ---------------------------------------------------------------------------

def _pooled_ap(pos, neg):
    hist = ScoreHistogram(n_bins=pos.shape[-1])
    hist.pos, hist.neg = pos, neg
    return hist.average_precision()


def paired_bootstrap_ap(per_image_a, per_image_b, n_resamples=1000, seed=config.GLOBAL_SEED):
    """95% CI of AP(a) - AP(b), resampling the SAME image indices for both.

    per_image_a / per_image_b: equal-length lists of (pos, neg) count arrays,
    one pair per image, from ScoreHistogram(n_bins=DEFAULT_BINS) -- the full
    resolution, so the pooled AP equals the reported one. A resample is a
    vector of per-image counts w, so pooled counts are w @ stack: one
    matrix-vector product instead of copying the 50 x 200k stack each round.

    Returns {"ap_a", "ap_b", "diff", "lo", "hi"} (full-sample APs and diff).
    """
    assert len(per_image_a) == len(per_image_b) > 0, "need paired, non-empty inputs"
    stacks = []
    for per_image in (per_image_a, per_image_b):
        pos = np.stack([p for p, _ in per_image]).astype(np.float64)
        neg = np.stack([n for _, n in per_image]).astype(np.float64)
        stacks.append((pos, neg))
    n = len(per_image_a)
    full = [_pooled_ap(pos.sum(0), neg.sum(0)) for pos, neg in stacks]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        aps = [_pooled_ap(w @ pos, w @ neg) for pos, neg in stacks]
        diffs[i] = aps[0] - aps[1]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"ap_a": full[0], "ap_b": full[1], "diff": full[0] - full[1],
            "lo": float(lo), "hi": float(hi)}

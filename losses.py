"""Loss functions: L_total = L_seg + alpha * L_OOD_total + beta * L_calib.

Phase 2a trains with the first two terms only (train.py). The third,
L_calib, is added by the Phase 2b fine-tune in calibrate.py -- SoftECELoss
below, optionally restricted to a band around each object's edge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def build_seg_criterion(class_weights=None, ignore_index=255):
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)


def ood_bce_loss(ood_logit, target_mask, pos_weight=None):
    """Binary cross-entropy on LOGITS with a fixed positive-class weight.

    Two changes from the earlier version, both aimed at the collapse:

    1. with_logits. The previous code applied sigmoid in the head and then
       called F.binary_cross_entropy on the result. Under this much class
       imbalance the heads drive their pre-activations deeply negative, and
       once there the gradient through the explicit sigmoid vanishes -- the
       head cannot recover regardless of what later batches contain. That is
       the signature the collapse check found. Folding the sigmoid into the
       loss keeps the gradient well-conditioned at the same operating point.

    2. Fixed pos_weight instead of per-batch inverse frequency. The old
       weight was recomputed from whatever happened to be pasted into the
       current batch and was capped at 100, so the effective gradient scale
       swung by two orders of magnitude between consecutive steps -- and
       collapsed to 1.0 entirely on batches with no pasted object. A fixed
       weight makes the objective stationary, which is a precondition for
       the optimiser converging to anything stable.
    """
    if pos_weight is None:
        pos_weight = config.OOD_POS_WEIGHT
    weight_t = torch.as_tensor(pos_weight, dtype=ood_logit.dtype,
                               device=ood_logit.device)
    return F.binary_cross_entropy_with_logits(
        ood_logit, target_mask, pos_weight=weight_t)


def compute_total_loss(seg_logits, seg_labels, ood_logits, ood_target,
                       seg_criterion, alpha=config.ALPHA_OOD):
    """L_total = L_seg + alpha * L_OOD_total, per the paper's Part-3 block.

    ood_logits: (B, num_heads, H, W) raw logits, NOT probabilities.

    The paper writes the per-head aggregation as a SUM:

        L_OOD_total = L_OOD(head1) + L_OOD(head2) + L_OOD(head3)

    This used to average instead. Sum and mean differ only by a constant
    factor of num_heads, which alpha absorbs entirely -- so the previous code
    was not wrong, it was the paper's formula with alpha implicitly divided by
    three. It is now written as the paper writes it, with that factor moved
    into ALPHA_OOD where it is visible.

    This matters more than cosmetics from Phase 2b onward: once
    beta * L_calib joins the objective, three terms have to be balanced
    against each other, and that is far harder to reason about when one of
    them carries a hidden 1/3 that does not appear in any equation.
    """
    l_seg = seg_criterion(seg_logits, seg_labels)
    num_heads = ood_logits.shape[1]
    per_head = torch.stack([
        ood_bce_loss(ood_logits[:, h], ood_target) for h in range(num_heads)
    ])
    l_ood = per_head.sum() if config.OOD_HEAD_AGGREGATION == "sum" else per_head.mean()
    total = l_seg + alpha * l_ood
    return total, l_seg, l_ood


class SoftECELoss(nn.Module):
    """Differentiable surrogate of Expected Calibration Error -- L_calib.

    Hard-binned ECE (metrics.ScoreHistogram.ece) has zero gradient almost
    everywhere: moving a score slightly does not change which bin it is in.
    Here each score is spread over its two nearest bin centres with triangular
    weights, so bin confidence and bin accuracy become smooth functions of
    the scores and the gap between them can be trained on.

    Squared gap rather than absolute: same minimum, but the gradient does not
    flip sign discontinuously at a perfectly calibrated bin.

    Temperature scaling can only stretch every score by one global factor.
    This loss can move scores in one bin without moving another, which is
    what reshaping calibration near object edges needs -- see calibrate.py.
    """

    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins
        edges = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        self.width = 1.0 / n_bins

    def forward(self, scores, targets):
        scores = scores.reshape(-1)
        targets = targets.reshape(-1).float()
        n = scores.numel()
        if n == 0:
            return scores.new_zeros(())

        dist = (scores.unsqueeze(1) - self.centers.unsqueeze(0)).abs()
        weight = torch.clamp(1.0 - dist / self.width, min=0.0)   # (n, bins)

        bin_mass = weight.sum(dim=0)
        bin_conf = (weight * scores.unsqueeze(1)).sum(dim=0)
        bin_acc = (weight * targets.unsqueeze(1)).sum(dim=0)

        safe_mass = bin_mass.clamp(min=1e-8)
        conf = bin_conf / safe_mass
        acc = bin_acc / safe_mass
        return ((bin_mass / n) * (conf - acc) ** 2).sum()


def boundary_band_torch(ood_target, radius_px):
    """(B,H,W) binary target -> (B,H,W) bool, pixels within radius_px of an
    object edge (both sides).

    GPU counterpart of metrics.boundary_band, used to restrict L_calib to
    the same region the boundary-ECE metric scores. Square neighbourhood
    (max-pool) where the metric uses a Euclidean distance -- slightly wider
    at the corners, which is harmless for a training mask and avoids a
    distance transform on every batch.

    Images with no pasted object get an empty band.
    """
    if radius_px <= 0:
        return torch.ones_like(ood_target, dtype=torch.bool)
    k = 2 * radius_px + 1
    t = ood_target.unsqueeze(1).float()
    dilated = F.max_pool2d(t, k, stride=1, padding=radius_px)
    eroded = -F.max_pool2d(-t, k, stride=1, padding=radius_px)
    return (dilated > eroded).squeeze(1)

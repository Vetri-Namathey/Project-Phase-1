"""Loss functions: L_total = L_seg + alpha * L_OOD + beta * L_calib.

L_calib (SoftECELoss below) is Phase 2b -- a differentiable soft-binning
surrogate for real ECE, since metrics.py's histogram-based ECE is reporting
-only (hard bin edges, no gradient). See PLAN.md's "Calibration (L_calib)
gate" section for why this has to be a joint fine-tune rather than post-hoc
temperature scaling: a scalar rescale cannot reshape calibration spatially,
which is the whole point (beating temp scaling on boundary-region ECE/UBQ).
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
    """ood_logits: (B, num_heads, H, W) raw logits, NOT probabilities."""
    l_seg = seg_criterion(seg_logits, seg_labels)
    num_heads = ood_logits.shape[1]
    l_ood = torch.stack([
        ood_bce_loss(ood_logits[:, h], ood_target) for h in range(num_heads)
    ]).mean()
    total = l_seg + alpha * l_ood
    return total, l_seg, l_ood


class SoftECELoss(nn.Module):
    """Differentiable surrogate for Expected Calibration Error.

    metrics.py's ScoreHistogram.ece() uses HARD bin edges (np.clip + integer
    bin index) -- exactly what real ECE is defined as, but with zero gradient
    w.r.t. the score, so it cannot be a training loss. This replaces the hard
    indicator "is this score in bin k?" with a soft one: a narrow triangular
    kernel centered on each bin, so every score contributes (mostly) to its
    nearest bin but the contribution varies smoothly as the score moves. As
    the kernel narrows, this converges to the real histogram ECE.

    Matches metrics.py's convention: n_bins=15 equal-width bins over [0,1],
    "confidence" = the raw score (P(anomalous)), "accuracy" = the empirical
    positive rate within a bin -- i.e. this is calibration of the anomaly
    score itself, not top-1 classification confidence.
    """

    def __init__(self, n_bins=15, sharpness=None):
        super().__init__()
        self.n_bins = n_bins
        edges = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        # Triangular half-width = one full bin, so a score's weight is spread
        # over roughly its own bin and its two neighbours -- enough overlap
        # to keep gradients flowing into empty-looking bins, not so much that
        # bins blur into a single global average.
        self.width = 1.0 / n_bins
        self.sharpness = sharpness

    def forward(self, scores, targets):
        """scores, targets: any shape, flattened internally. targets in
        {0, 1} (float). Returns a scalar loss, differentiable w.r.t. scores.
        """
        scores = scores.reshape(-1)
        targets = targets.reshape(-1).float()
        n = scores.numel()
        if n == 0:
            return scores.new_zeros(())

        # (n, n_bins) soft membership: triangular kernel, clipped at 0, so a
        # score contributes only to nearby bins (matches the reporting ECE's
        # single-bin membership in the limit width -> 0).
        dist = (scores.unsqueeze(1) - self.centers.unsqueeze(0)).abs()
        weight = torch.clamp(1.0 - dist / self.width, min=0.0)

        bin_mass = weight.sum(dim=0)                      # (n_bins,)
        bin_conf = (weight * scores.unsqueeze(1)).sum(dim=0)
        bin_acc = (weight * targets.unsqueeze(1)).sum(dim=0)

        safe_mass = bin_mass.clamp(min=1e-8)
        conf = bin_conf / safe_mass
        acc = bin_acc / safe_mass

        # Squared, not absolute, difference: abs() has a zero/undefined
        # gradient exactly at conf == acc, which is where a well-calibrated
        # bin sits -- squaring keeps the gradient smooth everywhere and only
        # changes the loss's scale, not what minimizes it.
        per_bin = (bin_mass / n) * (conf - acc) ** 2
        return per_bin.sum()

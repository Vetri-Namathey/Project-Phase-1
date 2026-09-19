"""Loss functions: L_total = L_seg + alpha * L_OOD.

No L_calib here -- that is Phase 2b scope, implemented separately later as
its own differentiable soft-binning surrogate.
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

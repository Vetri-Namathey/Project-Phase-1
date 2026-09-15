"""Loss functions (Section 03): L_total = L_seg + alpha * L_OOD.

No L_calib here -- that is Phase 2b / Section 13 scope, implemented
separately later as its own differentiable soft-binning surrogate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def build_seg_criterion(class_weights=None, ignore_index=255):
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)


def ood_bce_loss(ood_score, target_mask, pos_weight_cap=100.0):
    """CutMix triggers on only ~50% of images, and even then covers just
    15-35% of the frame's shorter dimension -- OOD (positive) pixels are a
    small fraction of any batch, well under 1% on average. Unweighted BCE
    lets the model take the easy route: predict "not OOD" everywhere, which
    minimises loss while being completely non-discriminative (this is what
    produced the ~random AUROC in the first real run). Upweight positive
    pixels by their inverse frequency in this batch, capped to avoid
    instability on batches with almost no positive pixels at all.
    """
    num_pos = target_mask.sum()
    num_neg = target_mask.numel() - num_pos
    if num_pos > 0:
        pos_weight_value = torch.clamp(num_neg / num_pos, max=pos_weight_cap)
    else:
        pos_weight_value = torch.tensor(1.0, device=target_mask.device)
    weight = 1.0 + (pos_weight_value - 1.0) * target_mask
    return F.binary_cross_entropy(ood_score, target_mask, weight=weight)


def compute_total_loss(seg_logits, seg_labels, ood_scores, ood_target, seg_criterion, alpha=config.ALPHA_OOD):
    l_seg = seg_criterion(seg_logits, seg_labels)
    num_heads = ood_scores.shape[1]
    l_ood = torch.stack([
        ood_bce_loss(ood_scores[:, h], ood_target) for h in range(num_heads)
    ]).mean()
    total = l_seg + alpha * l_ood
    return total, l_seg, l_ood

"""Evaluation metrics shared by Experiment A and Experiment B (Section 07)."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(scores, labels):
    return float(roc_auc_score(labels, scores))


def compute_fpr_at_tpr(scores, labels, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = min(int(np.searchsorted(tpr, target_tpr)), len(fpr) - 1)
    return float(fpr[idx])


def compute_ece(confidences, labels, n_bins=15):
    """Hard-binning Expected Calibration Error -- EVALUATION REPORTING ONLY.

    This is not used as a training loss. The differentiable soft-binning ECE
    surrogate (Karandikar et al. ICML 2021) used as L_calib during Phase 2b
    training is a separate implementation -- it goes in losses.py later, not
    here. Do not import this function as a loss.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)

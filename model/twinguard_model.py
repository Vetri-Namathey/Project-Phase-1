"""TwinGuard model: frozen encoder + segmentation head + N independently
seeded OOD heads.
"""

import torch
import torch.nn as nn

import config
from model.encoder import build_encoder
from model.ood_head import OODHead
from model.seg_head import SegmentationHead


class TwinGuardModel(nn.Module):
    def __init__(self, num_ood_heads=3, ood_seeds=None,
                 num_seg_classes=config.NUM_SEG_CLASSES):
        super().__init__()
        self.encoder = build_encoder()
        in_channels_list = self.encoder.config.hidden_sizes

        self.seg_head = SegmentationHead(in_channels_list, num_seg_classes)

        if ood_seeds is None:
            ood_seeds = (config.OOD_HEAD_SEEDS_3HEAD if num_ood_heads == 3
                         else config.OOD_HEAD_SEEDS_1HEAD)
        assert len(ood_seeds) == num_ood_heads, "seed list must match num_ood_heads"

        self.ood_heads = nn.ModuleList([
            OODHead(in_channels_list, dropout_p=config.OOD_HEAD_DROPOUT_P, seed=seed)
            for seed in ood_seeds
        ])

    def forward(self, images):
        output_size = images.shape[-2:]
        with torch.no_grad():
            enc_out = self.encoder(pixel_values=images, output_hidden_states=True)
        hidden_states = enc_out.hidden_states

        seg_logits = self.seg_head(hidden_states, output_size)

        # (B, num_heads, H, W) logits -- the training signal.
        ood_logits = torch.stack(
            [head(hidden_states, output_size) for head in self.ood_heads], dim=1)
        ood_scores = torch.sigmoid(ood_logits)

        # Disagreement between independently-seeded heads is the project's
        # epistemic uncertainty signal. Computed and returned here so it can
        # actually be measured, rather than being an architectural claim with
        # nothing reading it. std over heads; 0 for a single-head model.
        if ood_scores.shape[1] > 1:
            ood_disagreement = ood_scores.std(dim=1, unbiased=False)
        else:
            ood_disagreement = torch.zeros_like(ood_scores[:, 0])

        return {
            "seg_logits": seg_logits,
            "ood_logits": ood_logits,
            "ood_scores": ood_scores,
            "ood_fused": ood_scores.mean(dim=1),
            "ood_disagreement": ood_disagreement,
        }

    # -----------------------------------------------------------------
    # Dual-mode inference (paper Section III-A)
    # -----------------------------------------------------------------
    # The abstract's second contribution: "a dual-mode inference strategy
    # will confine stochastic forward passes to a lightweight multi-head
    # ensemble over a frozen SegFormer-B5 encoder, reserving dense spatial
    # uncertainty estimation for safety-triggered events while a single
    # deterministic pass sustains continuous operation within real-time
    # bounds."
    #
    # The whole latency argument rests on one fact: the encoder runs ONCE.
    # N=10 MC-Dropout passes over the full B5 backbone would cost >1000ms.
    # Running them over the two-conv heads, reusing cached features, is
    # cheap. encode() exists to make that reuse explicit and impossible to
    # get wrong by accident.

    @torch.no_grad()
    def encode(self, images):
        """Runs the frozen encoder once and returns its multi-scale features.

        Split out from forward() so the safety path can reuse one encoder
        pass across many head passes -- which is the entire basis of the
        latency claim.
        """
        enc_out = self.encoder(pixel_values=images, output_hidden_states=True)
        return enc_out.hidden_states

    @torch.no_grad()
    def heads_from_features(self, hidden_states, output_size, stochastic=False):
        """One pass of all OOD heads over cached features.

        stochastic=True activates dropout in the heads ONLY. The encoder is
        untouched and stays in eval mode -- its features are already computed
        and cached, so there is nothing stochastic about them by construction.
        """
        was_training = [head.training for head in self.ood_heads]
        for head in self.ood_heads:
            head.train(stochastic)
        try:
            logits = torch.stack(
                [head(hidden_states, output_size) for head in self.ood_heads], dim=1)
        finally:
            for head, prev in zip(self.ood_heads, was_training):
                head.train(prev)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_dual_mode(self, images, n_passes=None, trigger_threshold=None,
                          force_safety=False):
        """Continuous mode, escalating to safety-triggered mode on demand.

        Continuous: encoder once, heads once, dropout OFF, score = mean of
        the 3 heads. This is what runs on every frame.

        Safety-triggered (only when the anomaly score crosses the threshold):
        the SAME cached encoder features are reused while each head runs
        n_passes times with dropout ON. Per Section III-A this yields two
        distinct uncertainties, which the paper is careful to separate:

          epistemic  -- variance across one head's own N passes, averaged
                        over heads. "This model is unstable on this input."
          parametric -- variance of the 3 heads' mean predictions from each
                        other. "Three independently-converged decision
                        boundaries disagree", which is the signal that
                        actually responds to genuine novelty.

        Returns a dict; `triggered` says which mode ran.
        """
        n_passes = n_passes or config.MC_DROPOUT_PASSES
        if trigger_threshold is None:
            trigger_threshold = config.SAFETY_TRIGGER_THRESHOLD

        output_size = images.shape[-2:]
        hidden_states = self.encode(images)          # <-- the only encoder pass

        scores = self.heads_from_features(hidden_states, output_size, stochastic=False)
        fused = scores.mean(dim=1)

        # Per-image trigger: does ANY pixel look anomalous enough to be worth
        # the expensive path? max over pixels, per image in the batch.
        peak = fused.flatten(1).max(dim=1).values
        triggered = bool(force_safety or (peak >= trigger_threshold).any())

        result = {
            "seg_logits": self.seg_head(hidden_states, output_size),
            "ood_scores": scores,
            "ood_fused": fused,
            "peak_score": peak,
            "triggered": triggered,
            "mode": "safety" if triggered else "continuous",
            "n_passes": n_passes if triggered else 1,
        }

        if not triggered:
            # Continuous mode still reports head disagreement -- it is free,
            # since all three heads already ran.
            result["parametric_uncertainty"] = (
                scores.std(dim=1, unbiased=False) if scores.shape[1] > 1
                else torch.zeros_like(fused))
            result["epistemic_uncertainty"] = None
            return result

        # (n_passes, B, num_heads, H, W) -- encoder features reused throughout.
        stochastic = torch.stack([
            self.heads_from_features(hidden_states, output_size, stochastic=True)
            for _ in range(n_passes)
        ], dim=0)

        per_head_mean = stochastic.mean(dim=0)                      # (B,heads,H,W)
        per_head_var = stochastic.var(dim=0, unbiased=False)        # (B,heads,H,W)

        result["epistemic_uncertainty"] = per_head_var.mean(dim=1)  # avg over heads
        result["parametric_uncertainty"] = (
            per_head_mean.std(dim=1, unbiased=False) if per_head_mean.shape[1] > 1
            else torch.zeros_like(fused))
        result["ood_fused_mc"] = per_head_mean.mean(dim=1)
        return result

    def trainable_parameters(self):
        return list(self.seg_head.parameters()) + list(self.ood_heads.parameters())


def verify_encoder_frozen(model):
    return all(not p.requires_grad for p in model.encoder.parameters())


def verify_heads_independent(model):
    flat = [torch.cat([p.detach().flatten() for p in head.parameters()])
            for head in model.ood_heads]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if torch.equal(flat[i], flat[j]):
                return False
    return True

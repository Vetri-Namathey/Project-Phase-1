"""TwinGuard model (Section 02): frozen encoder + segmentation head + N
independently-seeded OOD heads.
"""

import torch
import torch.nn as nn

import config
from model.encoder import build_encoder
from model.ood_head import OODHead
from model.seg_head import SegmentationHead


class TwinGuardModel(nn.Module):
    def __init__(self, num_ood_heads=3, ood_seeds=None, num_seg_classes=config.NUM_SEG_CLASSES):
        super().__init__()
        self.encoder = build_encoder()
        in_channels_list = self.encoder.config.hidden_sizes

        self.seg_head = SegmentationHead(in_channels_list, num_seg_classes)

        if ood_seeds is None:
            ood_seeds = config.OOD_HEAD_SEEDS_3HEAD if num_ood_heads == 3 else config.OOD_HEAD_SEEDS_1HEAD
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
        ood_scores = torch.stack(
            [head(hidden_states, output_size) for head in self.ood_heads], dim=1
        )  # (B, num_heads, H, W)

        return {
            "seg_logits": seg_logits,
            "ood_scores": ood_scores,
            "ood_fused": ood_scores.mean(dim=1),
        }

    def trainable_parameters(self):
        return list(self.seg_head.parameters()) + list(self.ood_heads.parameters())


def verify_encoder_frozen(model):
    return all(not p.requires_grad for p in model.encoder.parameters())


def verify_heads_independent(model):
    flat = [torch.cat([p.detach().flatten() for p in head.parameters()]) for head in model.ood_heads]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if torch.equal(flat[i], flat[j]):
                return False
    return True

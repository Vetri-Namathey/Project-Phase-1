"""Segmentation head (Section 02, HEAD 1): multi-scale fusion decode head,
same project-upsample-concat-fuse pattern as HuggingFace's own
SegformerDecodeHead (verified against their source before writing this).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, num_classes, embed_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, hidden_states, output_size):
        target_hw = hidden_states[0].shape[-2:]
        feats = []
        for proj, hs in zip(self.proj, hidden_states):
            f = proj(hs)
            f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            feats.append(f)
        fused = self.fuse(torch.cat(feats, dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

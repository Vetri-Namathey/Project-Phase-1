"""Independent OOD head (Section 02, HEADS 2/3/4): 2 conv layers + dropout
+ sigmoid. Each head consumes the same raw multi-scale encoder features as
the segmentation head, but with its own non-shared weights -- zero
trainable parameters are shared between heads, so their disagreement is a
real independence signal, not an artifact of a shared upstream layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OODHead(nn.Module):
    def __init__(self, in_channels_list, dropout_p=0.3, hidden_dim=128, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        total_in = sum(in_channels_list)
        self.conv1 = nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, hidden_states, output_size):
        target_hw = hidden_states[0].shape[-2:]
        feats = [F.interpolate(hs, size=target_hw, mode="bilinear", align_corners=False) for hs in hidden_states]
        x = torch.cat(feats, dim=1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x).squeeze(1)

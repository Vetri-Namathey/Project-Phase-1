"""Independent OOD head: 2 conv layers + dropout, emitting LOGITS.

Each head consumes the same raw multi-scale encoder features as the
segmentation head but with its own non-shared weights -- zero trainable
parameters are shared between heads, so their disagreement is a real
independence signal rather than an artifact of a shared upstream layer.

The head returns logits, not probabilities. Two reasons:

  1. Training stability. Applying sigmoid here and then F.binary_cross_
     entropy downstream is the standard saturation trap: once the
     pre-activations go deeply negative under heavy class imbalance the
     gradient path through sigmoid collapses, and the head gets stuck
     predicting the base rate. That is the exact signature the collapse
     diagnostic found (mean ~0.001, std ~0.01, unmoved by any data change).
     binary_cross_entropy_with_logits folds the sigmoid into the loss and
     keeps the gradient well-conditioned.

  2. Temperature scaling operates on logits. The planned Phase 3 ablation
     ("why not just apply the simple post-hoc fix?") is not implementable
     against a head that only ever emits a probability.

Call sigmoid at the point of use -- TwinGuardModel does this and exposes
both forms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OODHead(nn.Module):
    def __init__(self, in_channels_list, dropout_p=0.3, hidden_dim=128, seed=None):
        super().__init__()
        total_in = sum(in_channels_list)
        self.conv1 = nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

        if seed is not None:
            self._reinit(seed)

    def _reinit(self, seed):
        """Re-initialise this head's weights from a dedicated seed.

        Uses a local torch.Generator rather than torch.manual_seed. The
        previous version reseeded the GLOBAL RNG inside __init__, which meant
        constructing the model silently reset global randomness to the last
        head's seed -- overriding config.GLOBAL_SEED for data shuffling and
        dropout, so runs were not reproducible in the way the config claimed.
        """
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for module in (self.conv1, self.conv2):
            fan_in = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
            bound = (1.0 / fan_in) ** 0.5
            with torch.no_grad():
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, hidden_states, output_size):
        """Returns per-pixel LOGITS of shape (B, H, W)."""
        target_hw = hidden_states[0].shape[-2:]
        feats = [
            F.interpolate(hs, size=target_hw, mode="bilinear", align_corners=False)
            for hs in hidden_states
        ]
        x = torch.cat(feats, dim=1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x.squeeze(1)

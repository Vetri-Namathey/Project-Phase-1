"""Shared input preprocessing.

One place for the encoder's expected input statistics, so the training
loader, the Fishyscapes evaluator and the diagnostic scripts cannot drift
apart. They previously did: Experiment A went through the HuggingFace
processor (which normalises) while Experiment B fed raw [0,1] tensors,
which meant the two experiments were not being shown the same thing.
"""

import numpy as np
import torch
from PIL import Image

import config

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


def normalize_imagenet(image_t):
    """(3,H,W) or (B,3,H,W) float tensor in [0,1] -> encoder-space tensor.

    The SegFormer encoder is frozen, so it cannot adapt to a different input
    range -- getting this wrong costs real AUROC rather than just slowing
    convergence.
    """
    mean = _MEAN.to(image_t.device, image_t.dtype)
    std = _STD.to(image_t.device, image_t.dtype)
    if image_t.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (image_t - mean) / std


def denormalize_imagenet(image_t):
    """Inverse of normalize_imagenet, for overlay/debug visualisations."""
    mean = _MEAN.to(image_t.device, image_t.dtype)
    std = _STD.to(image_t.device, image_t.dtype)
    if image_t.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (image_t * std + mean).clamp(0.0, 1.0)


def load_image_tensor(image_path, device=None):
    """Disk -> (1,3,H,W) tensor in encoder space.

    The single entry point for turning an image file into model input, used
    by the Fishyscapes evaluator and every diagnostic script. It lives here
    rather than in a training script because the one bug this module exists
    to prevent -- evaluation preprocessing drifting away from training
    preprocessing -- is exactly what happens when each caller rolls its own.
    """
    image = Image.open(image_path).convert("RGB").resize(
        (config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
    tensor = tensor.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = normalize_imagenet(tensor)
    return tensor.to(device) if device is not None else tensor

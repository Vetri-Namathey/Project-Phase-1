"""Frozen SegFormer MiT encoder (Section 02)."""

from transformers import SegformerModel

import config


def build_encoder():
    encoder = SegformerModel.from_pretrained(config.ENCODER_NAME)
    for param in encoder.parameters():
        param.requires_grad = False  # CRITICAL -- encoder never updates
    encoder.eval()
    return encoder

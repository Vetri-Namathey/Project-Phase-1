"""Anomaly object sources for CutMix outlier exposure.

CutMix needs exactly one thing from a source: an (RGB crop, binary mask)
pair, both numpy, both the same HxW, mask in {0,1}. CARLA produces that via
its semantic-segmentation camera; COCO produces it via human-traced polygon
outlines. Everything downstream is identical, so the two live behind one
interface here and nothing else in the pipeline needs to know which is in
use.

CARLA is the project's canonical source. COCO exists as a temporary
stand-in for the period where no CARLA server is available, and is selected
via config.ANOMALY_SOURCE.
"""

import glob
import os
import random

import numpy as np
from PIL import Image

import config


class _ObjectBank:
    """A list of (image_path, mask_path) pairs, sampled uniformly."""

    def __init__(self, pairs, name):
        self.pairs = pairs
        self.name = name

    def __len__(self):
        return len(self.pairs)

    def sample(self, rng=None):
        """Returns (rgb_crop, binary_mask) cropped to the object's bounding
        box, or None if this object's mask turned out to be empty."""
        pick = (rng or random).randrange(len(self.pairs))
        image_path, mask_path = self.pairs[pick]

        image = np.array(Image.open(image_path).convert("RGB"))
        if mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"{self.name}: mask {mask.shape[:2]} does not match image "
                f"{image.shape[:2]} for {os.path.basename(image_path)}"
            )

        ys, xs = np.where(mask == 1)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _build_carla_bank():
    images = sorted(glob.glob(os.path.join(config.CARLA_IMAGES_DIR, "*.png")))
    masks = sorted(glob.glob(os.path.join(config.CARLA_MASKS_DIR, "*.npy")))
    if len(images) == 0:
        raise FileNotFoundError(
            f"ANOMALY_SOURCE='carla' but no images found in "
            f"{config.CARLA_IMAGES_DIR}. Run generate_anomalies.py against a "
            f"running CARLA server, or set ANOMALY_SOURCE='coco' in config.py."
        )
    if len(images) != len(masks):
        raise ValueError(
            f"CARLA images/masks must pair 1:1 -- found {len(images)} images "
            f"and {len(masks)} masks."
        )
    return _ObjectBank(list(zip(images, masks)), "carla")


def _build_coco_bank():
    images = sorted(glob.glob(os.path.join(config.COCO_OBJECTS_DIR, "*_rgb.png")))
    pairs = []
    for image_path in images:
        mask_path = image_path.replace("_rgb.png", "_mask.png")
        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))
    if len(pairs) == 0:
        raise FileNotFoundError(
            f"ANOMALY_SOURCE='coco' but no cutouts found in "
            f"{config.COCO_OBJECTS_DIR}. Run: python download_coco_anomalies.py"
        )
    return _ObjectBank(pairs, "coco")


def _build_both_bank():
    """Pools curated-CARLA and exclusion-filtered-COCO pairs into one bank,
    sampled uniformly per paste (not source-then-object). Reduces the risk of
    the model locking onto either source's own low-level statistical
    signature (CG-render tells vs. COCO-photo tells) as a shortcut, on top of
    just adding volume.

    Rebalanced per config.CARLA_BANK_TARGET / config.COCO_BANK_TARGET
    (PLAN.md, 2026-09-21): only 45 distinct CARLA objects exist, so reaching
    CARLA_BANK_TARGET repeats those same 45 files rather than adding new
    ones (no CARLA server here to render more) -- this shifts the CARLA:COCO
    sampling ratio without adding CARLA visual diversity. COCO is a seeded
    random subsample so the run is reproducible across machines.
    """
    carla_bank = _build_carla_bank()
    coco_bank = _build_coco_bank()

    carla_target = getattr(config, "CARLA_BANK_TARGET", len(carla_bank.pairs))
    coco_target = getattr(config, "COCO_BANK_TARGET", len(coco_bank.pairs))

    carla_pairs = list(carla_bank.pairs)
    if carla_target > len(carla_pairs):
        reps = -(-carla_target // len(carla_pairs))  # ceil div
        carla_pairs = (carla_pairs * reps)[:carla_target]
    elif carla_target < len(carla_pairs):
        carla_pairs = random.Random(config.GLOBAL_SEED).sample(carla_pairs, carla_target)

    coco_pairs = list(coco_bank.pairs)
    if coco_target < len(coco_pairs):
        coco_pairs = random.Random(config.GLOBAL_SEED).sample(coco_pairs, coco_target)
    elif coco_target > len(coco_pairs):
        raise ValueError(
            f"config.COCO_BANK_TARGET={coco_target} exceeds the {len(coco_pairs)} "
            f"COCO objects on disk -- lower the target or re-run "
            f"download_coco_anomalies.py for more."
        )

    return _ObjectBank(carla_pairs + coco_pairs, "both")


_BUILDERS = {"carla": _build_carla_bank, "coco": _build_coco_bank, "both": _build_both_bank}


def build_anomaly_bank(source=None):
    """Returns the object bank named by config.ANOMALY_SOURCE (or `source`)."""
    source = (source or config.ANOMALY_SOURCE).lower()
    if source not in _BUILDERS:
        raise ValueError(
            f"unknown ANOMALY_SOURCE {source!r} -- expected one of "
            f"{sorted(_BUILDERS)}"
        )
    bank = _BUILDERS[source]()
    print(f"anomaly source: {bank.name} ({len(bank)} objects)")
    return bank

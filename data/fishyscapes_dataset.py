"""Fishyscapes Lost & Found pairing -- EVALUATION ONLY, never wired into a
training DataLoader.

The OOD ground-truth labels (from Zenodo) and the underlying RGB images
(from the Lost&Found dataset) come from two unrelated sources with no shared
folder tree, so pairing them requires filename matching: a label named
"{idx}_{city}_{seq}_{frame}_labels.png" corresponds to the image
"{city}_{seq}_{frame}_leftImg8bit.png" somewhere under
FISHYSCAPES_IMAGES_ROOT/{train,test}/{city}/. Verified 100/100 match.

Label convention (verified by inspecting real pixel values): 0 = normal,
1 = OOD/anomaly, 255 = ignore/void. Measured composition across all 100
images: 170,619,888 valid pixels, 477,664 of them anomalous -- a 0.28%
positive rate, which is why ECE alone is a misleading headline number.
"""

import glob
import os
import random
import re

import config


def list_fishyscapes_pairs():
    """All (image_path, label_path) tuples, sorted deterministically."""
    label_paths = sorted(glob.glob(
        os.path.join(config.FISHYSCAPES_LABELS_DIR, "*_labels.png")))
    if not label_paths:
        raise FileNotFoundError(
            f"no Fishyscapes labels under {config.FISHYSCAPES_LABELS_DIR}")

    img_index = {}
    for split in ("train", "test"):
        pattern = os.path.join(
            config.FISHYSCAPES_IMAGES_ROOT, split, "*", "*_leftImg8bit.png")
        for p in glob.glob(pattern):
            key = os.path.basename(p).replace("_leftImg8bit.png", "")
            img_index[key] = p

    pairs = []
    missing = []
    for label_path in label_paths:
        fname = os.path.basename(label_path)
        match = re.match(r"^\d+_(.+)_labels\.png$", fname)
        if match is None:
            missing.append(fname)
            continue
        key = match.group(1)
        if key not in img_index:
            missing.append(fname)
            continue
        pairs.append((img_index[key], label_path))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} Fishyscapes labels have no matching image "
            f"(first few: {missing[:3]}). Check FISHYSCAPES_IMAGES_ROOT.")
    return pairs


def split_fishyscapes_pairs(pairs=None,
                            val_fraction=config.FISHYSCAPES_VAL_FRACTION,
                            seed=config.FISHYSCAPES_SPLIT_SEED):
    """Deterministic (val, test) split.

    Checkpoint selection must read only the val half. Picking the best epoch
    by score on the same images that get reported is model selection on the
    test set: it biases every reported number upward even though Fishyscapes
    never enters training, and it undercuts the strict-dataset rule the
    project states prominently.

    The split is seeded and sorted-input-based, so it is identical across
    machines and across runs -- the test half is never silently reshuffled
    into the val half between experiments.
    """
    pairs = pairs if pairs is not None else list_fishyscapes_pairs()
    ordered = sorted(pairs, key=lambda pair: os.path.basename(pair[1]))
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    return shuffled[:n_val], shuffled[n_val:]

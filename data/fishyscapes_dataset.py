"""Fishyscapes Lost & Found pairing (Section 05 -- EVALUATION ONLY, never
wired into a training DataLoader).

The OOD ground-truth labels (from Zenodo) and the underlying RGB images
(from the Lost&Found dataset / HF mirror) come from two unrelated sources
with no shared folder tree, so pairing them requires filename matching:
a label named "{idx}_{city}_{seq}_{frame}_labels.png" corresponds to the
image "{city}_{seq}_{frame}_leftImg8bit.png" somewhere under
FISHYSCAPES_IMAGES_ROOT/{train,test}/{city}/. Verified 100/100 match.

Label convention (verified by inspecting real pixel values): 0 = normal,
1 = OOD/anomaly, 255 = ignore/void.
"""

import glob
import os
import re

import config


def list_fishyscapes_pairs():
    """Returns a list of (image_path, label_path) tuples for all Fishyscapes
    Lost & Found validation images."""
    label_paths = sorted(glob.glob(os.path.join(config.FISHYSCAPES_LABELS_DIR, "*_labels.png")))

    img_index = {}
    for split in ("train", "test"):
        pattern = os.path.join(config.FISHYSCAPES_IMAGES_ROOT, split, "*", "*_leftImg8bit.png")
        for p in glob.glob(pattern):
            key = os.path.basename(p).replace("_leftImg8bit.png", "")
            img_index[key] = p

    pairs = []
    for label_path in label_paths:
        fname = os.path.basename(label_path)
        match = re.match(r"^\d+_(.+)_labels\.png$", fname)
        key = match.group(1)
        image_path = img_index[key]
        pairs.append((image_path, label_path))

    return pairs

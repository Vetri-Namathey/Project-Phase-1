"""Cityscapes leftImg8bit / gtFine loader.

Layout:
    CITYSCAPES_IMAGES_ROOT/{split}/{city}/{city}_{seq}_{frame}_leftImg8bit.png
    CITYSCAPES_LABELS_ROOT/{split}/{city}/{city}_{seq}_{frame}_gtFine_labelIds.png
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from data.transforms import normalize_imagenet

# Standard Cityscapes raw label id -> 19-class trainId mapping. 255 = ignore.
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255,
}

# Vectorised form of the dict above -- the per-key boolean-mask loop it
# replaces ran 34 full-image comparisons per sample.
_LUT = np.full(256, 255, dtype=np.uint8)
for _raw_id, _train_id in ID_TO_TRAINID.items():
    if _raw_id >= 0:
        _LUT[_raw_id] = _train_id


class CityscapesDataset(Dataset):
    """Returns (image, seg_label).

    `normalize=False` yields the image in raw [0,1] space. CutMix needs that,
    because it composites in pixel space and normalises once at the end --
    pasting into already-normalised tensors would apply the shift twice.
    """

    def __init__(self, split="train", size=(config.INPUT_HEIGHT, config.INPUT_WIDTH),
                 normalize=True):
        self.size = size
        self.normalize = normalize
        self.images = sorted(glob.glob(
            os.path.join(config.CITYSCAPES_IMAGES_ROOT, split, "*", "*_leftImg8bit.png")
        ))
        self.labels = [
            p.replace(config.CITYSCAPES_IMAGES_ROOT, config.CITYSCAPES_LABELS_ROOT)
             .replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            for p in self.images
        ]
        assert len(self.images) > 0, (
            f"no Cityscapes images found for split={split} under "
            f"{config.CITYSCAPES_IMAGES_ROOT}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(
            self.size[::-1], Image.BILINEAR)
        raw_label = np.array(Image.open(self.labels[idx]).resize(
            self.size[::-1], Image.NEAREST))

        train_label = _LUT[raw_label]

        image_t = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        image_t = image_t.permute(2, 0, 1).float() / 255.0
        if self.normalize:
            image_t = normalize_imagenet(image_t)

        label_t = torch.from_numpy(train_label.astype(np.int64))
        return image_t, label_t

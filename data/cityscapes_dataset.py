"""Cityscapes leftImg8bit / gtFine loader (Section 02/03).

Layout:
    CITYSCAPES_IMAGES_ROOT/{split}/{city}/{city}_{seq}_{frame}_leftImg8bit.png
    CITYSCAPES_LABELS_ROOT/{split}/{city}/{city}_{seq}_{frame}_gtFine_labelIds.png
"""

import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import config

# Standard Cityscapes raw label id -> 19-class trainId mapping. 255 = ignore.
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255,
}


class CityscapesDataset(Dataset):
    def __init__(self, split="train", size=(config.INPUT_HEIGHT, config.INPUT_WIDTH)):
        self.size = size
        self.images = sorted(glob.glob(
            os.path.join(config.CITYSCAPES_IMAGES_ROOT, split, "*", "*_leftImg8bit.png")
        ))
        self.labels = [
            p.replace(config.CITYSCAPES_IMAGES_ROOT, config.CITYSCAPES_LABELS_ROOT)
             .replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            for p in self.images
        ]
        assert len(self.images) > 0, f"no Cityscapes images found for split={split}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(self.size[::-1], Image.BILINEAR)
        raw_label = np.array(Image.open(self.labels[idx]).resize(self.size[::-1], Image.NEAREST))

        train_label = np.full_like(raw_label, 255, dtype=np.uint8)
        for raw_id, train_id in ID_TO_TRAINID.items():
            train_label[raw_label == raw_id] = train_id

        image_t = TF.to_tensor(image)
        label_t = torch.from_numpy(train_label).long()
        return image_t, label_t

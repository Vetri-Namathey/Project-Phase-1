"""CutMix augmentation (Section 05): pastes a random CARLA-generated OOD
object onto a Cityscapes image with probability CUTMIX_PROB, producing a
binary per-pixel OOD ground-truth mask alongside the existing segmentation
label. This is the only source of OOD training signal -- Fishyscapes is
never used here or anywhere in training.
"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class CutMixAugmentedDataset(Dataset):
    def __init__(self, base_dataset, p=config.CUTMIX_PROB):
        self.base = base_dataset
        self.p = p
        self.carla_images = sorted(glob.glob(os.path.join(config.CARLA_IMAGES_DIR, "*.png")))
        self.carla_masks = sorted(glob.glob(os.path.join(config.CARLA_MASKS_DIR, "*.npy")))
        assert len(self.carla_images) == len(self.carla_masks) and len(self.carla_images) > 0, \
            "CARLA images/masks must exist and be paired 1:1"

    def __len__(self):
        return len(self.base)

    def _load_random_carla_object(self):
        idx = random.randrange(len(self.carla_images))
        image = np.array(Image.open(self.carla_images[idx]).convert("RGB"))
        mask = np.load(self.carla_masks[idx])
        ys, xs = np.where(mask == 1)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def __getitem__(self, idx):
        image_t, seg_label = self.base[idx]
        _, h, w = image_t.shape
        ood_target = torch.zeros((h, w), dtype=torch.float32)

        if random.random() < self.p:
            obj = self._load_random_carla_object()
            if obj is not None:
                obj_img, obj_mask = obj
                oh, ow = obj_mask.shape
                scale = random.uniform(config.CUTMIX_SCALE_MIN, config.CUTMIX_SCALE_MAX) * min(h, w) / max(oh, ow)
                new_h, new_w = max(1, int(oh * scale)), max(1, int(ow * scale))

                obj_img_r = np.array(Image.fromarray(obj_img).resize((new_w, new_h), Image.BILINEAR))
                obj_mask_r = np.array(
                    Image.fromarray(obj_mask * 255).resize((new_w, new_h), Image.NEAREST)
                ) > 127

                max_y, max_x = h - new_h, w - new_w
                if max_y > 0 and max_x > 0:
                    py, px = random.randint(0, max_y), random.randint(0, max_x)
                    image_np = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    region = image_np[py:py + new_h, px:px + new_w]
                    region[obj_mask_r] = obj_img_r[obj_mask_r]
                    image_np[py:py + new_h, px:px + new_w] = region
                    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

                    ood_target[py:py + new_h, px:px + new_w][torch.from_numpy(obj_mask_r)] = 1.0

        return image_t, seg_label, ood_target

"""CutMix outlier exposure.

Pastes anomaly objects onto Cityscapes images and produces the binary
per-pixel OOD ground-truth mask alongside the segmentation label. This is
the only source of OOD training signal -- Fishyscapes is never used here or
anywhere else in training.

Objects come from data/anomaly_sources.py, which serves either the CARLA
bank (canonical) or the COCO bank (temporary stand-in) behind one
interface. This file does not know or care which.

Three properties of the paste are deliberate, and each one fixes a measured
mismatch against the real Fishyscapes test distribution:

  scale      Sampled log-uniformly in [CUTMIX_SCALE_MIN, CUTMIX_SCALE_MAX].
             Real anomalies have a median longest side of 0.043 of the image
             short side; the previous 0.30-0.55 range did not overlap the
             real distribution at all.

  placement  Restricted to drivable surface (road/sidewalk trainIds). Under
             uniform placement only 33% of objects landed on road -- the
             rest sat in sky, inside buildings or on top of cars, which is a
             cue that cannot transfer to real anomalies.

  appearance Brightness/contrast harmonised toward the destination region
             and the mask edge feathered. A hard-edged crop carrying foreign
             colour statistics is a shortcut the heads will learn instead of
             learning what an unfamiliar object looks like.
"""

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import Dataset

import config
from data.anomaly_sources import build_anomaly_bank
from data.transforms import normalize_imagenet


class CutMixAugmentedDataset(Dataset):
    """Wraps an UNNORMALISED CityscapesDataset and returns
    (image, seg_label, ood_target) with the image normalised at the end."""

    def __init__(self, base_dataset, p=config.CUTMIX_PROB, anomaly_bank=None):
        self.base = base_dataset
        self.p = p
        self.bank = anomaly_bank if anomaly_bank is not None else build_anomaly_bank()
        if getattr(base_dataset, "normalize", False):
            raise ValueError(
                "CutMixAugmentedDataset needs an unnormalised base dataset -- "
                "it composites in pixel space and normalises once at the end. "
                "Construct CityscapesDataset(..., normalize=False)."
            )

    def __len__(self):
        return len(self.base)

    def _sample_scale(self):
        """Log-uniform, so the small sizes that dominate real anomalies also
        dominate training rather than being a rare tail."""
        lo, hi = np.log(config.CUTMIX_SCALE_MIN), np.log(config.CUTMIX_SCALE_MAX)
        return float(np.exp(random.uniform(lo, hi)))

    def _sample_surface_position(self, seg_label, new_h, new_w):
        """Pick a top-left corner such that the object's centre sits on a
        drivable surface. Falls back to uniform placement if the frame has no
        valid surface (rare, but some Cityscapes frames are nearly all
        building/vegetation)."""
        h, w = seg_label.shape
        max_y, max_x = h - new_h, w - new_w
        if max_y <= 0 or max_x <= 0:
            return None

        valid = np.isin(seg_label, config.CUTMIX_VALID_SURFACE_TRAINIDS)
        # Centre must land inside the frame with the object fully on-image.
        half_h, half_w = new_h // 2, new_w // 2
        valid[:half_h, :] = False
        valid[h - (new_h - half_h):, :] = False
        valid[:, :half_w] = False
        valid[:, w - (new_w - half_w):] = False

        ys, xs = np.nonzero(valid)
        if len(ys) == 0:
            return random.randint(0, max_y), random.randint(0, max_x)

        pick = random.randrange(len(ys))
        return int(ys[pick]) - half_h, int(xs[pick]) - half_w

    @staticmethod
    def _harmonize(obj_rgb, obj_mask, dest_region):
        """Match the object's per-channel mean/std to the destination region.

        Keeps the object's own texture and shape -- only its exposure and
        colour cast move -- so it still reads as an unfamiliar object rather
        than becoming invisible.
        """
        obj = obj_rgb.astype(np.float32)
        sel = obj_mask.astype(bool)
        if sel.sum() < 16:
            return obj_rgb

        dest = dest_region.reshape(-1, 3).astype(np.float32)
        src = obj[sel]

        src_mean, src_std = src.mean(0), src.std(0) + 1e-5
        dest_mean, dest_std = dest.mean(0), dest.std(0) + 1e-5

        # Partial correction: full histogram matching washes the object out.
        strength = 0.6
        gain = 1.0 + strength * (dest_std / src_std - 1.0)
        bias = strength * (dest_mean - src_mean)

        out = (obj - src_mean) * gain + src_mean + bias
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _drop_fragments(obj_mask):
        """Remove mask fragments too small to be a learnable object.

        Source masks can be multi-part (COCO annotations frequently are), so
        scaling an object down leaves specks of one or two pixels behind.
        Labelling those "anomaly" is pure noise: nothing can detect a
        one-pixel object, and each speck still counts as a positive the loss
        has to explain. Returns None if nothing survives, in which case the
        paste is skipped entirely rather than producing an empty target.
        """
        labelled, n = ndimage.label(obj_mask)
        if n == 0:
            return None
        sizes = ndimage.sum(obj_mask, labelled, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes)
                if s >= config.CUTMIX_MIN_OBJECT_PIXELS]
        if not keep:
            return None
        return np.isin(labelled, keep).astype(np.uint8)

    @staticmethod
    def _feather(obj_mask):
        """Soft alpha at the mask edge, so the composite has no razor-sharp
        boundary for the heads to key on."""
        radius = config.CUTMIX_EDGE_FEATHER_PX
        if radius <= 0:
            return obj_mask.astype(np.float32)
        blurred = Image.fromarray((obj_mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0

    def _prepare_object(self, h, w):
        """Sample and scale one object, retrying a few times.

        A draw can fail for benign reasons -- an empty mask, a scale that
        rounds the object below the fragment threshold. Without a retry those
        turn into images with no anomaly at all, which wastes the positive
        signal the batch was supposed to carry.
        """
        for _ in range(4):
            obj = self.bank.sample()
            if obj is None:
                continue
            obj_rgb, obj_mask = obj
            oh, ow = obj_mask.shape

            scale = self._sample_scale() * min(h, w) / max(oh, ow)
            new_h = max(1, int(round(oh * scale)))
            new_w = max(1, int(round(ow * scale)))
            if new_h < 2 or new_w < 2 or new_h >= h or new_w >= w:
                continue

            obj_rgb_r = np.asarray(
                Image.fromarray(obj_rgb).resize((new_w, new_h), Image.BILINEAR))
            obj_mask_r = (np.asarray(
                Image.fromarray((obj_mask * 255).astype(np.uint8))
                .resize((new_w, new_h), Image.NEAREST)) > 127).astype(np.uint8)
            obj_mask_r = self._drop_fragments(obj_mask_r)
            if obj_mask_r is None:
                continue
            return obj_rgb_r, obj_mask_r, new_h, new_w
        return None

    def _paste_one(self, image_np, seg_label, ood_target):
        h, w = seg_label.shape
        prepared = self._prepare_object(h, w)
        if prepared is None:
            return
        obj_rgb_r, obj_mask_r, new_h, new_w = prepared

        pos = self._sample_surface_position(seg_label, new_h, new_w)
        if pos is None:
            return
        py, px = pos
        py = int(np.clip(py, 0, h - new_h))
        px = int(np.clip(px, 0, w - new_w))

        dest_region = image_np[py:py + new_h, px:px + new_w]
        if config.CUTMIX_HARMONIZE:
            obj_rgb_r = self._harmonize(obj_rgb_r, obj_mask_r, dest_region)

        alpha = self._feather(obj_mask_r)[..., None]
        blended = dest_region.astype(np.float32) * (1.0 - alpha) + \
            obj_rgb_r.astype(np.float32) * alpha
        image_np[py:py + new_h, px:px + new_w] = np.clip(blended, 0, 255).astype(np.uint8)

        # The OOD target stays a hard binary mask even though the composite
        # edge is soft -- the feather is an anti-shortcut measure, not a
        # change to what counts as anomalous.
        hard = obj_mask_r.astype(bool)
        ood_target[py:py + new_h, px:px + new_w][hard] = 1.0

        # These pixels no longer show the Cityscapes class the label claims,
        # so mark them ignore instead of training the seg head on a target
        # known to be wrong.
        seg_label[py:py + new_h, px:px + new_w][hard] = config.CUTMIX_SEG_IGNORE_INDEX

    def __getitem__(self, idx):
        image_t, seg_label = self.base[idx]
        _, h, w = image_t.shape

        image_np = (image_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        seg_np = seg_label.numpy().copy()
        ood_np = np.zeros((h, w), dtype=np.float32)

        if random.random() < self.p:
            n_objects = random.randint(config.CUTMIX_MIN_OBJECTS,
                                       config.CUTMIX_MAX_OBJECTS)
            for _ in range(n_objects):
                self._paste_one(image_np, seg_np, ood_np)

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_t = normalize_imagenet(image_t)

        return image_t, torch.from_numpy(seg_np).long(), torch.from_numpy(ood_np)

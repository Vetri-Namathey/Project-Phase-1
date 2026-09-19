"""SUPERSEDED AND BROKEN -- use `python check_data.py` instead.

Kept only as a record of the pre-merge check. It does not run against the
current pipeline: `CityscapesDataset(split="train")` now defaults to
`normalize=True` and `CutMixAugmentedDataset` rejects a normalised base
dataset outright, and the `img * 255` line below predates ImageNet
normalisation (it would render wrong colours even if constructed correctly).
`check_data.py` does the same visual check properly, plus object-size
statistics against the real Fishyscapes distribution.
"""

import os
import sys

from PIL import Image

# This file lives in legacy/, the data package lives at the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cityscapes_dataset import CityscapesDataset  # noqa: E402
from data.cutmix import CutMixAugmentedDataset  # noqa: E402

base = CityscapesDataset(split="train")
aug = CutMixAugmentedDataset(base, p=1.0)
img, label, ood = aug[0]

arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8").copy()
arr[ood.numpy() == 1] = [255, 0, 0]
Image.fromarray(arr).save("check_cutmix_overlay.png")
print("saved check_cutmix_overlay.png")

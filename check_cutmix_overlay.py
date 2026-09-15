from PIL import Image

from data.cityscapes_dataset import CityscapesDataset
from data.cutmix import CutMixAugmentedDataset

base = CityscapesDataset(split="train")
aug = CutMixAugmentedDataset(base, p=1.0)
img, label, ood = aug[0]

arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8").copy()
arr[ood.numpy() == 1] = [255, 0, 0]
Image.fromarray(arr).save("check_cutmix_overlay.png")
print("saved check_cutmix_overlay.png")

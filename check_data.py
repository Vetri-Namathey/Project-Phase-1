"""Visual checks on the anomaly training data.

    python check_data.py                 # CutMix composites + size statistics
    python check_data.py --object 25     # one raw object from the active bank

Both modes read whichever bank config.ANOMALY_SOURCE selects, so CARLA
frames and COCO cutouts are inspected the same way.

The CutMix mode is the one that matters: it shows the three properties that
were each measurably wrong before -- object size relative to the frame,
placement on drivable surface, and how well the paste blends in.
"""

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

import config
from data.anomaly_sources import build_anomaly_bank
from data.cityscapes_dataset import CityscapesDataset
from data.cutmix import CutMixAugmentedDataset
from data.transforms import denormalize_imagenet

# Measured from all 188 annotated objects in Fishyscapes Lost & Found, as a
# fraction of the image short side. Printed alongside ours so the comparison
# needs no lookup.
REAL_P25, REAL_MEDIAN, REAL_P75 = 0.022, 0.043, 0.072
REAL_PIXEL_RATE = 0.0028


def show_object(index, out):
    bank = build_anomaly_bank()
    if not 0 <= index < len(bank):
        raise SystemExit(f"--object must be in [0, {len(bank) - 1}]")

    image_path, mask_path = bank.pairs[index]
    image = np.array(Image.open(image_path).convert("RGB"))
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

    print(f"source: {config.ANOMALY_SOURCE}")
    print(f"image:  {image_path}  {image.shape}")
    print(f"mask:   {mask.shape} | unique {np.unique(mask)} | "
          f"object px {int(mask.sum())} ({mask.mean():.2%} of crop)")

    overlay = image.copy()
    overlay[mask == 1] = (0.5 * overlay[mask == 1] +
                          0.5 * np.array([255, 0, 0])).astype(np.uint8)
    out = out or f"check_data_object_{config.ANOMALY_SOURCE}_{index:03d}.png"
    Image.fromarray(overlay).save(out)
    print(f"saved {out}")


def show_cutmix(n, out):
    base = CityscapesDataset(split="train", normalize=False)
    aug = CutMixAugmentedDataset(base, p=1.0)

    tiles, sizes, rates = [], [], []
    for i in range(n):
        image_t, _, ood = aug[i * 7]
        rgb = (denormalize_imagenet(image_t).permute(1, 2, 0).numpy() * 255)
        rgb = rgb.astype(np.uint8).copy()

        mask = ood.numpy() > 0
        # Outline rather than fill, so the blend quality stays visible.
        edge = mask ^ np.pad(mask, 1, mode="constant")[:-2, 1:-1]
        edge |= mask ^ np.pad(mask, 1, mode="constant")[1:-1, :-2]
        rgb[edge] = [255, 0, 0]
        tiles.append(rgb)
        rates.append(float(mask.mean()))

        # Per connected component. The bounding box of every anomaly pixel at
        # once spans the gaps between separate objects and reports several
        # times the true size -- an earlier version of this script did that
        # and printed a reassuring-looking number that meant nothing.
        labelled, k = ndimage.label(mask)
        for j in range(1, k + 1):
            ys, xs = np.where(labelled == j)
            sizes.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
                         / config.INPUT_HEIGHT)

    grid = np.concatenate([np.concatenate(tiles[i:i + 2], axis=1)
                           for i in range(0, len(tiles) - 1, 2)], axis=0)
    out = out or "check_data_cutmix.png"
    Image.fromarray(grid).save(out)

    sizes = np.array(sizes)
    print(f"saved {out}  (source: {config.ANOMALY_SOURCE})")
    print(f"  {len(sizes)} objects across {n} images ({len(sizes) / n:.2f}/image)")
    print(f"  anomaly pixel rate:  ours {np.mean(rates):.4%}   "
          f"real {REAL_PIXEL_RATE:.3%}")
    print(f"  per-object longest side / image short side:")
    print(f"    ours  p25={np.percentile(sizes, 25):.3f} "
          f"median={np.median(sizes):.3f} p75={np.percentile(sizes, 75):.3f}")
    print(f"    real  p25={REAL_P25:.3f} median={REAL_MEDIAN:.3f} "
          f"p75={REAL_P75:.3f}")
    print(f"  scale range: {config.CUTMIX_SCALE_MIN}-{config.CUTMIX_SCALE_MAX} "
          f"(log-uniform)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=int, default=None,
                        help="inspect one raw object from the bank instead")
    parser.add_argument("--n", type=int, default=6, help="CutMix samples to render")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.object is not None:
        show_object(args.object, args.out)
    else:
        show_cutmix(args.n, args.out)


if __name__ == "__main__":
    main()

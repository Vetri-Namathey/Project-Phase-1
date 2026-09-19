"""Build a bank of anomaly object cutouts from COCO val2017.

TEMPORARY STAND-IN for the CARLA object bank, used only while no CARLA
server is available. CARLA remains the canonical source (config.ANOMALY_
SOURCE defaults to "carla"); this script exists so training is not blocked
on simulator access.

What it produces: data/coco_objects/{NNNN}_{category}_rgb.png plus a
matching _mask.png -- the same (RGB crop, binary mask) pair that
generate_anomalies.py produces from CARLA, so data/cutmix.py consumes both
through one interface and nothing downstream changes.

Why COCO objects are valid outlier exposure: every category that overlaps
Cityscapes' 19 known classes is excluded (config.COCO_EXCLUDED_CATEGORIES),
so what remains -- a suitcase, a teddy bear, an elephant, a microwave -- is
genuinely not part of the model's known label set. This is the standard
outlier exposure used by PEBAL (ECCV 2022), DenseHybrid (ECCV 2022) and
Mask2Anomaly (ICCV 2023).

No pycocotools dependency: COCO stores instance outlines as polygon vertex
lists, which PIL.ImageDraw rasterises directly.

Licensing: COCO images are Flickr photos under Creative Commons; the
annotations are CC BY 4.0. Academic use is fine and requires citation
(Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014).

Usage:
    python download_coco_anomalies.py                 # ~1GB download, once
    python download_coco_anomalies.py --max-objects 4000
    python download_coco_anomalies.py --keep-archives # keep the zips
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

import config

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# An object has to be big enough to carry real texture once cropped, and not
# so close to the frame edge that it is a sliced-off fragment rather than a
# whole object.
MIN_MASK_PIXELS = 900          # ~30x30
MIN_BBOX_SIDE = 24
MIN_FILL_RATIO = 0.25          # mask area / bbox area -- rejects thin slivers
MAX_BORDER_TOUCH_FRAC = 0.30   # fraction of bbox perimeter allowed on the edge


def _download(url, dest):
    if os.path.exists(dest):
        print(f"  already present: {os.path.basename(dest)}")
        return
    print(f"  downloading {url}")

    def hook(count, block_size, total):
        if total <= 0:
            return
        pct = min(100.0, count * block_size * 100.0 / total)
        sys.stdout.write(f"\r    {pct:5.1f}%  ({total / 1e6:.0f} MB total)")
        sys.stdout.flush()

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    sys.stdout.write("\n")


def _unzip(archive, dest_dir, marker):
    if os.path.exists(marker):
        print(f"  already extracted: {os.path.basename(marker)}")
        return
    print(f"  extracting {os.path.basename(archive)}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def _polygons_to_mask(segmentation, height, width):
    """COCO polygon format -> binary mask. Returns None for RLE (crowd)
    annotations, which are multi-object regions rather than one object."""
    if not isinstance(segmentation, list):
        return None
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in segmentation:
        if len(poly) < 6:  # fewer than 3 vertices
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(pts, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def _touches_border(mask, y0, y1, x0, x1):
    h, w = mask.shape
    on_edge = 0
    total = 0
    if y0 == 0:
        on_edge += mask[0, x0:x1].sum()
    if y1 >= h:
        on_edge += mask[h - 1, x0:x1].sum()
    if x0 == 0:
        on_edge += mask[y0:y1, 0].sum()
    if x1 >= w:
        on_edge += mask[y0:y1, w - 1].sum()
    total = 2 * (y1 - y0) + 2 * (x1 - x0)
    return total > 0 and (on_edge / total) > MAX_BORDER_TOUCH_FRAC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-objects", type=int, default=3000)
    parser.add_argument("--download-dir", default="data/_coco_download")
    parser.add_argument("--out-dir", default=config.COCO_OBJECTS_DIR)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    images_zip = os.path.join(args.download_dir, "val2017.zip")
    annotations_zip = os.path.join(args.download_dir, "annotations_trainval2017.zip")
    images_dir = os.path.join(args.download_dir, "val2017")
    annotations_json = os.path.join(
        args.download_dir, "annotations", "instances_val2017.json"
    )

    print("[1/4] fetching COCO val2017")
    _download(IMAGES_URL, images_zip)
    _unzip(images_zip, args.download_dir, images_dir)
    _download(ANNOTATIONS_URL, annotations_zip)
    _unzip(annotations_zip, args.download_dir, annotations_json)

    print("[2/4] loading annotations")
    with open(annotations_json, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    excluded = {name.lower() for name in config.COCO_EXCLUDED_CATEGORIES}

    # Fail loudly rather than silently dropping a typo'd exclusion -- a
    # single leaked "car" would teach the model that cars are anomalies.
    known = {name.lower() for name in categories.values()}
    unknown = excluded - known
    if unknown:
        raise ValueError(
            f"COCO_EXCLUDED_CATEGORIES names categories that do not exist in "
            f"COCO: {sorted(unknown)}. Fix config.py -- a typo here silently "
            f"lets a Cityscapes-known class into the anomaly bank."
        )

    kept_ids = {cid for cid, name in categories.items() if name.lower() not in excluded}
    print(f"  {len(categories)} COCO categories, {len(excluded)} excluded, "
          f"{len(kept_ids)} usable")
    print(f"  excluded: {sorted(excluded)}")

    images_by_id = {img["id"]: img for img in coco["images"]}

    print("[3/4] extracting object cutouts")
    rng = np.random.default_rng(args.seed)
    annotations = [
        a for a in coco["annotations"]
        if a["category_id"] in kept_ids and not a.get("iscrowd", 0)
        and a.get("area", 0) >= MIN_MASK_PIXELS
    ]
    rng.shuffle(annotations)

    for stale in os.listdir(args.out_dir):
        if stale.endswith(("_rgb.png", "_mask.png")):
            os.remove(os.path.join(args.out_dir, stale))

    saved = 0
    per_category = Counter()
    rejected = Counter()

    for ann in annotations:
        if saved >= args.max_objects:
            break

        meta = images_by_id[ann["image_id"]]
        mask = _polygons_to_mask(ann["segmentation"], meta["height"], meta["width"])
        if mask is None:
            rejected["rle/crowd"] += 1
            continue

        ys, xs = np.where(mask == 1)
        if len(ys) < MIN_MASK_PIXELS:
            rejected["too few pixels"] += 1
            continue

        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        if min(bh, bw) < MIN_BBOX_SIDE:
            rejected["bbox too small"] += 1
            continue
        if len(ys) / float(bh * bw) < MIN_FILL_RATIO:
            rejected["sliver"] += 1
            continue
        if _touches_border(mask, y0, y1, x0, x1):
            rejected["cut off at frame edge"] += 1
            continue

        image_path = os.path.join(images_dir, meta["file_name"])
        if not os.path.exists(image_path):
            rejected["image missing"] += 1
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        crop = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

        name = categories[ann["category_id"]].replace(" ", "_")
        stem = os.path.join(args.out_dir, f"{saved:05d}_{name}")
        Image.fromarray(crop).save(stem + "_rgb.png")
        Image.fromarray(crop_mask * 255).save(stem + "_mask.png")

        per_category[name] += 1
        saved += 1
        if saved % 250 == 0:
            print(f"  {saved} cutouts")

    print(f"\n[4/4] done -- {saved} cutouts in {args.out_dir}")
    print(f"  distinct categories: {len(per_category)}")
    print("  most common:", ", ".join(
        f"{n}={c}" for n, c in per_category.most_common(10)))
    if rejected:
        print("  rejected:", ", ".join(f"{k}={v}" for k, v in rejected.most_common()))

    leaked = sorted(set(per_category) & {e.replace(" ", "_") for e in excluded})
    if leaked:
        raise AssertionError(f"EXCLUDED CATEGORY LEAKED INTO BANK: {leaked}")
    print("  verified: no Cityscapes-overlapping category present")

    if not args.keep_archives:
        for path in (images_zip, annotations_zip):
            if os.path.exists(path):
                os.remove(path)
        shutil.rmtree(images_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(args.download_dir, "annotations"), ignore_errors=True)
        print("  cleaned up downloads (pass --keep-archives to retain them)")


if __name__ == "__main__":
    main()

"""Draw what the model actually flags, on real Fishyscapes photos.

This is the visual counterpart to the numbers in check_runs.py. AUROC and AP
say the ranking is good; they do not show you WHERE the model is looking. A
detector can post a respectable AUROC while boxing sky, building facades, or
half the frame -- which is exactly what the pre-audit pipeline did, and what
no scalar metric in this repo would have caught.

For each frame it reports, and draws:

  score inside the real object   mean fused score over ground-truth anomaly px
  score on background            mean fused score over everything else
  separation                     the gap between them
  boxes drawn                    connected components in the top-scoring region
  boxes ON the object            how many of those boxes actually overlap
                                 ground truth -- the number that matters, and
                                 the one a heatmap screenshot cannot tell you

Ground truth is drawn in green, the model's own boxes in red. If red sits on
green, the detector is looking at the right thing.

    python check_detections.py
    python check_detections.py --n 6 --percentile 97 --out proof.png
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from utils import get_device, load_trained_model

MIN_BOX_PIXELS = 40          # ignore single-pixel specks when boxing


@torch.no_grad()
def score_frame(model, image_path, device, out_shape):
    """Fused OOD score at the label's native resolution."""
    tensor = load_image_tensor(image_path, device)
    out = model(tensor)
    fused = F.interpolate(out["ood_fused"].unsqueeze(1), size=out_shape,
                          mode="bilinear", align_corners=False)
    return fused.squeeze().float().cpu().numpy()


def boxes_from_score(score, threshold):
    """Box every region scoring above an ABSOLUTE threshold.

    Not a percentile. A percentile always selects that fraction of the frame
    whether or not anything is there: at the 97th percentile of a 2-megapixel
    image you box 63,000 pixels, while a real anomaly occupies roughly 1,500.
    The result is dozens of background boxes and a detector that looks broken
    when it is not -- the first version of this script did exactly that and
    reported "2% of boxes land on a real anomaly" for a model whose in-object
    score was 0.69 against a 0.001 background.

    An absolute threshold is also the honest one, because it is what a
    deployed system would actually use: a fixed trigger level, not "flag the
    top 3% of whatever I am looking at".
    """
    mask = score >= threshold
    labelled, n = ndimage.label(mask)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labelled == i)
        if len(ys) < MIN_BOX_PIXELS:
            continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return boxes, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="absolute fused-score threshold for boxing")
    parser.add_argument("--out", default="check_detections.png")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_3HEAD,
                        help="checkpoint to inspect (default: config.CHECKPOINT_3HEAD)")
    args = parser.parse_args()

    device = get_device()
    model = load_trained_model(args.checkpoint, device=device, num_heads=len(config.OOD_HEAD_SEEDS_3HEAD))

    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())

    # Frames that actually contain an anomaly -- boxing a frame with nothing
    # in it proves nothing either way.
    usable = []
    for image_path, label_path in test_pairs:
        label = np.array(Image.open(label_path))
        if (label == 1).sum() >= 200:
            usable.append((image_path, label_path, label))
        if len(usable) >= args.n:
            break

    print(f"\n{'frame':<44} {'in-obj':>7} {'bkgd':>7} {'sep':>7} "
          f"{'boxes':>6} {'on-obj':>7}")
    print("-" * 82)

    tiles, totals = [], []
    for image_path, label_path, label in usable:
        score = score_frame(model, image_path, device, label.shape)

        gt = label == 1
        valid = label != 255
        bkgd = valid & ~gt

        in_obj = float(score[gt].mean())
        on_bkgd = float(score[bkgd].mean())
        boxes, _ = boxes_from_score(score, args.threshold)
        flagged_frac = float((score >= args.threshold).mean())

        # A box counts as "on the object" if it overlaps ground truth at all.
        on_object = sum(1 for x0, y0, x1, y1 in boxes
                        if gt[y0:y1 + 1, x0:x1 + 1].any())
        totals.append((on_object, len(boxes)))

        name = os.path.basename(image_path)
        print(f"{name[:42]:<42} {in_obj:>7.3f} {on_bkgd:>7.3f} "
              f"{in_obj - on_bkgd:>7.3f} {flagged_frac:>8.3%} {len(boxes):>6} "
              f"{on_object:>3}/{len(boxes):<3}")

        rgb = np.array(Image.open(image_path).convert("RGB"))
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)

        # Ground truth first, in green, so red boxes draw over it.
        gt_lab, gt_n = ndimage.label(gt)
        for i in range(1, gt_n + 1):
            ys, xs = np.where(gt_lab == i)
            if len(ys) < 50:
                continue
            draw.rectangle([xs.min() - 4, ys.min() - 4, xs.max() + 4, ys.max() + 4],
                           outline=(40, 220, 90), width=6)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(230, 60, 40), width=6)

        w, h = canvas.size
        tiles.append(canvas.resize((int(w * args.scale), int(h * args.scale))))

    hit = sum(a for a, _ in totals)
    drawn = sum(b for _, b in totals)
    print("-" * 90)
    print(f"{'TOTAL':<42} {'':>7} {'':>7} {'':>7} {'':>9} {drawn:>6} {hit:>3}/{drawn:<3}"
          f"  ({hit / max(drawn, 1):.0%} of boxes land on a real anomaly)")

    grid = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
    y = 0
    for t in tiles:
        grid.paste(t, (0, y))
        y += t.height
    grid.save(args.out)
    print(f"\nsaved {args.out}   green = ground truth, red = model's own boxes")


if __name__ == "__main__":
    main()

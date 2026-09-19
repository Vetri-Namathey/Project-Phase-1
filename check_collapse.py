"""Collapse diagnostic on a saved checkpoint.

Collapse means the heads have stopped telling anomalous pixels apart from
normal ones. The number that measures this is the SEPARATION -- mean score
on anomalous pixels minus mean score on normal ones -- not the global output
std.

Global std is misleading here and used to be what this script reported. The
Fishyscapes positive rate is 0.24%, so in a healthy detector almost every
pixel is a correctly-near-zero negative and the global std is low by
construction. An earlier version of this check flagged a model scoring
0.99 AUROC as collapsed.

  separation > 0.05   healthy -- the two classes score differently
  separation < 0.01   collapsed, IF AUROC is also near chance
  mean near 0.5       indecision (the head has no idea either way)
  mean near 0.0       collapsed to "always normal"

train.py logs all of this every epoch; this script is for inspecting a
checkpoint after the fact.

    python check_collapse.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs
from data.transforms import load_image_tensor
from metrics import ScoreHistogram
from utils import get_device, load_trained_model


def main():
    device = get_device()
    num_heads = len(config.OOD_HEAD_SEEDS_3HEAD)
    model = load_trained_model(device=device, num_heads=num_heads)

    pos_scores = [[] for _ in range(num_heads)]
    neg_scores = [[] for _ in range(num_heads)]
    hists = [ScoreHistogram() for _ in range(num_heads)]

    with torch.no_grad():
        for image_path, label_path in list_fishyscapes_pairs()[:12]:
            label_map = np.array(Image.open(label_path))
            valid = label_map != 255
            labels = label_map[valid]

            out = model(load_image_tensor(image_path, device))
            per_head = F.interpolate(out["ood_scores"], size=label_map.shape,
                                     mode="bilinear", align_corners=False)
            per_head = per_head.squeeze(0).cpu().numpy()[:, valid]

            for h in range(num_heads):
                hists[h].update(per_head[h], labels)
                pos_scores[h].append(per_head[h][labels == 1])
                neg_scores[h].append(per_head[h][labels != 1])

    print(f"\ncheckpoint: {config.CHECKPOINT_3HEAD}")
    print(f"encoder:    {config.ENCODER_NAME}\n")

    for h in range(num_heads):
        pos = np.concatenate(pos_scores[h])
        neg = np.concatenate(neg_scores[h])
        sep = float(pos.mean() - neg.mean())
        auroc = hists[h].auroc()

        if sep >= 0.05:
            verdict = "healthy"
        elif sep >= 0.01 or auroc >= 0.70:
            verdict = "weak but discriminating"
        elif 0.4 < neg.mean() < 0.6:
            verdict = "COLLAPSED (indecision, stuck near 0.5)"
        else:
            verdict = "COLLAPSED (always-normal)"

        print(f"head {h}: anomalous mean={pos.mean():.4f}  normal mean={neg.mean():.4f}  "
              f"separation={sep:+.4f}  auroc={auroc:.4f}")
        print(f"         global std={np.concatenate([pos, neg]).std():.4f} "
              f"(low is normal at a {len(pos) / (len(pos) + len(neg)):.2%} positive "
              f"rate -- do not read it as collapse)  -> {verdict}")


if __name__ == "__main__":
    main()

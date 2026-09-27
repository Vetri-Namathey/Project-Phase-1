"""Novelty 5: the 1-head vs 3-head vs calibrated table, measured the same way.

For each checkpoint:
  * Fishyscapes TEST half: AUROC / AP / FPR@95 / ECE of the fused score, and
    the head-disagreement AUROC (train.evaluate_ood -- the exact code that
    selected the checkpoints).
  * MC-Dropout within-head AUROC on the same clean test images (fp32, N
    passes): what a single head can offer as an uncertainty signal WITHOUT a
    second head to disagree with.
  * Cityscapes val mIoU on ALL 500 images (training only checks 100 for
    speed), plus per-class IoU, so a drop can be located -- the plan asks for
    per-class, not just the aggregate.

    python check_ablation.py
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.cityscapes_dataset import CityscapesDataset
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from metrics import ConfusionMatrix, ScoreHistogram
from train import amp_context, evaluate_ood
from utils import get_device, load_trained_model

CITYSCAPES_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole",
                      "traffic light", "traffic sign", "vegetation", "terrain", "sky",
                      "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                      "bicycle"]

DEFAULT_MODELS = [
    ("1-head", "runs/phase2a_1head_run1/model_1head_best.pth", 1),
    ("3-head", "runs/phase2a_coco_run1/model_3head_best.pth", 3),
    ("3-head + L_calib", "runs/phase2b_calib2/model_3head_calib_best.pth", 3),
]


@torch.no_grad()
def full_miou(model, dataset, device):
    model.eval()
    cm = ConfusionMatrix(num_classes=config.NUM_SEG_CLASSES, ignore_index=255)
    for i in range(len(dataset)):
        image, label = dataset[i]
        with amp_context(device):
            out = model(image.unsqueeze(0).to(device))
        cm.update(out["seg_logits"].float().argmax(1).squeeze(0).cpu().numpy(), label.numpy())
    return cm


@torch.no_grad()
def within_head_auroc(model, pairs, device, n_passes):
    """Variance of each head across MC-Dropout passes, averaged over heads,
    scored as an anomaly detector. fp32: the variances are small enough that
    bf16 rounding would distort them."""
    hist = ScoreHistogram()
    for image_path, label_path in pairs:
        label = np.array(Image.open(label_path))
        valid = label != 255
        out = model.predict_dual_mode(load_image_tensor(image_path, device),
                                      n_passes=n_passes, force_safety=True)
        var = F.interpolate(out["epistemic_uncertainty"].unsqueeze(1), size=label.shape,
                            mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        hist.update(np.clip(var[valid] / 0.25, 0, 1), label[valid])
    return hist.auroc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=config.MC_DROPOUT_PASSES)
    args = parser.parse_args()

    device = get_device()
    _, test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    val_ds = CityscapesDataset(split="val", normalize=True)

    rows, per_class = [], {}
    for name, path, heads in DEFAULT_MODELS:
        print(f"--- {name}: {path}", flush=True)
        model = load_trained_model(path, device, num_heads=heads)
        m = evaluate_ood(model, test, device, heads)
        within = within_head_auroc(model, test, device, args.passes)
        cm = full_miou(model, val_ds, device)
        rows.append((name, m, within, cm.miou()))
        per_class[name] = cm.per_class_iou()
        del model
        torch.cuda.empty_cache()

    print(f"\nFishyscapes test half (50) + Cityscapes val mIoU ({len(val_ds)} images)")
    print(f"{'model':<18} {'AUROC':>7} {'AP':>7} {'FPR@95':>7} {'ECE':>7} "
          f"{'disagree AUROC':>15} {'MC within AUROC':>16} {'mIoU':>7}")
    for name, m, within, miou in rows:
        dis = m["auroc_disagreement"] if not name.startswith("1-head") else float("nan")
        print(f"{name:<18} {m['auroc']:>7.4f} {m['ap']:>7.4f} {m['fpr95']:>7.4f} "
              f"{m['ece']:>7.4f} {dis:>15.4f} {within:>16.4f} {miou:>7.4f}")

    names = [r[0] for r in rows]
    print(f"\nper-class IoU (Cityscapes val, {len(val_ds)} images)")
    print(f"{'class':<14}" + "".join(f"{n:>18}" for n in names))
    for c, cls in enumerate(CITYSCAPES_CLASSES):
        vals = [per_class[n][c] for n in names]
        spread = np.nanmax(vals) - np.nanmin(vals)
        print(f"{cls:<14}" + "".join(f"{v:>18.4f}" for v in vals) +
              ("   <- spread > 0.02" if spread > 0.02 else ""))


if __name__ == "__main__":
    main()

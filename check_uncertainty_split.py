"""Novelty 7: does TwinGuard separate "the sensor is struggling" from
"this object is genuinely unfamiliar"?

The safety-triggered mode produces two uncertainty signals (model/
twinguard_model.py, predict_dual_mode):

  within-head   variance of ONE head across N MC-Dropout passes, averaged
                over heads ("epistemic_uncertainty" in the code). Expected to
                rise when the INPUT is degraded -- the head is unstable on it.
  between-head  spread of the 3 heads' mean predictions ("parametric_
                uncertainty"). Expected to rise on a NOVEL OBJECT -- three
                independently initialised decision boundaries disagree there.

Both are epistemic (neither models sensor noise explicitly), so the plan's
vocabulary is between-head / within-head, not epistemic / aleatoric.

Test: the Fishyscapes TEST half, each image run clean and under each
degradation (noise, motion blur, fog) at the middle of the training severity
range, labels untouched. Two questions, both answered from the same passes:

  1. Novelty: on CLEAN images, how well does each signal separate anomaly
     pixels from normal ones (AUROC)?
  2. Degradation: on NORMAL pixels only (no anomaly present), how much does
     each signal rise when the input is degraded (degraded / clean)?

The split is supported if between-head wins (1) and within-head wins (2).
Either failing is a reportable result, not a script failure.

    python check_uncertainty_split.py --checkpoint runs/phase2a_coco_run1/model_3head_best.pth
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

import config
from data.degradations import DEGRADATIONS
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import normalize_imagenet
from metrics import ScoreHistogram
from utils import get_device, load_trained_model

CONDITIONS = ("clean", "gaussian_noise", "motion_blur", "fog")
# Signals are rescaled to [0,1] for the histogram: a std of values in [0,1]
# is at most 0.5, a variance at most 0.25.
SCALE = {"fused": 1.0, "between": 0.5, "within": 0.25}


def load_pair(image_path, label_path):
    size = (config.INPUT_WIDTH, config.INPUT_HEIGHT)
    rgb = np.asarray(Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR),
                     dtype=np.float32) / 255.0
    label = np.asarray(Image.open(label_path).resize(size, Image.NEAREST))
    return rgb, label


def degrade(rgb, name, rng):
    if name == "clean":
        return rgb
    fn, (lo, hi) = DEGRADATIONS[name]
    return fn(rgb, (lo + hi) / 2, rng).astype(np.float32)


@torch.no_grad()
def signals(model, rgb, device, n_passes):
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model.predict_dual_mode(normalize_imagenet(x), n_passes=n_passes, force_safety=True)
    return {"fused": out["ood_fused"].squeeze(0).cpu().numpy(),
            "between": out["parametric_uncertainty"].squeeze(0).cpu().numpy(),
            "within": out["epistemic_uncertainty"].squeeze(0).cpu().numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_3HEAD)
    parser.add_argument("--passes", type=int, default=config.MC_DROPOUT_PASSES)
    parser.add_argument("--n-images", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    model = load_trained_model(args.checkpoint, device)
    _, test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    test = test[:args.n_images]
    rng = np.random.default_rng(config.GLOBAL_SEED)

    hists = {c: {s: ScoreHistogram() for s in SCALE} for c in CONDITIONS}
    sums = {c: {s: {"normal": 0.0, "anomaly": 0.0} for s in SCALE} for c in CONDITIONS}
    counts = {"normal": 0, "anomaly": 0}

    t0 = time.time()
    for i, (image_path, label_path) in enumerate(test):
        rgb, label = load_pair(image_path, label_path)
        valid = label != 255
        normal, anomaly = valid & (label == 0), valid & (label == 1)
        counts["normal"] += int(normal.sum())
        counts["anomaly"] += int(anomaly.sum())
        for c in CONDITIONS:
            sig = signals(model, degrade(rgb, c, rng), device, args.passes)
            for s, m in sig.items():
                hists[c][s].update(np.clip(m[valid] / SCALE[s], 0, 1), label[valid].astype(np.int64))
                sums[c][s]["normal"] += float(m[normal].sum())
                sums[c][s]["anomaly"] += float(m[anomaly].sum())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test)} images ({time.time() - t0:.0f}s)", flush=True)

    mean = {c: {s: {k: sums[c][s][k] / counts[k] for k in counts} for s in SCALE}
            for c in CONDITIONS}

    print(f"\n{len(test)} Fishyscapes test images, {args.passes} MC-Dropout passes/head, "
          f"degradations at mid training severity, fp32")
    print(f"\n{'condition':<15} {'fused AUROC':>11} {'fused AP':>9} | "
          f"{'between-head':^30} | {'within-head':^30}")
    print(f"{'':<15} {'':>11} {'':>9} | {'normal':>9} {'anomaly':>9} {'AUROC':>9} | "
          f"{'normal':>9} {'anomaly':>9} {'AUROC':>9}")
    for c in CONDITIONS:
        f = hists[c]["fused"]
        row = f"{c:<15} {f.auroc():>11.4f} {f.average_precision():>9.4f} |"
        for s in ("between", "within"):
            row += (f" {mean[c][s]['normal']:>9.5f} {mean[c][s]['anomaly']:>9.5f} "
                    f"{hists[c][s].auroc():>9.4f} |")
        print(row)

    print("\n1. NOVELTY -- clean images, anomaly vs normal pixels:")
    b_auc, w_auc = hists["clean"]["between"].auroc(), hists["clean"]["within"].auroc()
    b_lift = mean["clean"]["between"]["anomaly"] / mean["clean"]["between"]["normal"]
    w_lift = mean["clean"]["within"]["anomaly"] / mean["clean"]["within"]["normal"]
    print(f"   between-head AUROC {b_auc:.4f} (anomaly/normal {b_lift:.1f}x)   "
          f"within-head AUROC {w_auc:.4f} (anomaly/normal {w_lift:.1f}x)")

    print("\n2. DEGRADATION -- normal pixels only, degraded / clean:")
    wins = 0
    for c in CONDITIONS[1:]:
        b = mean[c]["between"]["normal"] / mean["clean"]["between"]["normal"]
        w = mean[c]["within"]["normal"] / mean["clean"]["within"]["normal"]
        wins += w > b
        print(f"   {c:<15} between-head x{b:.2f}   within-head x{w:.2f}   "
              f"({'within-head rises more' if w > b else 'between-head rises more'})")

    print("\nVERDICT")
    novelty_ok = b_auc > w_auc
    degradation_ok = wins == len(CONDITIONS) - 1
    print(f"   between-head is the better novelty signal:          {'YES' if novelty_ok else 'NO'}")
    print(f"   within-head responds more to every degradation:     "
          f"{'YES' if degradation_ok else f'NO ({wins}/{len(CONDITIONS) - 1})'}")
    print("   Split supported." if novelty_ok and degradation_ok else
          "   Split NOT fully supported -- report as measured.")


if __name__ == "__main__":
    main()

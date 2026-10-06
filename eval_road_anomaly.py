"""Second real-photo eval set: RoadAnomaly21 (SegmentMeIfYouCan anomaly track).
PLAN.md P5. Ported from exp_v2 (agent_state/exp_v2_snapshot/eval_road_anomaly.py)
with a required --checkpoint (rule 11), a required --input-scale, and the zip md5 check.

Only the 10 VALIDATION images have public labels; the 100 test labels are
withheld and scored only by the official online benchmark. So this is a
cross-dataset sanity check on 10 images, not a benchmark number -- report it
as such. Nothing is fitted or selected on it.

It is also a very different test from Fishyscapes Lost & Found:
  * web photos from many cameras, 1280x720 (resized to the model's 1024x512
    input exactly like every other image, so the aspect ratio is stretched)
  * anomalies are large and close (animals, aircraft, boats): 2-36% of each
    image vs 0.28% in Lost & Found

Scores TwinGuard (fused 3-head score) and, with --with-baseline, Experiment
A's MSP on the same images with the same metric code.

    python eval_road_anomaly.py --checkpoint model_3head_best.pth --input-scale 1 --zip <path>\dataset_AnomalyTrack.zip --with-baseline
"""

import argparse
import glob
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.transforms import load_image_tensor
from metrics import ScoreHistogram
from train import amp_context
from utils import get_device, load_trained_model


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validation_pairs():
    labels = sorted(glob.glob(os.path.join(config.ROAD_ANOMALY21_DIR, "labels_masks",
                                           "validation*_labels_semantic.png")))
    pairs = []
    for label in labels:
        stem = os.path.basename(label).replace("_labels_semantic.png", "")
        image = os.path.join(config.ROAD_ANOMALY21_DIR, "images", stem + ".jpg")
        if os.path.exists(image):
            pairs.append((image, label))
    if len(pairs) != 10:
        raise SystemExit(f"expected 10 RoadAnomaly21 validation pairs under "
                         f"{config.ROAD_ANOMALY21_DIR}, found {len(pairs)}")
    return pairs


@torch.no_grad()
def twinguard_scores(model, image_path, shape, device):
    with amp_context(device):
        out = model(load_image_tensor(image_path, device))
    logits = F.interpolate(out["ood_logits"].float(), size=shape, mode="bilinear",
                           align_corners=False)
    return torch.sigmoid(logits).mean(dim=1).squeeze(0).cpu().numpy()


def summary_line(name, hist, pos_rate):
    m = hist.summary()
    return (f"{name:<22} AP={m['ap']:.4f}  AUROC={m['auroc']:.4f}  "
            f"FPR@95={m['fpr95']:.4f}  ECE={m['ece']:.4f}   (anomaly pixels {pos_rate:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. model_3head_best.pth (rule 11)")
    parser.add_argument("--input-scale", choices=("1",), required=True,
                        help="P3 kept scale 1 (1024x512); other scales are not implemented here")
    parser.add_argument("--zip", default=None,
                        help="dataset_AnomalyTrack.zip, to verify its md5 before scoring")
    parser.add_argument("--with-baseline", action="store_true",
                        help="also score Experiment A (MSP) on the same images")
    args = parser.parse_args()

    if args.zip:
        got = md5_of(args.zip)
        ok = got == config.ROAD_ANOMALY21_ZIP_MD5
        print(f"zip md5 {got}  expected {config.ROAD_ANOMALY21_ZIP_MD5}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("RoadAnomaly21 zip md5 mismatch -- re-download before scoring")
    else:
        print("zip md5 not checked (no --zip given): record as UNVERIFIED")

    device = get_device()
    pairs = validation_pairs()
    labels = [np.array(Image.open(lab)) for _, lab in pairs]
    print(f"checkpoint {args.checkpoint}, input scale {args.input_scale}, data {config.ROAD_ANOMALY21_DIR}")

    results = []
    model = load_trained_model(args.checkpoint, device)
    hist, per_image = ScoreHistogram(), []
    for (img, _), lab in zip(pairs, labels):
        s = twinguard_scores(model, img, lab.shape, device)
        valid = lab != 255
        hist.update(s[valid], lab[valid])
        h1 = ScoreHistogram()
        h1.update(s[valid], lab[valid])
        per_image.append((os.path.basename(img), (lab[valid] == 1).mean(),
                          h1.auroc(), h1.average_precision()))
    results.append(("TwinGuard (3-head)", hist))
    del model
    torch.cuda.empty_cache()

    if args.with_baseline:
        from experiment_a import load_baseline, predict
        processor, baseline = load_baseline(device)
        hist_a = ScoreHistogram()
        for (img, _), lab in zip(pairs, labels):
            _, ood = predict(img, processor, baseline, device)
            valid = lab != 255
            hist_a.update(ood[valid], lab[valid])
        results.append(("Experiment A (MSP)", hist_a))

    total = sum((lab != 255).sum() for lab in labels)
    pos = sum((lab == 1).sum() for lab in labels)
    print(f"\nRoadAnomaly21 validation split: {len(pairs)} images, {total:,} valid pixels (pooled)")
    for name, h in results:
        print(summary_line(name, h, pos / total))
    print("\nTwinGuard per image:")
    for name, rate, auc, ap in per_image:
        print(f"  {name:<20} anomaly {rate:6.2%}   AP {ap:.4f}   AUROC {auc:.4f}")
    print("\n10 images only -- a cross-dataset sanity check, not a benchmark number. "
          "The official 100-image test labels are withheld (online submission only).")


if __name__ == "__main__":
    main()

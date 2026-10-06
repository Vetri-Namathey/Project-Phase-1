"""P3: full-resolution decision, Fishyscapes VAL half only. Eval-only.

The encoder was fine-tuned on 1024x1024 crops of full-resolution
Cityscapes, but the heads were trained -- and every number so far measured
-- at 1024x512 input. Feeding 2048x1024 might recover small/far objects, or
might hurt, because the heads only ever saw half-resolution feature
statistics. This script measures it on the val half and applies the rule
PLAN.md P3 fixed before any number existed:

  ADOPT m    dAP = AP(m) - AP(1) >= +0.03, paired-bootstrap CI excluding 0,
             and dFPR@95 <= +0.01. Two qualifying modes: the higher AP;
             within 0.01 of each other: scale 2 (cheaper than ms).
  HURTS      dAP <= -0.03 with the CI excluding 0. Keep scale 1.
  NO GAIN    anything else. Keep scale 1.

Band-ECE r=8 per mode is printed for the Finding 2 wording only; nothing is
decided on it. Modes: 1 (1024x512), 2 (2048x1024), ms (mean of the per-head
logits of both passes at label resolution). Never touches the test half.

Usage:
    python eval_scale.py --raw model_3head_best.pth --modes 1 2 ms [--window]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from eval_spatial import forward_logits
from metrics import ScoreHistogram, boundary_band, paired_bootstrap_ap
from utils import get_device, load_trained_model

N_BOOTSTRAP = 1000
ADOPT_DAP = 0.03
MAX_DFPR = 0.01
TIE_AP = 0.01


def decide(rows):
    """rows: {mode: {"dap", "lo", "hi", "dfpr", "ap"}} for modes other than 1.
    Returns (per-mode verdicts, chosen mode)."""
    verdicts = {}
    for mode, r in rows.items():
        if r["dap"] >= ADOPT_DAP and r["lo"] > 0 and r["dfpr"] <= MAX_DFPR:
            verdicts[mode] = "ADOPT"
        elif r["dap"] <= -ADOPT_DAP and r["hi"] < 0:
            verdicts[mode] = "HURTS"
        else:
            verdicts[mode] = "NO GAIN"
    adopted = [m for m in rows if verdicts[m] == "ADOPT"]
    if not adopted:
        return verdicts, "1"
    best = max(adopted, key=lambda m: rows[m]["ap"])
    if "2" in adopted and rows[best]["ap"] - rows["2"]["ap"] < TIE_AP:
        best = "2"
    return verdicts, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True,
                        help=f"checkpoint, e.g. {config.PRIMARY_RAW} (explicit, MISTAKES.md M15)")
    parser.add_argument("--modes", nargs="+", choices=["1", "2", "ms"], default=["1", "2", "ms"])
    parser.add_argument("--window", action="store_true",
                        help="run scale 2 as two overlapping windows (only if P0b chose windows)")
    args = parser.parse_args()
    if "1" not in args.modes or len(args.modes) < 2:
        parser.error("--modes must include 1 (the baseline) and at least one other mode")

    device = get_device()
    model = load_trained_model(args.raw, device)
    fishy_val, fishy_test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    pairs = fishy_val
    assert pairs is fishy_val  # val only: the decision must never see the test half
    print(f"P3 scale decision on {len(pairs)} Fishyscapes VAL images, modes {args.modes}"
          f"{' (scale 2 windowed)' if args.window else ''}, raw={args.raw}")

    r = config.TEMPERATURE_BAND_RADIUS_PX
    whole = {m: ScoreHistogram() for m in args.modes}
    band = {m: ScoreHistogram() for m in args.modes}
    per_image = {m: [] for m in args.modes}
    need = {1} | ({2} if {"2", "ms"} & set(args.modes) else set())
    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid, anomaly = label_map != 255, label_map == 1
        img_band = boundary_band(anomaly, radius_px=r, valid=valid)
        up = {s: F.interpolate(forward_logits(model, image_path, device, s, args.window and s == 2)
                               .unsqueeze(0), size=label_map.shape, mode="bilinear",
                               align_corners=False).squeeze(0)
              for s in need}
        logits = {"1": up[1]}
        if 2 in up:
            logits["2"] = up[2]
            logits["ms"] = (up[1] + up[2]) / 2
        labels = anomaly.astype(np.int64)
        for m in args.modes:
            scores = torch.sigmoid(logits[m]).mean(dim=0).numpy()
            whole[m].update(scores[valid], labels[valid])
            band[m].update(scores[img_band], labels[img_band])
            h = ScoreHistogram()
            h.update(scores[valid], labels[valid])
            per_image[m].append((h.pos, h.neg))

    print(f"\n{'mode':<5} {'AUROC':>7} {'AP':>7} {'FPR@95':>7} {'ECE':>8} {f'band-ECE r={r}':>14}")
    summary = {}
    for m in args.modes:
        s = whole[m].summary()
        s["band_ece"] = band[m].ece()
        summary[m] = s
        print(f"{m:<5} {s['auroc']:>7.4f} {s['ap']:>7.4f} {s['fpr95']:>7.4f} "
              f"{s['ece']:>8.5f} {s['band_ece']:>14.4f}")

    rows = {}
    for m in args.modes:
        if m == "1":
            continue
        res = paired_bootstrap_ap(per_image[m], per_image["1"], n_resamples=N_BOOTSTRAP,
                                  seed=config.GLOBAL_SEED)
        rows[m] = {"dap": res["diff"], "lo": res["lo"], "hi": res["hi"], "ap": summary[m]["ap"],
                   "dfpr": summary[m]["fpr95"] - summary["1"]["fpr95"]}
        print(f"\nmode {m} vs 1 (val, paired bootstrap {N_BOOTSTRAP}): dAP={res['diff']:+.4f} "
              f"95% CI [{res['lo']:+.4f}, {res['hi']:+.4f}]  dFPR@95={rows[m]['dfpr']:+.4f}  "
              f"d band-ECE r={r}={summary[m]['band_ece'] - summary['1']['band_ece']:+.4f} "
              f"(descriptive)")

    verdicts, chosen = decide(rows)
    print("\npre-registered rule (PLAN.md P3): " +
          ", ".join(f"mode {m}: {v}" for m, v in verdicts.items()))
    print(f"chosen input scale: {chosen}" +
          ("  -- HEADLINE-CHANGING: stop and tell the user before any further run"
           if chosen != "1" else "  (scale 1 kept)"))


if __name__ == "__main__":
    main()

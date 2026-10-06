"""X5 same-photo paste test (PLAN.md P4). Eval-only.

Pastes anomaly-bank objects onto the REAL Fishyscapes test photos with the
training paste appearance (data/cutmix.py: harmonisation, feathering, fragment
drop), then compares pasted vs real objects in the SAME photos. The original
photo is the exact removal of every paste, so no inpainting is involved.

Design (P4 procedure, fixes M18):
  - ONE paste per forward pass: per photo, 1 forward on the original + 1 per
    arm. Pastes never share an image, so they cannot affect each other.
  - Same centre and same longest-side target for all arms of a photo, so the
    arms differ only in which object is pasted.
  - The centre sits on the model's own road/sidewalk prediction, far enough
    from the real object that the paste cannot touch it. A placement whose
    paste meets the real mask dilated by KEEP_APART_PX is rejected; each arm
    retries up to MAX_TRIES objects, else that arm is dropped for that photo.
  - Real-object drift (its score after a paste minus before) gets a paired CI
    per arm. It must include 0, else the pastes affect the real object and the
    primary contrast is confounded.

Arms:
  coco_seen    one of the 1500 COCO cutouts in the training bank (seeded subsample)
  coco_unseen  one of the other 1500, never used in training
  carla_bank   one of the CARLA objects the A1 audit labels "aligned" (by default)
  real         the photo's own Fishyscapes object (original photo)

Pre-registered contrasts (PLAN.md P4): PRIMARY edge gap unseen paste - real;
SECONDARY object score seen - unseen (memorisation; exploratory unless U0-4
confirms the local COCO bank is the training one). Everything else exploratory.

    python paste_test.py --check-banks                       (CPU, no model)
    python paste_test.py --raw model_3head_best.pth --input-scale 1 | Tee-Object -FilePath paste_test.log
"""

import os
import sys

if "--check-banks" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # rule 13 ("" is unset on Windows): the CPU check must not reach CUDA

import argparse
import csv
import json
import random

import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy import ndimage

import config
from data.anomaly_sources import _ObjectBank, _build_both_bank, _build_carla_bank, _build_coco_bank
from data.cutmix import CutMixAugmentedDataset as CutMix
from metrics import ScoreHistogram, boundary_band

ARMS = ("coco_seen", "coco_unseen", "carla_bank")
KEEP_APART_PX = 48
MAX_TRIES = 10
# Training samples the longest side log-uniformly in [0.012, 0.20] x short side
# (6-102 px at 1024x512). Below ~16 px most cutouts fall under the fragment
# threshold (CUTMIX_MIN_OBJECT_PIXELS), so the shared target is drawn from
# [16, 102] px: a recorded deviation from the training range.
TARGET_MIN_PX = 16
N_EXAMPLES = 6
A1_CSV = os.path.join("agent_state", "a1_carla_tiles", "a1_carla_tiles.csv")
# Outline colours in the example PNGs: seen=blue, unseen=yellow, CARLA=magenta, real=green.
OUTLINE = {"coco_seen": (61, 90, 254), "coco_unseen": (255, 210, 0),
           "carla_bank": (230, 0, 230), "real": (31, 164, 99)}


def coco_banks():
    """Same seeded subsample as data/anomaly_sources._build_both_bank."""
    pairs = list(_build_coco_bank().pairs)
    used = random.Random(config.GLOBAL_SEED).sample(pairs, config.COCO_BANK_TARGET)
    used_set = set(used)
    return _ObjectBank(used, "coco_seen"), _ObjectBank([p for p in pairs if p not in used_set], "coco_unseen")


def carla_bank(a1_csv, classes):
    """The CARLA objects whose A1 label is in `classes` (A1: masks of some
    objects are misaligned, M22)."""
    with open(a1_csv, newline="") as f:
        keep = {row["object"] for row in csv.DictReader(f) if row["a1_class"] in classes}
    pairs = [p for p in _build_carla_bank().pairs
             if os.path.splitext(os.path.basename(p[0]))[0] in keep]
    return _ObjectBank(pairs, "carla_bank")


def check_banks(a1_csv):
    """CPU-only bank-identity checks (P4 (i), (ii))."""
    import torch
    assert not torch.cuda.is_available(), "check must run on CPU (rule 13)"
    coco = list(_build_coco_bank().pairs)
    print(f"(i)  COCO cutout pairs on disk: {len(coco)}  (expected 3000)")
    assert len(coco) == 3000, len(coco)
    seen, unseen = coco_banks()
    both = _build_both_bank().pairs
    train_coco = both[config.CARLA_BANK_TARGET:]
    print(f"(ii) both-bank: {len(both)} entries = {config.CARLA_BANK_TARGET} CARLA + {len(train_coco)} COCO")
    assert train_coco == seen.pairs, "coco_banks() does not reproduce _build_both_bank's COCO subsample"
    assert not set(seen.pairs) & set(unseen.pairs) and len(unseen) == 3000 - config.COCO_BANK_TARGET
    print(f"     seen = training COCO subsample, same order: PASS; unseen {len(unseen)}, disjoint: PASS")
    for classes in (("aligned",), ("aligned", "unsure")):
        print(f"     CARLA objects with A1 class in {classes}: {len(carla_bank(a1_csv, classes))} of 45")
    print("Bank checks PASS. Caveat (U0-4): this proves the local bank matches the code, "
          "not that the local files are the ones the pod trained on.")


def paste_at(image, obj, cy, cx, target):
    """Paste `obj` (rgb crop, mask) centred at (cy, cx), longest side = target
    px, with the training harmonisation + feather. -> (image, mask) or None."""
    obj_rgb, obj_mask = obj
    oh, ow = obj_mask.shape
    s = target / max(oh, ow)
    nh, nw = max(1, int(round(oh * s))), max(1, int(round(ow * s)))
    H, W = image.shape[:2]
    y0, x0 = cy - nh // 2, cx - nw // 2
    if nh < 2 or nw < 2 or y0 < 0 or x0 < 0 or y0 + nh > H or x0 + nw > W:
        return None
    rgb = np.asarray(Image.fromarray(obj_rgb).resize((nw, nh), Image.BILINEAR))
    m = (np.asarray(Image.fromarray((obj_mask * 255).astype(np.uint8)).resize((nw, nh), Image.NEAREST)) > 127)
    m = CutMix._drop_fragments(m.astype(np.uint8))
    if m is None:
        return None
    dest = image[y0:y0 + nh, x0:x0 + nw]
    if config.CUTMIX_HARMONIZE:
        rgb = CutMix._harmonize(rgb, m, dest)
    alpha = CutMix._feather(m)[..., None]
    out = image.copy()
    out[y0:y0 + nh, x0:x0 + nw] = np.clip(dest * (1.0 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)
    full = np.zeros((H, W), bool)
    full[y0:y0 + nh, x0:x0 + nw] = m.astype(bool)
    return out, full


def pick_placement(rng, surface, real, valid):
    """Shared (centre, target) for one photo: target log-uniform in
    [TARGET_MIN_PX, CUTMIX_SCALE_MAX x short side]; centre on predicted
    road/sidewalk, with room for any object of that size away from the real one."""
    H, W = surface.shape
    lo, hi = np.log(TARGET_MIN_PX), np.log(config.CUTMIX_SCALE_MAX * min(H, W))
    dist = ndimage.distance_transform_edt(~real) if real.any() else np.full((H, W), np.inf)
    for _ in range(MAX_TRIES):
        target = int(round(np.exp(rng.uniform(lo, hi))))
        ok = surface & valid & (dist > KEEP_APART_PX + 0.75 * target)
        half = target // 2 + 1
        ok[:half], ok[H - half:], ok[:, :half], ok[:, W - half:] = False, False, False, False
        ys, xs = np.nonzero(ok)
        if len(ys):
            k = rng.randrange(len(ys))
            return int(ys[k]), int(xs[k]), target
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required="--check-banks" not in sys.argv,
                        help="checkpoint, e.g. model_3head_best.pth (required, rule 11)")
    parser.add_argument("--input-scale", choices=("1",), required="--check-banks" not in sys.argv,
                        help="P3 kept scale 1 (1024x512); other scales are not implemented here")
    parser.add_argument("--a1-csv", default=A1_CSV)
    parser.add_argument("--carla-classes", default="aligned",
                        help="comma list of A1 classes for the CARLA arm (A1 labels pending the user's eyeball)")
    parser.add_argument("--check-banks", action="store_true", help="CPU bank-identity checks only")
    parser.add_argument("--out", default="explain_out")
    args = parser.parse_args()
    if args.check_banks:
        check_banks(args.a1_csv)
        return

    from explain import MIN_OBJECT_PX, MIN_REGION_PX, RING, bootstrap_ci, load_pair, run, stage_attribution
    from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
    from utils import get_device, load_trained_model

    os.makedirs(args.out, exist_ok=True)
    seen, unseen = coco_banks()
    banks = {"coco_seen": seen, "coco_unseen": unseen,
             "carla_bank": carla_bank(args.a1_csv, tuple(args.carla_classes.split(",")))}
    device = get_device()
    model = load_trained_model(args.raw, device)
    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    print(f"X5 same-photo paste test: {len(test_pairs)} Fishyscapes test photos; one paste per forward; "
          f"arms {ARMS} (bank sizes {[len(b) for b in banks.values()]}; CARLA = A1 '{args.carla_classes}') "
          f"+ the real object; edge band +-{RING}px at 1024x512 (= r=8 at label res); "
          f"shared target {TARGET_MIN_PX}-{int(config.CUTMIX_SCALE_MAX * config.INPUT_HEIGHT)} px; "
          f"keep-apart {KEEP_APART_PX} px; {MAX_TRIES} tries per arm")

    def measure(fused, m, valid):
        band = boundary_band(m, RING, valid=valid, side="both")
        inner, outer = band & m, band & ~m
        return band, {"px": int(m.sum()), "score": float(fused[m].mean()),
                      "band_gap": float(fused[band].mean() - m[band].mean()),
                      "inner": float(fused[inner].mean()) if inner.any() else None,
                      "outer": float(fused[outer].mean()) if outer.any() else None}

    def attribution(image, m):
        contrib = stage_attribution(model, image, m, device)
        share = (np.abs(contrib) / np.abs(contrib).sum(1, keepdims=True)).mean(0)
        return {"share": share.tolist(), "sign": np.sign(contrib.mean(0)).tolist()}

    rows, dropped = [], {arm: 0 for arm in ARMS}
    no_placement = 0
    hists = {arm: ScoreHistogram(n_bins=1500) for arm in ARMS + ("real",)}
    for idx, (image_path, label_path) in enumerate(test_pairs):
        image, real, valid = load_pair(image_path, label_path)
        real = real & valid
        fused0, _, seg = run(model, image, device)
        row = {"image": os.path.basename(image_path)}
        has_real = real.sum() >= MIN_OBJECT_PX
        if has_real:
            band, row["real"] = measure(fused0, real, valid)
            row["real"].update(attribution(image, real))
            hists["real"].update(fused0[band], real[band].astype(np.uint8))

        surface = np.isin(seg, config.CUTMIX_VALID_SURFACE_TRAINIDS)
        place = pick_placement(random.Random(config.GLOBAL_SEED * 1000 + idx), surface, real, valid)
        if place is None:
            no_placement += 1
            rows.append(row)
            print(f"[{idx + 1}/{len(test_pairs)}] {row['image']}  no placement found -- pastes skipped")
            continue
        cy, cx, target = place
        row["placement"] = {"cy": cy, "cx": cx, "target_px": target}
        near_real = ndimage.distance_transform_edt(~real) <= KEEP_APART_PX if real.any() else np.zeros_like(real)
        examples = []
        for a, arm in enumerate(ARMS):
            rng = random.Random(config.GLOBAL_SEED * 1000 + idx * 10 + a + 1)
            result, tries = None, 0
            while result is None and tries < MAX_TRIES:
                tries += 1
                obj = banks[arm].sample(rng)
                if obj is None:
                    continue
                result = paste_at(image, obj, cy, cx, target)
                if result is not None and ((result[1] & near_real).any() or result[1].sum() < MIN_REGION_PX):
                    result = None
            if result is None:
                dropped[arm] += 1
                continue
            pasted, m = result
            fused1, _, _ = run(model, pasted, device)
            band, rec = measure(fused1, m, valid)
            rec.update(attribution(pasted, m))
            rec["tries"] = tries
            rec["surface_frac"] = float(surface[m].mean())
            rec["score_original_photo"] = float(fused0[m].mean())  # exact removal
            if has_real:
                rec["real_drift"] = float(fused1[real].mean() - fused0[real].mean())
            hists[arm].update(fused1[band], m[band].astype(np.uint8))
            row[arm] = rec
            examples.append((arm, pasted, m, fused1))
        rows.append(row)

        if idx < N_EXAMPLES and examples:
            tiles = []
            for arm, pasted, m, fused1 in examples:
                outlined = pasted.copy()
                for name, mm in ((arm, m), ("real", real)):
                    outlined[mm & ~ndimage.binary_erosion(mm, iterations=2)] = OUTLINE[name]
                heat = (cm.inferno(np.clip(fused1, 0, 1))[..., :3] * 255).astype(np.uint8)
                tiles.append(np.concatenate([outlined, (0.4 * pasted + 0.6 * heat).astype(np.uint8)], 0))
            Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(args.out, f"paste_test_{idx:02d}.png"))
        print(f"[{idx + 1}/{len(test_pairs)}] {row['image']}  target {target}px  " + "  ".join(
            f"{arm} {row[arm]['score']:.2f}" for arm in ARMS + ("real",) if arm in row))

    with open(os.path.join(args.out, "paste_test_per_image.json"), "w") as f:
        json.dump(rows, f, indent=1)

    rng = np.random.default_rng(config.GLOBAL_SEED)
    get = lambda arm, k: np.array([r[arm][k] for r in rows if arm in r and r[arm].get(k) is not None])
    print(f"\nPhotos without a placement: {no_placement}. Arms dropped after {MAX_TRIES} tries: {dropped}")
    print("Per arm (mean over photos). band_gap = mean score - true positive rate in the edge band "
          "(> 0 over-confident, < 0 under-confident). removed = score on the same pixels in the original photo.")
    print(f"  {'arm':12s} {'n':>3s} {'px':>6s} {'score':>6s} {'removed':>8s} {'band ECE':>9s} "
          f"{'band_gap':>9s} {'inner':>6s} {'outer':>6s} {'surf':>5s}   stage shares 1-4 (sign)")
    for arm in ARMS + ("real",):
        n = len(get(arm, "score"))
        if not n:
            continue
        removed = "  (same)" if arm == "real" else f"{get(arm, 'score_original_photo').mean():8.3f}"
        surf = "    -" if arm == "real" else f"{get(arm, 'surface_frac').mean():5.2f}"
        share = np.array([r[arm]["share"] for r in rows if arm in r]).mean(0)
        sign = np.array([r[arm]["sign"] for r in rows if arm in r]).mean(0)
        print(f"  {arm:12s} {n:3d} {get(arm, 'px').mean():6.0f} {get(arm, 'score').mean():6.3f} {removed} "
              f"{hists[arm].ece():9.4f} {get(arm, 'band_gap').mean():+9.4f} "
              f"{get(arm, 'inner').mean():6.3f} {get(arm, 'outer').mean():6.3f} {surf}   "
              + " ".join(f"{s:.2f}({g:+.1f})" for s, g in zip(share, sign)))

    print("\nPaired comparisons over photos that have both (95% CI, bootstrap over photos):")

    def paired(a, b, key, label, idx=None):
        both = [r for r in rows if a in r and b in r and r[a].get(key) is not None and r[b].get(key) is not None]
        if len(both) < 5:
            print(f"  {label}: only {len(both)} photos -- not tested")
            return
        val = (lambda r, arm: r[arm][key][idx]) if idx is not None else (lambda r, arm: r[arm][key])
        m, lo, hi = bootstrap_ci([val(r, a) - val(r, b) for r in both], rng)
        print(f"  {label:58s} n={len(both):2d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print(" PRIMARY (pre-registered):")
    paired("coco_unseen", "real", "band_gap", "edge gap, unseen paste - real")
    print(" SECONDARY (pre-registered; exploratory until U0-4 confirms the bank):")
    paired("coco_seen", "coco_unseen", "score", "object score, seen - unseen COCO (memorisation)")
    print(" CONFOUND CHECK (each CI must include 0):")
    for arm in ARMS:
        d = get(arm, "real_drift")
        if len(d) >= 5:
            m, lo, hi = bootstrap_ci(d, rng)
            print(f"  {'real-object drift after a ' + arm + ' paste':58s} n={len(d):2d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(" EXPLORATORY:")
    paired("coco_seen", "real", "band_gap", "edge gap, seen paste - real")
    paired("carla_bank", "real", "band_gap", "edge gap, CARLA paste - real")
    paired("coco_unseen", "real", "score", "object score, unseen paste - real")
    paired("carla_bank", "coco_seen", "score", "object score, CARLA bank - seen COCO")
    paired("coco_unseen", "real", "share", "stage-1 share, unseen paste - real", idx=0)
    paired("coco_unseen", "real", "share", "stage-4 share, unseen paste - real", idx=3)
    print("Note: real objects and pastes differ in size; px per arm is in the table above.")


if __name__ == "__main__":
    main()

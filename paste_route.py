"""World-anchored CutMix pastes over a record_route.py recording (PLAN.md 2d).

The recording already has native CARLA props. This adds the COCO + CARLA-bank
cutouts the plan calls for, but pinned to fixed points on the road in WORLD
coordinates, so each object grows and shifts with perspective as the camera
drives toward it -- instead of a sticker jumping around the screen per frame.

How: each object gets a ground anchor (a later camera position on the route,
dropped to road level, shifted sideways) and a physical height in metres.
Every frame projects the anchor through that frame's camera pose + intrinsics
from meta.jsonl; pixel height = fy * height_m / depth. The paste itself reuses
data/cutmix.py's harmonisation and edge feathering, so it looks like a
training paste. GT = native prop mask OR paste mask.

Which objects:
- COCO: drawn only from the cutouts NOT in the training subsample
  (COCO_BANK_TARGET of the 3000 were used), so these are unseen instances.
- CARLA bank: only 45 exist and all were in training. Flagged as seen in
  pastes.json; say so in the video caption.

Scale cap: a paste stops being drawn once it exceeds --max-frac of the short
side (default 0.25, just above training's CUTMIX_SCALE_MAX=0.20), for the
same reason record_route.py drops large props.

CPU only, no CARLA needed. Usage:
    python paste_route.py --root video_town02 --objects 10
Writes <root>_pasted/ with rgb/, mask/, meta.jsonl, pastes.json, frame_pastes.jsonl,
which eval_spatial.py --carla-root and render_video.py read directly.
"""

import argparse
import json
import math
import os
import random
import shutil

import numpy as np
from PIL import Image

import config
from data.anomaly_sources import _build_carla_bank, _build_coco_bank, _ObjectBank
from data.cutmix import CutMixAugmentedDataset

CAMERA_HEIGHT_M = 1.5  # record_route.py
MIN_DEPTH_M = 2.0
MAX_BBOX_FILL = 0.85
PROP_CENTRE_UP_M = 0.5  # project a native prop's centre, not its base, for its depth


def rotation_matrix(cam):
    """CARLA local->world rotation (same as carla.Transform.get_matrix)."""
    cy, sy = math.cos(math.radians(cam["yaw"])), math.sin(math.radians(cam["yaw"]))
    cp, sp = math.cos(math.radians(cam["pitch"])), math.sin(math.radians(cam["pitch"]))
    cr, sr = math.cos(math.radians(cam["roll"])), math.sin(math.radians(cam["roll"]))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr]])


def project(point, cam, K):
    """World point -> (u, v, depth). CARLA camera: x forward, y right, z up."""
    local = rotation_matrix(cam).T @ (point - np.array([cam["x"], cam["y"], cam["z"]]))
    depth = local[0]
    if depth <= 1e-6:
        return None
    return K["cx"] + K["fx"] * local[1] / depth, K["cy"] - K["fy"] * local[2] / depth, depth


def unseen_coco_bank():
    """Complement of _build_both_bank's seeded COCO subsample."""
    coco_pairs = list(_build_coco_bank().pairs)
    target = getattr(config, "COCO_BANK_TARGET", len(coco_pairs))
    used = set(random.Random(config.GLOBAL_SEED).sample(coco_pairs, target)) \
        if target < len(coco_pairs) else set(coco_pairs)
    unseen = [p for p in coco_pairs if p not in used]
    print(f"COCO: {len(coco_pairs)} cutouts, {len(used)} used in training, {len(unseen)} unseen")
    return _ObjectBank(unseen, "coco_unseen")


def paste(image, gt, obj_rgb, obj_mask, u, v, height_px, occluder=None):
    """Bottom-centre of the object at (u, v), clipped to the frame. Pixels in
    `occluder` (native props closer to the camera) are left untouched."""
    oh, ow = obj_mask.shape
    new_h = max(2, int(round(height_px)))
    new_w = max(2, int(round(ow * new_h / oh)))
    rgb_r = np.asarray(Image.fromarray(obj_rgb).resize((new_w, new_h), Image.BILINEAR))
    mask_r = (np.asarray(Image.fromarray((obj_mask * 255).astype(np.uint8))
                         .resize((new_w, new_h), Image.NEAREST)) > 127).astype(np.uint8)
    mask_r = CutMixAugmentedDataset._drop_fragments(mask_r)
    if mask_r is None:
        return None

    H, W = gt.shape
    y0, x0 = int(round(v)) - new_h, int(round(u)) - new_w // 2
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + new_h), min(W, x0 + new_w)
    if ye <= ys or xe <= xs:
        return None
    oy, ox = ys - y0, xs - x0
    rgb_c = rgb_r[oy:oy + ye - ys, ox:ox + xe - xs]
    mask_c = mask_r[oy:oy + ye - ys, ox:ox + xe - xs].copy()
    if occluder is not None:
        mask_c[occluder[ys:ye, xs:xe]] = 0
    if mask_c.sum() < config.CUTMIX_MIN_OBJECT_PIXELS:
        return None

    dest = image[ys:ye, xs:xe]
    if config.CUTMIX_HARMONIZE:
        rgb_c = CutMixAugmentedDataset._harmonize(rgb_c, mask_c, dest)
    alpha = CutMixAugmentedDataset._feather(mask_c)
    if occluder is not None:
        alpha[occluder[ys:ye, xs:xe]] = 0.0  # feather must not bleed onto the nearer prop
    alpha = alpha[..., None]
    image[ys:ye, xs:xe] = np.clip(dest * (1 - alpha) + rgb_c * alpha, 0, 255).astype(np.uint8)
    gt[ys:ye, xs:xe] |= mask_c.astype(bool)
    return [xs, ys, xe, ye]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="record_route.py output folder")
    parser.add_argument("--out", default=None, help="default: <root>_pasted")
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--max-frac", type=float, default=0.25,
                        help="stop drawing a paste above this fraction of the short side")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.root.rstrip("/\\") + "_pasted"
    rng = random.Random(args.seed)

    with open(os.path.join(args.root, "meta.jsonl")) as f:
        meta = [json.loads(line) for line in f]
    n = len(meta)
    banks = {"coco": unseen_coco_bank(), "carla": _build_carla_bank()}

    # Anchors spread over the route, skipping the first 10% (nothing ahead yet
    # to approach). Alternate source and side so the mix is visible.
    objects = []
    for k in range(args.objects):
        frame_idx = int(n * (0.1 + 0.9 * (k + 0.5) / args.objects))
        frame_idx = min(frame_idx, n - 1)
        cam = meta[frame_idx]["camera"]
        yaw = math.radians(cam["yaw"])
        lateral = (1 if k % 2 == 0 else -1) * rng.uniform(0.8, 1.6)
        ground = np.array([cam["x"] - math.sin(yaw) * lateral,
                           cam["y"] + math.cos(yaw) * lateral,
                           cam["z"] - CAMERA_HEIGHT_M])
        source = "coco" if k % 2 == 0 else "carla"
        bank = banks[source]
        for _ in range(10):
            pick = rng.randrange(len(bank.pairs))
            sample = _ObjectBank([bank.pairs[pick]], bank.name).sample()  # crop to bbox
            # Skip near-rectangular cutouts (COCO "dining table" style): a photo
            # patch reads as a pasted sticker in a video, not as an object.
            if sample is not None and sample[1].mean() < MAX_BBOX_FILL:
                break
        if sample is None:
            raise SystemExit(f"object {k}: 10 empty draws from the {source} bank -- check the bank")
        obj_rgb, obj_mask = sample
        objects.append({"id": k, "source": source, "file": os.path.basename(bank.pairs[pick][0]),
                        "seen_in_training": source == "carla",
                        "mechanism": "cutmix_paste", "anchor_frame": frame_idx,
                        "anchor_world": ground.tolist(), "lateral_m": lateral,
                        "height_m": rng.uniform(0.3, 0.9),
                        "_rgb": obj_rgb, "_mask": obj_mask})
        print(f"object {k}: {source} {objects[-1]['file']} at frame {frame_idx}, "
              f"lateral {lateral:+.2f} m, height {objects[-1]['height_m']:.2f} m")

    for sub in ("rgb", "mask"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    shutil.copy(os.path.join(args.root, "meta.jsonl"), os.path.join(out, "meta.jsonl"))
    shutil.copy(os.path.join(args.root, "props.json"), os.path.join(out, "props.json"))
    with open(os.path.join(args.root, "props.json")) as f:
        prop_centres = [np.array([p["transform"]["x"], p["transform"]["y"],
                                  p["transform"]["z"] + PROP_CENTRE_UP_M]) for p in json.load(f)]

    shown = 0
    with open(os.path.join(out, "frame_pastes.jsonl"), "w") as fp:
        for m in meta:
            name = f"{m['frame']:06d}"
            image = np.array(Image.open(os.path.join(args.root, "rgb", name + ".png")).convert("RGB"))
            gt = np.load(os.path.join(args.root, "mask", name + ".npy")).astype(bool)
            native = gt.copy()  # native prop pixels only, before any paste
            K, cam = m["intrinsics"], m["camera"]
            short = min(K["width"], K["height"])
            # The native mask isn't split per prop, so a paste is hidden behind
            # ALL native-prop pixels if any prop in front of the camera is nearer.
            prop_depths = [p[2] for p in (project(c, cam, K) for c in prop_centres) if p]

            visible = []
            for obj in objects:
                proj = project(np.array(obj["anchor_world"]), cam, K)
                if proj is None or proj[2] < MIN_DEPTH_M:
                    continue
                u, v, depth = proj
                height_px = K["fy"] * obj["height_m"] / depth
                if height_px > args.max_frac * short:
                    continue
                visible.append((depth, obj, u, v, height_px))

            boxes = []
            for depth, obj, u, v, height_px in sorted(visible, key=lambda t: -t[0]):  # far first
                behind_prop = any(d < depth for d in prop_depths)
                box = paste(image, gt, obj["_rgb"], obj["_mask"], u, v, height_px,
                            occluder=native if behind_prop else None)
                if box is not None:
                    boxes.append({"id": obj["id"], "source": obj["source"], "box": box,
                                  "depth_m": round(depth, 2)})
            shown += bool(boxes)

            Image.fromarray(image).save(os.path.join(out, "rgb", name + ".png"))
            np.save(os.path.join(out, "mask", name + ".npy"), gt)
            fp.write(json.dumps({"frame": m["frame"], "pastes": boxes}) + "\n")

    with open(os.path.join(out, "pastes.json"), "w") as f:
        json.dump([{k: v for k, v in o.items() if not k.startswith("_")} for o in objects], f, indent=2)
    print(f"done: {n} frames, {shown} with at least one paste -> {out}")


if __name__ == "__main__":
    main()

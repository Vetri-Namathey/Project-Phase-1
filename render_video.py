"""Split-screen demo video from a recorded (and pasted) CARLA route (PLAN.md 2d).

Left column  = baseline: head 0 of the 3-head model on its own. This is a
               single-head ABLATION of the same model, not Experiment A (a
               separately trained 1-head model whose weights are lost). Caption
               it that way.
Right column = TwinGuard: the 3-head ensemble (mean score, std disagreement).

Rows, both columns: camera (+ GT outline) | anomaly heatmap | head
disagreement | detection boxes. The baseline's disagreement panel is empty on
purpose: one head cannot disagree with itself, which is the point of the
comparison.

Scores are drawn on a fixed scale (no per-frame normalisation), so colour
means the same thing in every frame and nothing flickers. Boxes use one fixed
threshold for both columns.

Run with CARLA CLOSED (8 GB GPU):
    python render_video.py --root video_town02_pasted
Writes static/video/twinguard_<root>.mp4 (H.264, plays in the browser) and
prints per-object detection counts for both columns.
"""

import argparse
import json
import os

import av
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from data.transforms import load_image_tensor
from eval_spatial import carla_pairs
from train import amp_context
from utils import get_device, load_trained_model

PANEL_W, PANEL_H = 512, 256
HEADER_H = 40
ACCENT = (61, 90, 254)      # frontend --accent #3D5AFE (frames are RGB)
GT_GREEN = (31, 164, 99)    # frontend --good
HEAT_ALPHA = 0.6
# Max std of three values in [0, 1] is ~0.47, but the heads agree closely:
# at 0.3 the first full render (2026-10-04) showed an almost blank panel.
# 0.1 makes the edge disagreement visible; anything above it saturates.
DISAGREE_SCALE = 0.1
# 40 px in the first render boxed specks of a few pixels (6.7 false boxes per
# frame). 150 halved the false boxes (3.40/frame) but also dropped real far
# objects: found fell 77.2% -> 68.9% (MISTAKES.md M3 -- the earlier claim that
# 150 "keeps every real one" was an estimate from one example, refuted by the
# re-render). Default only; --min-box overrides it, chosen once by eye (P6b).
MIN_BOX_PX = 150


@torch.no_grad()
def head_scores(model, image_path, shape, device):
    """-> (3, H, W) per-head sigmoid scores at frame resolution."""
    with amp_context(device):
        out = model(load_image_tensor(image_path, device))
    up = F.interpolate(out["ood_logits"].float(), size=shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(up)[0].cpu().numpy()


def small(arr):
    return cv2.resize(arr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def label(panel, text):
    cv2.rectangle(panel, (0, 0), (8 + 8 * len(text), 20), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def overlay(rgb, values, cmap, weighted=False):
    """weighted=True: blend only as strongly as the value itself, so zero
    disagreement leaves the camera image untouched instead of tinting it."""
    values = np.clip(values, 0, 1)
    heat = (cmap(values)[..., :3] * 255).astype(np.float32)
    alpha = HEAT_ALPHA * (values[..., None] if weighted else 1.0)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def detect(score, threshold, min_box_px=MIN_BOX_PX):
    """Connected components above a fixed threshold -> boxes + the binary map."""
    binary = score > threshold
    labels, n = ndimage.label(binary)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or (labels[sl] == i).sum() < min_box_px:
            continue
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return boxes, binary


def box_panel(rgb, boxes):
    arr = rgb.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(arr, (x0, y0), (x1, y1), ACCENT, 3)
    return arr


def count_hits(gt, boxes):
    """GT objects (connected components) touched by a box, and boxes touching no GT."""
    labels, n = ndimage.label(gt)
    hit = set()
    false_boxes = 0
    for x0, y0, x1, y1 in boxes:
        inside = np.unique(labels[y0:y1, x0:x1])
        inside = inside[inside > 0]
        hit.update(inside.tolist())
        false_boxes += int(len(inside) == 0)
    return n, len(hit), false_boxes


def column(rgb, gt, score, disagreement, boxes):
    cam = rgb.copy()
    cam[outline(gt)] = GT_GREEN
    panels = [label(small(cam), "camera + ground truth (green)"),
              label(small(overlay(rgb, score, cm.inferno)), "anomaly score (0-1, fixed scale)")]
    if disagreement is None:
        blank = np.full((PANEL_H, PANEL_W, 3), 24, np.uint8)
        cv2.putText(blank, "n/a: a single head cannot", (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(blank, "disagree with itself", (140, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(label(blank, "head disagreement"))
    else:
        panels.append(label(small(overlay(rgb, disagreement / DISAGREE_SCALE, cm.viridis, weighted=True)),
                            f"head disagreement (std, full colour at {DISAGREE_SCALE})"))
    panels.append(label(small(box_panel(rgb, boxes)), f"detections: {len(boxes)}"))
    return np.concatenate(panels, axis=0)


def header(text_left, text_right):
    bar = np.full((HEADER_H, 2 * PANEL_W, 3), 10, np.uint8)
    cv2.putText(bar, text_left, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, text_right, (PANEL_W + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="paste_route.py (or record_route.py) output folder")
    parser.add_argument("--raw", required=True,
                        help="checkpoint, e.g. model_3head_best.pth (explicit, MISTAKES.md rule 11)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="one fixed box threshold for both columns")
    parser.add_argument("--min-box", type=int, default=MIN_BOX_PX,
                        help="smallest box drawn, in pixels of the connected region")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    name = os.path.basename(os.path.normpath(args.root))
    out_path = args.out or os.path.join("static", "video", f"twinguard_{name}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = get_device()
    model = load_trained_model(args.raw, device)
    pairs = carla_pairs(args.root)
    top = header("Baseline: single head (head 0 only)", "TwinGuard: 3-head ensemble")

    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = 2 * PANEL_W, HEADER_H + 4 * PANEL_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20"}

    totals = {"baseline": [0, 0, 0], "ours": [0, 0, 0]}  # gt objects, hit, false boxes
    for i, (image_path, mask_path) in enumerate(pairs):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        gt = np.load(mask_path).astype(bool)
        heads = head_scores(model, image_path, gt.shape, device)
        base_score, ours_score = heads[0], heads.mean(axis=0)
        base_boxes, _ = detect(base_score, args.threshold, args.min_box)
        ours_boxes, _ = detect(ours_score, args.threshold, args.min_box)
        for key, boxes in (("baseline", base_boxes), ("ours", ours_boxes)):
            n, hit, false = count_hits(gt, boxes)
            totals[key] = [totals[key][0] + n, totals[key][1] + hit, totals[key][2] + false]

        frame = np.concatenate([
            top,
            np.concatenate([column(rgb, gt, base_score, None, base_boxes),
                            column(rgb, gt, ours_score, heads.std(axis=0), ours_boxes)], axis=1),
        ], axis=0)
        cv2.line(frame, (PANEL_W, HEADER_H), (PANEL_W, frame.shape[0]), (255, 255, 255), 2)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
        if i % 50 == 0:
            print(f"frame {i}/{len(pairs)}")

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"\nwrote {len(pairs)} frames -> {out_path}  (threshold {args.threshold}, min box {args.min_box} px)")
    for key, (n, hit, false) in totals.items():
        print(f"  {key:8s}  GT objects found {hit}/{n} ({hit / max(n, 1):.1%})  "
              f"boxes on no object: {false}  ({false / len(pairs):.2f} per frame)")
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w") as f:
        json.dump({"frames": len(pairs), "threshold": args.threshold, "min_box_px": args.min_box,
                   "root": args.root,
                   "baseline": "head 0 of the 3-head model (ablation, not Experiment A)",
                   "totals": {k: {"gt_objects": v[0], "found": v[1], "false_boxes": v[2]}
                              for k, v in totals.items()}}, f, indent=2)


if __name__ == "__main__":
    main()

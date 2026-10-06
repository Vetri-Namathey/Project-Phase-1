"""PLAN.md 2a (V0 pilot) visual check: 5 overlay PNGs from a record_route.py folder.

eval_spatial.py --carla-root gives the numbers; this gives the eyeball check the
V0 go/no-go also needs -- does the heatmap actually sit on the props, and does
the diff-based GT mask line up with what's visible in the frame?

Each PNG is [RGB with GT outline | fused-score heatmap over RGB with GT outline].
Picks the 5 frames with the most GT pixels, spread out over the route. Raw
model, no temperature -- this is a "does it fire on the right thing" check,
not a calibration one.

Run with CARLA CLOSED (8 GB GPU, see CLAUDE.md):
    python overlay_pilot.py --root video_pilot
Writes <root>/overlays/NNNNNN.png (gitignored with the rest of /video_*/).
"""

import argparse
import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

from data.transforms import load_image_tensor
from eval_spatial import carla_pairs
from train import amp_context
from utils import get_device, load_trained_model

N_FRAMES = 5
GT_COLOR = np.array([31, 164, 99], dtype=np.uint8)  # --good in index.css
HEAT_ALPHA = 0.55


def pick_frames(pairs, n):
    """The n frames with the most GT pixels, but at most one per route segment
    so the 5 aren't near-duplicates of the same prop from 1 m apart."""
    sizes = [int(np.load(m).sum()) for _, m in pairs]
    if not any(sizes):
        return [], sizes
    segments = np.array_split(np.arange(len(pairs)), n)
    picked = [int(seg[np.argmax([sizes[i] for i in seg])]) for seg in segments if len(seg)]
    return [i for i in picked if sizes[i] > 0], sizes


def outline(mask):
    return mask & ~binary_erosion(mask, iterations=2)


@torch.no_grad()
def fused_score(model, image_path, shape, device):
    with amp_context(device):
        out = model(load_image_tensor(image_path, device))
    logits = out["ood_logits"].float()
    up = F.interpolate(logits, size=shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(up).mean(dim=1)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="video_pilot")
    parser.add_argument("--raw", default="model_3head_best.pth",
                        help="repo-root checkpoint, NOT config.CHECKPOINT_3HEAD (stale)")
    args = parser.parse_args()

    pairs = carla_pairs(args.root)
    picked, sizes = pick_frames(pairs, N_FRAMES)
    print(f"{len(pairs)} frames, {sum(s > 0 for s in sizes)} with a visible prop")
    if not picked:
        print("No frame has any GT pixels -- V0 fails on recording, not on the model. "
              "Check props.json / record_route.py placement before scoring anything.")
        return

    device = get_device()
    model = load_trained_model(args.raw, device)
    out_dir = os.path.join(args.root, "overlays")
    os.makedirs(out_dir, exist_ok=True)

    for i in picked:
        image_path, mask_path = pairs[i]
        rgb = np.array(Image.open(image_path).convert("RGB"))
        mask = np.load(mask_path).astype(bool)
        score = fused_score(model, image_path, mask.shape, device)

        heat = (cm.inferno(score)[..., :3] * 255).astype(np.uint8)
        blended = (rgb * (1 - HEAT_ALPHA) + heat * HEAT_ALPHA).astype(np.uint8)
        edge = outline(mask)
        left, right = rgb.copy(), blended
        left[edge] = GT_COLOR
        right[edge] = GT_COLOR

        name = os.path.splitext(os.path.basename(image_path))[0]
        Image.fromarray(np.concatenate([left, right], axis=1)).save(
            os.path.join(out_dir, f"{name}.png"))
        print(f"frame {name}: gt_px={int(mask.sum())}  "
              f"mean score inside={score[mask].mean():.3f}  "
              f"outside={score[~mask].mean():.4f}  "
              f"max outside={score[~mask].max():.3f}")
    print(f"wrote {len(picked)} overlays to {out_dir}")


if __name__ == "__main__":
    main()

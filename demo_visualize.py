"""Prototype demo visualization: runs the trained TwinGuard checkpoint on
real Fishyscapes Lost&Found anomaly images and renders, per image, the raw
frame, predicted segmentation, fused OOD heatmap, ground-truth anomaly mask,
each of the 3 individual OOD heads, and a head-disagreement map (per-pixel
std across heads) -- the actual novelty signal this project claims.

Not a live demo (see PPT_Update_Content.html Slide 8 for that, planned for
after Phase 2b/calibration). This is a batch, static prototype showing the
trained detection pipeline actually working on real anomaly photos.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs
from model.twinguard_model import TwinGuardModel

NUM_IMAGES = 6
OUT_DIR = "demo_output"

# Standard Cityscapes trainId (0-18) color palette.
CITYSCAPES_PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)


def colorize_segmentation(class_map):
    return CITYSCAPES_PALETTE[class_map]


def pick_demo_pairs(n):
    """Prefer images with a substantial, clearly visible anomaly region --
    a tiny few-pixel blob makes for an unconvincing demo frame even if it's
    a technically valid Fishyscapes sample."""
    pairs = list_fishyscapes_pairs()
    scored = []
    for image_path, label_path in pairs:
        label = np.array(Image.open(label_path))
        anomaly_pixels = (label == 1).sum()
        scored.append((anomaly_pixels, image_path, label_path))
    scored.sort(reverse=True)
    return [(img, lbl) for _, img, lbl in scored[:n]]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_3HEAD, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded checkpoint: {config.CHECKPOINT_3HEAD}")

    os.makedirs(OUT_DIR, exist_ok=True)
    demo_pairs = pick_demo_pairs(NUM_IMAGES)
    print(f"selected {len(demo_pairs)} demo images (largest anomaly regions first)")

    with torch.no_grad():
        for idx, (image_path, label_path) in enumerate(demo_pairs):
            raw_image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = raw_image.size
            label_map = np.array(Image.open(label_path))

            image_resized = raw_image.resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
            image_t = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_t = image_t.to(device)

            out = model(image_t)

            seg_class_map = out["seg_logits"].argmax(dim=1).squeeze(0).cpu().numpy()
            seg_color = colorize_segmentation(seg_class_map)

            ood_scores = out["ood_scores"]  # (1, 3, H, W)
            fused = F.interpolate(
                out["ood_fused"].unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            per_head = F.interpolate(
                ood_scores, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze(0).cpu().numpy()  # (3, H, W)
            disagreement = per_head.std(axis=0)

            gt_display = np.where(label_map == 1, 1.0, 0.0)

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(
                f"TwinGuard prototype -- {os.path.basename(image_path)}",
                fontsize=13,
            )

            axes[0, 0].imshow(raw_image)
            axes[0, 0].set_title("Raw frame")

            seg_display = Image.fromarray(seg_color).resize((orig_w, orig_h), Image.NEAREST)
            axes[0, 1].imshow(seg_display)
            axes[0, 1].set_title("Predicted segmentation")

            axes[0, 2].imshow(fused, cmap="inferno", vmin=0, vmax=1)
            axes[0, 2].set_title("Fused OOD heatmap")

            axes[0, 3].imshow(gt_display, cmap="gray")
            axes[0, 3].set_title("Ground truth anomaly (Fishyscapes)")

            for h in range(3):
                axes[1, h].imshow(per_head[h], cmap="inferno", vmin=0, vmax=1)
                axes[1, h].set_title(f"Head {h}")

            axes[1, 3].imshow(disagreement, cmap="viridis")
            axes[1, 3].set_title("Head disagreement (std) -- the uncertainty signal")

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, f"demo_{idx:02d}.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out_path}")

    print(f"\ndone. {len(demo_pairs)} demo images written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

"""Lightweight demo backend (Section 08's live-demo plan, cut down for the
review timeline -- see PPT_Update_Content.html Slide 8 for the full planned
version with WebSocket streaming + CARLA playback, which comes after Phase
2b/calibration, not before).

Loads the verified checkpoint once at startup, runs real inference on the
curated demo images ONE TIME, and caches every panel as a static PNG --
so the actual review click-through is instant and can't fail on
GPU/inference hiccups live in front of an audience. The inference is real
(this is the actual trained TwinGuard pipeline), it just isn't re-run on
every click.
"""

import json
import os

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs
from model.twinguard_model import TwinGuardModel

NUM_IMAGES = 6
DISPLAY_WIDTH = 720
GRID_W = 96  # coarse value-lookup grid for hover readouts -- doesn't need per-pixel resolution
NUM_TRAINING_SAMPLES = 8
STATIC_DIR = "static"
GENERATED_DIR = os.path.join(STATIC_DIR, "generated")

# Real, logged results from the two actual RunPod training runs (see train_log.txt
# on each run -- these are not placeholders). Checkpoint A isolates the mit-b5
# backbone swap alone; Checkpoint B additionally splits the OOD heads onto their
# own 10x-lower learning rate (config.USE_OOD_HEAD_LR_SPLIT).
# Full progression, oldest first -- v1/v2/v3 are local-dev (mit-b2) attempts
# with only a single best-AUROC number logged (no per-epoch history kept at
# the time); Checkpoint A/B are the RunPod mit-b5 runs with full epoch logs.
# Numbers match PPT_Update_Content.html Slide 4 and this project's train_log.txt files.
EXPERIMENT_HISTORY = [
    {
        "id": "v1",
        "label": "Experiment B v1 -- unweighted loss",
        "description": "Below random chance -- severe class imbalance let the model learn \"predict nothing is anomalous\" as a free shortcut.",
        "best_auroc": 0.4932,
        "epochs": None,
    },
    {
        "id": "v2",
        "label": "Experiment B v2 -- reweighted loss",
        "description": "Real improvement from inverse-frequency positive-pixel weighting, but the OOD heads were still numerically collapsed underneath (mean/std diagnostic).",
        "best_auroc": 0.6282,
        "epochs": None,
    },
    {
        "id": "v3",
        "label": "Experiment B v3 -- CutMix tuning",
        "description": "Plateaued 0.62-0.63 -- three consecutive \"more/better data\" fixes produced the exact same collapse signature, ruling out data imbalance as the sole cause.",
        "best_auroc": 0.63,
        "epochs": None,
    },
    {
        "id": "checkpoint_a",
        "label": "Checkpoint A -- real mit-b5 backbone, RunPod A40",
        "description": "Isolates the backbone swap alone (flat LR, USE_OOD_HEAD_LR_SPLIT=False). Full 15 epochs run to completion. Did not clear the 0.75 gate -- backbone capacity ruled out as the sole cause.",
        "best_auroc": 0.5727,
        "epochs": [
            0.5481, 0.4735, 0.5727, 0.5414, 0.5431, 0.4877, 0.5266, 0.5001,
            0.4565, 0.4598, 0.5069, 0.5196, 0.5511, 0.5360, 0.5326,
        ],
    },
    {
        "id": "checkpoint_b",
        "label": "Checkpoint B -- + OOD head LR split (current demo checkpoint)",
        "description": "Run only after Checkpoint A's confirmed failure, per this project's own sequencing rule. OOD heads given their own 10x-lower LR. Best result at epoch 1 (near-initialization), degrading with further training -- LR split alone does not fully fix the collapse.",
        "best_auroc": 0.6190,
        "epochs": [0.6190, 0.4858, 0.4972, 0.4869, 0.5024, 0.5046, 0.5103, 0.4701],
    },
]

CITYSCAPES_PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)


def resize_for_display(image):
    w, h = image.size
    new_h = int(h * DISPLAY_WIDTH / w)
    return image.resize((DISPLAY_WIDTH, new_h), Image.BILINEAR)


def apply_colormap(array, cmap_name, out_size):
    cmap = matplotlib.colormaps[cmap_name]  # cm.get_cmap was removed in matplotlib>=3.9
    norm = np.clip(array, 0, 1)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored).resize(out_size, Image.BILINEAR)


def colorize_segmentation(class_map, out_size):
    color = CITYSCAPES_PALETTE[class_map]
    return Image.fromarray(color).resize(out_size, Image.NEAREST)


def save_grid(array, out_path, grid_w=GRID_W):
    """Downsample a 2D float array to a coarse grid and save as JSON, so the
    frontend can look up "the value near the cursor" on hover without
    shipping a full-resolution array over HTTP."""
    h, w = array.shape
    grid_h = max(1, round(grid_w * h / w))
    t = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    small = F.interpolate(t, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
    values = small.squeeze().numpy().round(4).tolist()
    if grid_h == 1:
        values = [values]
    with open(out_path, "w") as f:
        json.dump({"width": grid_w, "height": grid_h, "values": values}, f)


def draw_detection_box(raw_image, fused, out_size, percentile=97, min_area_frac=0.0005):
    """Threshold the fused heatmap at its own top percentile (fused scores run
    very low in absolute terms -- see check_collapse.py -- so a fixed
    threshold like 0.5 would find nothing) and draw a box around the largest
    resulting region. This is the actual "proof a specific object was
    flagged" panel, not a diffuse heatmap the audience has to interpret."""
    threshold = np.percentile(fused, percentile)
    mask = (fused > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = resize_for_display(raw_image)
    scale_x = display.width / fused.shape[1]
    scale_y = display.height / fused.shape[0]
    arr = np.array(display).copy()

    min_area = min_area_frac * fused.shape[0] * fused.shape[1]
    boxes_drawn = 0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x0, y0 = int(x * scale_x), int(y * scale_y)
        x1, y1 = int((x + w) * scale_x), int((y + h) * scale_y)
        cv2.rectangle(arr, (x0, y0), (x1, y1), (214, 64, 31), 3)  # frontend --accent
        boxes_drawn += 1

    return Image.fromarray(arr).resize(out_size, Image.BILINEAR), boxes_drawn


def score_localization(image_path, label_path, model, device):
    """How well does the model's own heatmap agree with the real anomaly
    mask for this frame? Mean score inside the true object minus mean score
    outside it -- used to pick the clearest real example to lead the demo
    with, instead of just the frame with the largest ground-truth area."""
    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    label_map = np.array(Image.open(label_path))

    image_resized = raw_image.resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
    image_t = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_t = image_t.to(device)

    with torch.no_grad():
        out = model(image_t)
    fused = F.interpolate(
        out["ood_fused"].unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    valid = label_map != 255
    inside = fused[valid & (label_map == 1)]
    outside = fused[valid & (label_map == 0)]
    if inside.size == 0 or outside.size == 0:
        return -1.0, 0.0, 0.0
    score_inside, score_outside = float(inside.mean()), float(outside.mean())
    return score_inside - score_outside, score_inside, score_outside


def pick_demo_pairs(n, model, device):
    pairs = list_fishyscapes_pairs()
    scored = []
    print(f"[demo backend] scoring {len(pairs)} candidate frames for best real-object localization...")
    for image_path, label_path in pairs:
        separation, score_inside, score_outside = score_localization(image_path, label_path, model, device)
        scored.append((separation, image_path, label_path, score_inside, score_outside))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:n]


def build_manifest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo backend] device: {device}")

    model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD).to(device)
    try:
        model.load_state_dict(torch.load(config.CHECKPOINT_3HEAD, map_location=device, weights_only=True))
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load {config.CHECKPOINT_3HEAD} into a model built with "
            f"USE_DEV_ENCODER={config.USE_DEV_ENCODER} ({config.ENCODER_NAME}). "
            f"The checkpoint's filename encodes the config it was trained with "
            f"(config.py's CHECKPOINT_3HEAD) -- check config.py matches the "
            f"checkpoint you're trying to load before rerunning the server."
        ) from e
    model.eval()
    print(f"[demo backend] loaded checkpoint: {config.CHECKPOINT_3HEAD}")

    os.makedirs(GENERATED_DIR, exist_ok=True)
    demo_pairs = pick_demo_pairs(NUM_IMAGES, model, device)
    print(f"[demo backend] precomputing {len(demo_pairs)} demo images (best real localization first)...")

    manifest = []
    with torch.no_grad():
        for idx, (separation, image_path, label_path, score_inside, score_outside) in enumerate(demo_pairs):
            image_id = f"demo_{idx:02d}"
            out_dir = os.path.join(GENERATED_DIR, image_id)
            os.makedirs(out_dir, exist_ok=True)

            raw_image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = raw_image.size
            out_size = (DISPLAY_WIDTH, int(orig_h * DISPLAY_WIDTH / orig_w))
            label_map = np.array(Image.open(label_path))
            anomaly_px = int((label_map == 1).sum())

            image_resized = raw_image.resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
            image_t = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_t = image_t.to(device)

            out = model(image_t)

            seg_class_map = out["seg_logits"].argmax(dim=1).squeeze(0).cpu().numpy()
            fused = F.interpolate(
                out["ood_fused"].unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            per_head = F.interpolate(
                out["ood_scores"], size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze(0).cpu().numpy()
            disagreement = per_head.std(axis=0)
            disagreement_norm = disagreement / (disagreement.max() + 1e-8)

            resize_for_display(raw_image).save(os.path.join(out_dir, "raw.png"))
            colorize_segmentation(seg_class_map, out_size).save(os.path.join(out_dir, "segmentation.png"))
            apply_colormap(fused, "inferno", out_size).save(os.path.join(out_dir, "fused.png"))
            for h in range(3):
                apply_colormap(per_head[h], "inferno", out_size).save(os.path.join(out_dir, f"head{h}.png"))
            apply_colormap(disagreement_norm, "viridis", out_size).save(os.path.join(out_dir, "disagreement.png"))
            gt = np.where(label_map == 1, 255, 0).astype(np.uint8)
            Image.fromarray(gt).resize(out_size, Image.NEAREST).save(os.path.join(out_dir, "ground_truth.png"))

            detection_img, boxes_drawn = draw_detection_box(raw_image, fused, out_size)
            detection_img.save(os.path.join(out_dir, "detection.png"))

            save_grid(fused, os.path.join(out_dir, "fused_grid.json"))
            for h in range(3):
                save_grid(per_head[h], os.path.join(out_dir, f"head{h}_grid.json"))

            manifest.append({
                "id": image_id,
                "title": os.path.basename(image_path),
                "anomaly_pixels": anomaly_px,
                "score_inside": round(score_inside, 4),
                "score_outside": round(score_outside, 4),
                "boxes_drawn": boxes_drawn,
                "panels": {
                    "raw": f"/static/generated/{image_id}/raw.png",
                    "detection": f"/static/generated/{image_id}/detection.png",
                    "segmentation": f"/static/generated/{image_id}/segmentation.png",
                    "fused": f"/static/generated/{image_id}/fused.png",
                    "ground_truth": f"/static/generated/{image_id}/ground_truth.png",
                    "head0": f"/static/generated/{image_id}/head0.png",
                    "head1": f"/static/generated/{image_id}/head1.png",
                    "head2": f"/static/generated/{image_id}/head2.png",
                    "disagreement": f"/static/generated/{image_id}/disagreement.png",
                },
                "grids": {
                    "fused": f"/static/generated/{image_id}/fused_grid.json",
                    "head0": f"/static/generated/{image_id}/head0_grid.json",
                    "head1": f"/static/generated/{image_id}/head1_grid.json",
                    "head2": f"/static/generated/{image_id}/head2_grid.json",
                },
            })
            print(
                f"[demo backend]   {image_id} -> {os.path.basename(image_path)} "
                f"(separation={separation:.4f}, inside={score_inside:.4f}, outside={score_outside:.4f}, "
                f"boxes={boxes_drawn})"
            )

    manifest_path = os.path.join(GENERATED_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[demo backend] manifest written to {manifest_path}")
    return manifest


def build_training_samples():
    """Not a live CARLA feed and not a recorded video -- these are the actual
    50 static single-frame synthetic anomalies used to train the OOD heads
    (see generate_anomalies.py). Each is a different random object at a
    different simulated moment, not a continuous recording of one scene --
    see the plan doc's Novelty 8 for why a real video sequence doesn't exist
    yet. This gallery exists so a reviewer can see real training-data
    examples, honestly labeled as static synthetic samples.
    """
    images_dir = config.CARLA_IMAGES_DIR
    masks_dir = config.CARLA_MASKS_DIR
    out_root = os.path.join(GENERATED_DIR, "training")
    os.makedirs(out_root, exist_ok=True)

    all_images = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    step = max(1, len(all_images) // NUM_TRAINING_SAMPLES)
    picked = all_images[::step][:NUM_TRAINING_SAMPLES]

    samples = []
    for idx, fname in enumerate(picked):
        sample_id = f"sample_{idx:02d}"
        out_dir = os.path.join(out_root, sample_id)
        os.makedirs(out_dir, exist_ok=True)

        num = fname.replace("anomaly_", "").replace(".png", "")
        mask_path = os.path.join(masks_dir, f"anomaly_mask_{num}.npy")

        raw = Image.open(os.path.join(images_dir, fname)).convert("RGB")
        w, h = raw.size
        out_size = (360, int(h * 360 / w))
        resize_for_display(raw).resize(out_size, Image.BILINEAR).save(os.path.join(out_dir, "raw.png"))

        mask = np.load(mask_path)
        overlay_arr = np.array(raw).copy()
        highlight = np.array([214, 64, 31])  # frontend's --accent (#D6401F), index.css
        overlay_arr[mask == 1] = (0.4 * overlay_arr[mask == 1] + 0.6 * highlight).astype(np.uint8)
        Image.fromarray(overlay_arr).resize(out_size, Image.BILINEAR).save(os.path.join(out_dir, "overlay.png"))

        samples.append({
            "id": sample_id,
            "source_file": fname,
            "panels": {
                "raw": f"/static/generated/training/{sample_id}/raw.png",
                "overlay": f"/static/generated/training/{sample_id}/overlay.png",
            },
        })

    samples_path = os.path.join(out_root, "samples.json")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[demo backend] {len(samples)} training samples written to {out_root}")
    return samples


app = FastAPI(title="TwinGuard Demo")

_manifest_cache = None
_training_samples_cache = None


@app.on_event("startup")
def startup():
    global _manifest_cache, _training_samples_cache

    manifest_path = os.path.join(GENERATED_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        print("[demo backend] found existing precomputed manifest, reusing it (delete static/generated/ to force a rebuild)")
        with open(manifest_path) as f:
            _manifest_cache = json.load(f)
    else:
        _manifest_cache = build_manifest()

    samples_path = os.path.join(GENERATED_DIR, "training", "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            _training_samples_cache = json.load(f)
    else:
        _training_samples_cache = build_training_samples()


@app.get("/api/manifest")
def get_manifest():
    return _manifest_cache


@app.get("/api/history")
def get_history():
    return EXPERIMENT_HISTORY


@app.get("/api/training-samples")
def get_training_samples():
    return _training_samples_cache


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

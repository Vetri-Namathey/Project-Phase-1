"""Lightweight demo backend (Section 08's live-demo plan, cut down for the
review timeline -- see PPT_Update_Content.html Slide 8 for the full planned
version with WebSocket streaming + CARLA playback, which comes after Phase
2b/calibration, not before).

Loads the primary checkpoint once at startup, runs real inference on the
demo frames ONE TIME, and caches every panel as a static PNG -- so the
actual review click-through is instant and can't fail on GPU/inference
hiccups live in front of an audience. The inference is real (this is the
actual trained TwinGuard pipeline), it just isn't re-run on every click.

Fixed 2026-10-05 (MISTAKES.md M13; the same bugs exp_v2 fixed on 2026-09-27),
click-through UX unchanged:
  * input goes through data.transforms.load_image_tensor -- the ImageNet
    normalisation the frozen encoder needs. It used to get raw [0,1] pixels.
  * no second sigmoid: ood_fused / ood_scores are already probabilities, and
    sigmoid on top squashed every displayed score into [0.5, 0.73]. Scores
    are now built exactly as in evaluation: upsample the per-head logits,
    sigmoid, mean over heads.
  * loads config.PRIMARY_RAW (the 2000-bank model), not the stale
    config.CHECKPOINT_3HEAD training-output file.
  * frames come from the Fishyscapes TEST half only, by a rule stated in the
    page caption: the 3 frames with the highest per-frame AP and the 3 around
    the median per-frame AP. The old code ranked all 100 frames (val
    included) by ground-truth localisation and showed the best 6.
  * the training gallery shows real CutMix composites (as exp_v2 did), not
    the curated CARLA frames' misaligned masks (M10)
  * panels are written to static/generated/test_half/ with a meta.json
    recording the checkpoint, so stale panels are never reused and the old
    ones in static/generated/demo_* stay on disk untouched.
"""

import base64
import io
import json
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from metrics import ScoreHistogram
from train import amp_context
from utils import load_trained_model

CHECKPOINT = config.PRIMARY_RAW
NUM_HIGHEST = 3
NUM_MEDIAN = 3
DISPLAY_WIDTH = 720
GRID_W = 96  # coarse value-lookup grid for hover readouts -- doesn't need per-pixel resolution
NUM_TRAINING_SAMPLES = 8
STATIC_DIR = "static"
GENERATED_DIR = os.path.join(STATIC_DIR, "generated")
DEMO_DIR = os.path.join(GENERATED_DIR, "test_half")
# CutMix training gallery; the old curated-CARLA gallery in generated/training/
# is left on disk, unused (M10).
TRAINING_DIR = os.path.join(GENERATED_DIR, "training_cutmix")
# Bump when the panel pipeline or the frame rule changes, so cached panels
# built by older code are rebuilt instead of served.
CACHE_VERSION = 2   # 2: boxes at the fixed val-fitted threshold (was a per-image percentile)

# Box threshold: the raw fused score's max-F1 threshold fitted on the
# Fishyscapes VAL half (eval_fishyscapes_T122.log, raw row, ubq_local line,
# input scale 1). Fixed, not tuned on the frames shown, so a frame with
# nothing above it gets no box -- that is the honest outcome.
BOX_THRESHOLD = 0.47658

# Real, logged results, oldest first. v1/v2/v3 are local-dev (mit-b2)
# attempts with only a single best-AUROC number logged; Checkpoint A/B are
# the RunPod mit-b5 runs with full epoch logs (PPT_Update_Content.html
# Slide 4). The last entry is the model every current number comes from
# (train.log).
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
        "label": "Checkpoint B -- + OOD head LR split",
        "description": "Run only after Checkpoint A's confirmed failure, per this project's own sequencing rule. OOD heads given their own 10x-lower LR. Best result at epoch 1 (near-initialization), degrading with further training -- LR split alone does not fully fix the collapse.",
        "best_auroc": 0.6190,
        "epochs": [0.6190, 0.4858, 0.4972, 0.4869, 0.5024, 0.5046, 0.5103, 0.4701],
    },
    {
        "id": "twinguard_2000bank",
        "label": "TwinGuard 3-head, CARLA + COCO outlier exposure (current demo checkpoint)",
        "description": "Frozen Cityscapes SegFormer-B5 encoder + 3 independently seeded heads with BCE-with-logits, "
                       "trained 8 epochs with CutMix pastes from a 2000-object bank (500 CARLA tiles of 45 objects "
                       "+ 1500 of 3000 COCO cutouts). Epoch 4 selected on the Fishyscapes val half (val AP 0.6839). "
                       "Fishyscapes test half: AUROC 0.9920, AP 0.6218, FPR@95 0.0287, ECE 0.0004; head "
                       "disagreement alone AUROC 0.9865; Cityscapes mIoU 0.7616 (train.log).",
        "best_auroc": 0.9920,
        "epochs": None,
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


def draw_detection_box(raw_image, fused, out_size, threshold=BOX_THRESHOLD, min_area_frac=0.0005):
    """Box the (up to 3) largest regions whose fused score is >= the fixed
    val-fitted threshold. The old rule used each image's own 97th
    percentile, which always drew a box even when nothing was there
    (MISTAKES.md M13).
    """
    mask = (fused >= threshold).astype(np.uint8)
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
        cv2.rectangle(arr, (x0, y0), (x1, y1), (61, 90, 254), 3)  # frontend --accent
        boxes_drawn += 1

    return Image.fromarray(arr).resize(out_size, Image.BILINEAR), boxes_drawn


@torch.no_grad()
def score_maps(model, image_path, shape, device):
    """-> (fused (H,W), per_head (heads,H,W), seg class map) at label size.

    Same order as eval_spatial.fused_scores: upsample the per-head logits,
    sigmoid, then mean -- so a demo score is the number evaluation reports.
    """
    with amp_context(device):
        out = model(load_image_tensor(image_path, device))
    logits = F.interpolate(out["ood_logits"].float(), size=shape, mode="bilinear", align_corners=False)
    per_head = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    seg = out["seg_logits"].argmax(dim=1).squeeze(0).cpu().numpy()
    return per_head.mean(axis=0), per_head, seg


def pick_demo_pairs(model, device):
    """Test half only. Ranks every test frame with an anomaly by its own AP
    and takes the NUM_HIGHEST best plus the NUM_MEDIAN around the median, so
    the demo shows typical frames, not only a highlight reel. Display only:
    nothing is fitted or reported from this choice."""
    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    scored = []
    print(f"[demo backend] ranking {len(test_pairs)} test-half frames by per-frame AP...")
    for image_path, label_path in test_pairs:
        label_map = np.array(Image.open(label_path))
        anomaly, valid = label_map == 1, label_map != 255
        if not anomaly.any():
            continue
        fused, _, _ = score_maps(model, image_path, label_map.shape, device)
        hist = ScoreHistogram()
        hist.update(fused[valid], anomaly[valid].astype(np.int64))
        scored.append((hist.average_precision(), image_path, label_path))
    scored.sort(key=lambda t: t[0], reverse=True)
    highest = scored[:NUM_HIGHEST]
    mid = len(scored) // 2 - NUM_MEDIAN // 2
    median = [s for s in scored[mid:mid + NUM_MEDIAN] if s not in highest]
    return ([(s, "highest AP") for s in highest] + [(s, "median AP") for s in median]), len(scored)


def build_manifest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo backend] device: {device}")
    model = load_trained_model(CHECKPOINT, device)
    print(f"[demo backend] loaded checkpoint: {CHECKPOINT}")

    os.makedirs(DEMO_DIR, exist_ok=True)
    chosen, n_ranked = pick_demo_pairs(model, device)
    print(f"[demo backend] precomputing {len(chosen)} test-half frames...")

    manifest = []
    for idx, ((frame_ap, image_path, label_path), selection) in enumerate(chosen):
        image_id = f"demo_{idx:02d}"
        out_dir = os.path.join(DEMO_DIR, image_id)
        os.makedirs(out_dir, exist_ok=True)

        raw_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        out_size = (DISPLAY_WIDTH, int(orig_h * DISPLAY_WIDTH / orig_w))
        label_map = np.array(Image.open(label_path))
        valid, anomaly = label_map != 255, label_map == 1

        fused, per_head, seg_class_map = score_maps(model, image_path, label_map.shape, device)
        disagreement = per_head.std(axis=0)
        disagreement_norm = disagreement / (disagreement.max() + 1e-8)

        resize_for_display(raw_image).save(os.path.join(out_dir, "raw.png"))
        colorize_segmentation(seg_class_map, out_size).save(os.path.join(out_dir, "segmentation.png"))
        apply_colormap(fused, "inferno", out_size).save(os.path.join(out_dir, "fused.png"))
        for h in range(per_head.shape[0]):
            apply_colormap(per_head[h], "inferno", out_size).save(os.path.join(out_dir, f"head{h}.png"))
        apply_colormap(disagreement_norm, "viridis", out_size).save(os.path.join(out_dir, "disagreement.png"))
        gt = np.where(anomaly, 255, 0).astype(np.uint8)
        Image.fromarray(gt).resize(out_size, Image.NEAREST).save(os.path.join(out_dir, "ground_truth.png"))

        detection_img, boxes_drawn = draw_detection_box(raw_image, fused, out_size)
        detection_img.save(os.path.join(out_dir, "detection.png"))

        save_grid(fused, os.path.join(out_dir, "fused_grid.json"))
        for h in range(per_head.shape[0]):
            save_grid(per_head[h], os.path.join(out_dir, f"head{h}_grid.json"))

        url = f"/static/generated/test_half/{image_id}"
        score_inside = float(fused[valid & anomaly].mean())
        score_outside = float(fused[valid & ~anomaly].mean())
        manifest.append({
            "id": image_id,
            "title": os.path.basename(image_path),
            "selection": selection,
            "frame_ap": round(float(frame_ap), 4),
            "anomaly_pixels": int(anomaly.sum()),
            "score_inside": round(score_inside, 4),
            "score_outside": round(score_outside, 4),
            "boxes_drawn": boxes_drawn,
            "panels": {name: f"{url}/{name}.png" for name in
                       ["raw", "detection", "segmentation", "fused", "ground_truth",
                        "head0", "head1", "head2", "disagreement"]},
            "grids": {name: f"{url}/{name}_grid.json" for name in ["fused", "head0", "head1", "head2"]},
        })
        print(f"[demo backend]   {image_id} [{selection}] {os.path.basename(image_path)} "
              f"AP={frame_ap:.4f} inside={score_inside:.4f} outside={score_outside:.4f} "
              f"boxes={boxes_drawn}")

    with open(os.path.join(DEMO_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(DEMO_DIR, "meta.json"), "w") as f:
        json.dump({"checkpoint": CHECKPOINT, "checkpoint_mtime": os.path.getmtime(CHECKPOINT),
                   "cache_version": CACHE_VERSION, "test_frames_ranked": n_ranked,
                   "rule": f"Fishyscapes test half; {NUM_HIGHEST} highest per-frame AP + "
                           f"{NUM_MEDIAN} around the median"}, f, indent=2)
    print(f"[demo backend] manifest written to {DEMO_DIR}")
    return manifest


def build_training_samples():
    """Real CutMix composites from the training pipeline: Cityscapes frames
    with objects from the 2000-object bank (CARLA tiles + COCO cutouts,
    config.ANOMALY_SOURCE) pasted on road/sidewalk, and the mask the heads
    were told is anomalous. Ported from exp_v2. The old gallery overlaid the
    45 curated CARLA frames' masks, several of which are misaligned
    (MISTAKES.md M10), and called them the training data; the bank only cuts
    tiles from them. Drawn from the val split with a fixed seed so the
    gallery is reproducible; the training split is composited the same way.
    Written to its own folder, so the old gallery stays on disk untouched.
    """
    import random

    from scipy import ndimage

    from data.cityscapes_dataset import CityscapesDataset
    from data.cutmix import CutMixAugmentedDataset
    from data.transforms import denormalize_imagenet

    random.seed(config.GLOBAL_SEED)   # CutMix draws objects, scales, positions with `random`
    np.random.seed(config.GLOBAL_SEED)
    ds = CutMixAugmentedDataset(CityscapesDataset(split="val", normalize=False), p=1.0)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    rng = np.random.default_rng(config.GLOBAL_SEED)

    samples = []
    for idx, i in enumerate(rng.choice(len(ds), NUM_TRAINING_SAMPLES, replace=False)):
        image_t, _, ood = ds[int(i)]
        rgb = (denormalize_imagenet(image_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = ood.numpy() > 0.5
        overlay = rgb.copy()
        highlight = np.array([61, 90, 254])  # frontend's --accent (#3D5AFE), index.css
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * highlight).astype(np.uint8)
        sample_id = f"sample_{idx:02d}"
        out_dir = os.path.join(TRAINING_DIR, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        size = (360, int(rgb.shape[0] * 360 / rgb.shape[1]))
        Image.fromarray(rgb).resize(size, Image.BILINEAR).save(os.path.join(out_dir, "raw.png"))
        Image.fromarray(overlay).resize(size, Image.BILINEAR).save(os.path.join(out_dir, "overlay.png"))
        n_obj = int(ndimage.label(mask)[1])
        url = f"/static/generated/training_cutmix/{sample_id}"
        samples.append({
            "id": sample_id,
            "source_file": f"Cityscapes val #{int(i)} + {n_obj} pasted object(s), "
                           f"{mask.mean():.2%} of pixels",
            "panels": {"raw": f"{url}/raw.png", "overlay": f"{url}/overlay.png"},
        })

    with open(os.path.join(TRAINING_DIR, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[demo backend] {len(samples)} CutMix training samples written to {TRAINING_DIR}")
    return samples


def _cache_is_current():
    """Reuse panels only if they were built by this pipeline version from
    this exact checkpoint file."""
    meta_path = os.path.join(DEMO_DIR, "meta.json")
    if not (os.path.exists(meta_path) and os.path.exists(os.path.join(DEMO_DIR, "manifest.json"))):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return (meta.get("cache_version") == CACHE_VERSION
            and meta.get("checkpoint") == CHECKPOINT
            and abs(meta.get("checkpoint_mtime", 0) - os.path.getmtime(CHECKPOINT)) < 1)


app = FastAPI(title="TwinGuard Demo")

_manifest_cache = None
_training_samples_cache = None


@app.on_event("startup")
def startup():
    global _manifest_cache, _training_samples_cache

    if not os.path.exists(CHECKPOINT):
        raise RuntimeError(f"no checkpoint at {CHECKPOINT} (config.PRIMARY_RAW) -- run from the repo root")
    if _cache_is_current():
        print(f"[demo backend] reusing precomputed test-half panels for {CHECKPOINT}")
        with open(os.path.join(DEMO_DIR, "manifest.json")) as f:
            _manifest_cache = json.load(f)
    else:
        _manifest_cache = build_manifest()

    samples_path = os.path.join(TRAINING_DIR, "samples.json")
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


# --- Live inference on an uploaded image ("yes" item 14) -------------------
# The request body is the raw image file (no multipart, so no extra package).
# Same preprocessing and scoring path as score_maps / evaluation, and the same
# fixed val-fitted box threshold. Timings use cuda.synchronize so they are real.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
LIVE_MAX_WIDTH = 2048  # score at most at Fishyscapes label width
_live_model = None


def _png_b64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def infer_bytes(data, model, device):
    t0 = time.perf_counter()
    raw_image = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = raw_image.size
    if w > LIVE_MAX_WIDTH:
        h, w = int(round(h * LIVE_MAX_WIDTH / w)), LIVE_MAX_WIDTH
    tensor = load_image_tensor(io.BytesIO(data), device)
    _sync(device)
    t1 = time.perf_counter()
    with amp_context(device):
        out = model(tensor)
    _sync(device)
    t2 = time.perf_counter()
    logits = F.interpolate(out["ood_logits"].float(), size=(h, w), mode="bilinear", align_corners=False)
    per_head = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    fused = per_head.mean(axis=0)
    disagreement = per_head.std(axis=0)
    _sync(device)
    t3 = time.perf_counter()

    out_size = (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w))
    display = resize_for_display(raw_image.resize((w, h), Image.BILINEAR))
    detection, boxes = draw_detection_box(raw_image.resize((w, h), Image.BILINEAR), fused, out_size)
    d_max = float(disagreement.max()) or 1.0
    return {
        "width": w, "height": h,
        "input": _png_b64(display),
        "fused": _png_b64(apply_colormap(fused, "inferno", out_size)),
        "disagreement": _png_b64(apply_colormap(disagreement / d_max, "viridis", out_size)),
        "detection": _png_b64(detection),
        "boxes": boxes,
        "threshold": BOX_THRESHOLD,
        "max_score": float(fused.max()),
        "flagged_fraction": float((fused >= BOX_THRESHOLD).mean()),
        "timing_ms": {"preprocess": round(1000 * (t1 - t0), 1), "model_forward": round(1000 * (t2 - t1), 1),
                      "upsample_and_score": round(1000 * (t3 - t2), 1), "total": round(1000 * (t3 - t0), 1)},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": CHECKPOINT,
    }


@app.post("/api/infer")
async def infer(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "send the image file as the request body")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(415, "not a readable image (use JPG or PNG)")
    model, device = _get_live_model()
    return infer_bytes(data, model, device)


# --- Uploaded video: score every sampled frame, return a playable mp4 --------
# Not real time: the clip is processed, then played back. Frames are sampled at
# VIDEO_FPS and capped at VIDEO_MAX_FRAMES so a long clip cannot hold the GPU
# for minutes. Each output frame = detection (boxes at BOX_THRESHOLD) on top,
# anomaly-score heatmap below, same scoring path as the image endpoint.
MAX_VIDEO_BYTES = 200 * 1024 * 1024
VIDEO_FPS = 10
VIDEO_MAX_FRAMES = 300  # 30 s at 10 fps
VIDEO_SCORE_WIDTH = 1024
UPLOAD_DIR = os.path.join(GENERATED_DIR, "uploads")
UPLOADS_KEPT = 5


def _get_live_model():
    global _live_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _live_model is None:
        _live_model = load_trained_model(CHECKPOINT, device)
        dummy = torch.zeros(1, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH, device=device)
        with torch.no_grad(), amp_context(device):
            _live_model(dummy)  # warm-up so the first timing is not the CUDA init
    return _live_model, device


@torch.no_grad()
def infer_video_bytes(data, model, device):
    import av
    from data.transforms import pil_to_tensor

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = f"upload_{int(time.time() * 1000)}.mp4"
    out_path = os.path.join(UPLOAD_DIR, name)
    src = av.open(io.BytesIO(data))
    vstream = src.streams.video[0]
    src_fps = float(vstream.average_rate or 25)
    step = max(1.0, src_fps / VIDEO_FPS)

    out, ostream = None, None
    forward_ms, total_ms, frames_with_box, max_score, kept = [], [], 0, 0.0, 0
    next_pick = 0.0
    for i, frame in enumerate(src.decode(vstream)):
        if i + 1e-6 < next_pick:
            continue
        next_pick += step
        t0 = time.perf_counter()
        img = frame.to_image()
        w, h = img.size
        sw = min(w, VIDEO_SCORE_WIDTH)
        sh = int(round(h * sw / w))
        tensor = pil_to_tensor(img, device)
        _sync(device)
        t1 = time.perf_counter()
        with amp_context(device):
            logits = model(tensor)["ood_logits"].float()
        _sync(device)
        forward_ms.append(1000 * (time.perf_counter() - t1))
        fused = torch.sigmoid(F.interpolate(logits, size=(sh, sw), mode="bilinear",
                                            align_corners=False)).mean(dim=1).squeeze(0).cpu().numpy()
        out_w = DISPLAY_WIDTH
        out_h = int(sh * DISPLAY_WIDTH / sw) // 2 * 2  # yuv420p needs even sizes
        detection, boxes = draw_detection_box(img.resize((sw, sh), Image.BILINEAR), fused, (out_w, out_h))
        heat = apply_colormap(fused, "inferno", (out_w, out_h))
        panel = np.concatenate([np.array(detection), np.array(heat)], axis=0)
        if out is None:
            out = av.open(out_path, mode="w")
            ostream = out.add_stream("libx264", rate=VIDEO_FPS)
            ostream.width, ostream.height = panel.shape[1], panel.shape[0]
            ostream.pix_fmt = "yuv420p"
            ostream.options = {"crf": "23"}
        for packet in ostream.encode(av.VideoFrame.from_ndarray(panel, format="rgb24")):
            out.mux(packet)
        total_ms.append(1000 * (time.perf_counter() - t0))
        frames_with_box += boxes > 0
        max_score = max(max_score, float(fused.max()))
        kept += 1
        if kept >= VIDEO_MAX_FRAMES:
            break
    src.close()
    if out is None:
        raise HTTPException(415, "no decodable video frames")
    for packet in ostream.encode():
        out.mux(packet)
    out.close()

    # keep only the newest few uploads on disk
    old = sorted(f for f in os.listdir(UPLOAD_DIR) if f.startswith("upload_") and f.endswith(".mp4"))
    for f in old[:-UPLOADS_KEPT]:
        os.remove(os.path.join(UPLOAD_DIR, f))

    return {
        "url": f"/static/generated/uploads/{name}",
        "frames": kept, "fps": VIDEO_FPS, "source_fps": round(src_fps, 2),
        "truncated": kept >= VIDEO_MAX_FRAMES,
        "frames_with_box": int(frames_with_box), "max_score": max_score,
        "threshold": BOX_THRESHOLD,
        "ms_per_frame": {"model_forward": round(float(np.median(forward_ms)), 1),
                         "total": round(float(np.median(total_ms)), 1)},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": CHECKPOINT,
    }


@app.post("/api/infer-video")
async def infer_video(request: Request):
    from starlette.concurrency import run_in_threadpool
    data = await request.body()
    if not data:
        raise HTTPException(400, "send the video file as the request body")
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, f"video larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    model, device = _get_live_model()
    try:
        return await run_in_threadpool(infer_video_bytes, data, model, device)
    except HTTPException:
        raise
    except Exception as e:  # av raises its own error types for unreadable files
        raise HTTPException(415, f"could not read this video ({type(e).__name__}); try an MP4")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

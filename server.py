"""TwinGuard demo backend (Phase 3).

Loads the trained checkpoint once, runs the REAL pipeline on held-out
Fishyscapes test frames ONE TIME, and caches every panel as a static PNG plus
a small value grid for hover readouts. The review click-through is then
instant and cannot fail on a GPU hiccup in front of an audience. Nothing is
simulated: every panel is this checkpoint's actual output.

Design kept from the original teammate version (precompute + static panels +
hover grids). Changed for correctness, 2026-09-27:
  * inputs go through data.transforms.load_image_tensor, the same ImageNet
    normalisation as training and evaluation (the old code fed raw [0,1]
    pixels, which the frozen encoder was never trained on)
  * no second sigmoid: ood_scores / ood_fused are already probabilities
    (the old code squashed every score into 0.5-0.73)
  * frames come from the TEST half only, 3 "clearest" + 3 "typical", each
    labelled as such, instead of the 6 best of all 100
  * detection boxes use the max-F1 threshold fitted on the VAL half (the UBQ
    operating point), and each box is checked against the real object
  * dual-mode: real continuous vs safety-triggered latency, whether the
    trigger fires, and the MC-Dropout within-head map beside head disagreement
  * the training gallery shows real CutMix samples (Cityscapes + COCO), which
    is what this checkpoint was trained on -- not the misaligned CARLA frames
  * the cache records which checkpoint produced it and rebuilds if that
    changes, so a stale demo can never be served for a new model

    python server.py                         # http://127.0.0.1:8000
    TWINGUARD_DEMO_CHECKPOINT=<path> python server.py
"""

import json
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from scipy import ndimage

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import denormalize_imagenet, load_image_tensor
from metrics import ScoreHistogram, ubq_local
from train import amp_context
from utils import load_trained_model

CHECKPOINT = os.environ.get("TWINGUARD_DEMO_CHECKPOINT", config.CHECKPOINT_3HEAD)
NUM_CLEAREST = 3
NUM_TYPICAL = 3
DISPLAY_WIDTH = 720
GRID_W = 96          # coarse value grid for hover readouts
NUM_TRAINING_SAMPLES = 8
LATENCY_REPEATS = 5
STATIC_DIR = "static"
GENERATED_DIR = os.path.join(STATIC_DIR, "generated")
REACT_DIR = "static_react"   # frontend `npm run build` output (vite.config.js outDir)
CACHE_VERSION = 3

BOX_COLOR = (61, 90, 254)    # frontend --accent: model's own detections
GT_COLOR = (0, 200, 83)      # ground-truth outline

# Real, logged results -- every number is in runs/RESULTS.md with its log.
# Fishyscapes Lost&Found TEST half unless marked; oldest first.
EXPERIMENT_HISTORY = [
    {"id": "early", "label": "Early attempts (mit-b2 / mit-b5, pre-audit)",
     "description": "Unweighted and reweighted loss, CutMix tuning, backbone swap: AUROC stuck at 0.49-0.63. "
                    "Root causes found in the audit: sigmoid-in-head gradient collapse, oversized pasted objects, "
                    "un-normalised encoder input, selection on the test set.",
     "auroc": 0.6282, "ap": None, "fpr95": 0.9823, "status": "superseded"},
    {"id": "exp_a", "label": "Experiment A: off-the-shelf SegFormer-B5 (MSP), no training",
     "description": "The baseline to beat. Confidently labels most anomaly pixels as a known class, so it cannot "
                    "reach 95% recall without flagging everything (FPR@95 = 1.0).",
     "auroc": 0.8372, "ap": 0.0114, "fpr95": 1.0, "status": "baseline"},
    {"id": "both", "label": "CARLA + COCO outlier exposure (main branch)",
     "description": "Pooling the 45 CARLA frames with COCO cost 0.17 AP. 23 of 45 CARLA masks are misaligned "
                    "(asynchronous capture), so CARLA is paused until it is re-rendered.",
     "auroc": 0.9920, "ap": 0.6218, "fpr95": 0.0287, "status": "superseded"},
    {"id": "run1", "label": "TwinGuard 3-head, COCO outlier exposure (phase2a_coco_run1)",
     "description": "Frozen Cityscapes SegFormer-B5 encoder + 3 independently seeded heads, 8 epochs on the laptop "
                    "GPU. Val-selected epoch 8. mIoU 0.770 (all 500 Cityscapes val images). Head disagreement "
                    "alone separates anomalies at AUROC 0.985.",
     "auroc": 0.9910, "ap": 0.7499, "fpr95": 0.0209, "status": "current",
     "epochs_ap": [0.6290, 0.7037, None, None, 0.7882, None, None, 0.7499]},
    {"id": "one_head", "label": "Ablation: 1 head (phase2a_1head_run1)",
     "description": "Same recipe, one head. Detection equal within noise (better on test, worse on val). It has "
                    "no disagreement signal; MC-Dropout on one head reaches 0.968 but needs 10 extra passes.",
     "auroc": 0.9938, "ap": 0.7770, "fpr95": 0.0152, "status": "ablation"},
    {"id": "calib2", "label": "Phase 2b: L_calib fine-tune with edge-confidence anchor (phase2b_calib2)",
     "description": "Edge-band ECE 0.189 vs 0.195 raw (95% CI includes 0) and 0.106 for edge-fitted temperature. "
                    "Reported as a finding: calibration learned on pasted training objects does not transfer to "
                    "real anomalies, because the model is ~99% sure of pastes and 50-69% sure of real objects.",
     "auroc": 0.9910, "ap": 0.7527, "fpr95": 0.0206, "status": "finding"},
]

CITYSCAPES_PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def display_size(w, h):
    return DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)


def apply_colormap(array, cmap_name, out_size, vmax=1.0):
    cmap = matplotlib.colormaps[cmap_name]
    norm = np.clip(array / max(vmax, 1e-8), 0, 1)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored).resize(out_size, Image.BILINEAR)


def colorize_segmentation(class_map, out_size):
    return Image.fromarray(CITYSCAPES_PALETTE[class_map]).resize(out_size, Image.NEAREST)


def save_grid(array, out_path, grid_w=GRID_W):
    """Coarse value grid so the frontend can show "the value under the
    cursor" without shipping a full-resolution array."""
    h, w = array.shape
    grid_h = max(1, round(grid_w * h / w))
    t = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    small = F.interpolate(t, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
    values = small.squeeze().numpy().round(5).tolist()
    if grid_h == 1:
        values = [values]
    with open(out_path, "w") as f:
        json.dump({"width": grid_w, "height": grid_h, "values": values}, f)


def draw_detections(raw_image, fused, anomaly, threshold, out_size, min_area_px=64):
    """Boxes around connected regions scoring >= threshold (the val-fitted
    operating point), ground-truth outline in green, and for each box whether
    it overlaps a real object."""
    arr = np.array(raw_image).copy()
    # Scaled to the image so the outline survives the downscale to
    # DISPLAY_WIDTH (a 2px line on a 2048px frame vanishes at 720px).
    thickness = max(2, round(2.5 * raw_image.width / DISPLAY_WIDTH))
    edge = ndimage.binary_dilation(anomaly, iterations=thickness) & ~anomaly
    arr[edge] = GT_COLOR

    labelled, n = ndimage.label(fused >= threshold)
    boxes, on_object = 0, 0
    for sl, k in zip(ndimage.find_objects(labelled), range(1, n + 1)):
        region = labelled[sl] == k
        if region.sum() < min_area_px:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        hit = bool(anomaly[sl][region].any())
        cv2.rectangle(arr, (x0, y0), (x1, y1), BOX_COLOR, thickness)
        boxes += 1
        on_object += hit
    return Image.fromarray(arr).resize(out_size, Image.BILINEAR), boxes, on_object


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def fused_map(model, image_path, shape, device):
    with amp_context(device):
        out = model(load_image_tensor(image_path, device))
    logits = F.interpolate(out["ood_logits"].float(), size=shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(logits).mean(dim=1).squeeze(0).cpu().numpy()


def load_label(label_path):
    label = np.array(Image.open(label_path))
    return label == 1, label != 255


def fit_operating_threshold(model, val_pairs, device):
    """Max-F1 threshold on the val half, the same operating point eval_spatial
    uses for UBQ. Fitted once, applied unchanged to the test frames shown."""
    hist = ScoreHistogram()
    for image_path, label_path in val_pairs:
        anomaly, valid = load_label(label_path)
        s = fused_map(model, image_path, anomaly.shape, device)
        hist.update(s[valid], anomaly[valid].astype(np.int64))
    return hist.threshold_at_max_f1()


def pick_frames(model, test_pairs, device):
    """Every test frame ranked by separation (mean score inside the real
    object minus outside). Show the clearest AND the typical ones, labelled,
    so the demo is not a highlight reel."""
    scored = []
    for image_path, label_path in test_pairs:
        anomaly, valid = load_label(label_path)
        if not anomaly.any():
            continue
        s = fused_map(model, image_path, anomaly.shape, device)
        inside, outside = s[anomaly & valid].mean(), s[~anomaly & valid].mean()
        scored.append((float(inside - outside), image_path, label_path))
    scored.sort(key=lambda t: t[0], reverse=True)
    mid = len(scored) // 2 - NUM_TYPICAL // 2
    chosen = [(s, "clearest") for s in scored[:NUM_CLEAREST]] + \
             [(s, "typical") for s in scored[mid:mid + NUM_TYPICAL]]
    return chosen, len(scored)


@torch.no_grad()
def measure_latency(model, x, device):
    """Continuous (encoder once, heads once) vs safety-triggered (same cached
    encoder features, heads x MC_DROPOUT_PASSES), bf16 as in evaluation."""
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timed(**kw):
        sync()
        t0 = time.perf_counter()
        for _ in range(LATENCY_REPEATS):
            with amp_context(device):
                model.predict_dual_mode(x, **kw)
        sync()
        return (time.perf_counter() - t0) / LATENCY_REPEATS * 1000

    timed(force_safety=True)   # warm-up
    return timed(trigger_threshold=1.01), timed(force_safety=True)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(model, device):
    all_pairs = list_fishyscapes_pairs()
    val_pairs, test_pairs = split_fishyscapes_pairs(all_pairs)
    print(f"[demo backend] fitting the max-F1 operating threshold on {len(val_pairs)} val frames...")
    threshold = fit_operating_threshold(model, val_pairs, device)
    print(f"[demo backend] threshold = {threshold:.4f}; ranking {len(test_pairs)} test frames...")
    chosen, n_ranked = pick_frames(model, test_pairs, device)

    frames = []
    for idx, ((separation, image_path, label_path), selection) in enumerate(chosen):
        frame_id = f"demo_{idx:02d}"
        out_dir = os.path.join(GENERATED_DIR, frame_id)
        os.makedirs(out_dir, exist_ok=True)

        raw_image = Image.open(image_path).convert("RGB")
        w, h = raw_image.size
        out_size = display_size(w, h)
        anomaly, valid = load_label(label_path)

        x = load_image_tensor(image_path, device)
        cont_ms, safety_ms = measure_latency(model, x, device)
        # Panels in fp32: the MC-Dropout variances are small enough that bf16
        # would round them away.
        with torch.no_grad():
            out = model.predict_dual_mode(x, force_safety=True)

        def up(t):
            return F.interpolate(t.float().unsqueeze(1) if t.dim() == 3 else t.float(),
                                 size=(h, w), mode="bilinear", align_corners=False)

        fused = up(out["ood_fused"]).squeeze().cpu().numpy()
        per_head = up(out["ood_scores"]).squeeze(0).cpu().numpy()
        between = up(out["parametric_uncertainty"]).squeeze().cpu().numpy()
        within = up(out["epistemic_uncertainty"]).squeeze().cpu().numpy()
        seg = out["seg_logits"].argmax(dim=1).squeeze(0).cpu().numpy()
        peak = float(out["peak_score"].max())

        Image.fromarray(np.array(raw_image)).resize(out_size, Image.BILINEAR).save(
            os.path.join(out_dir, "raw.png"))
        colorize_segmentation(seg, out_size).save(os.path.join(out_dir, "segmentation.png"))
        apply_colormap(fused, "inferno", out_size).save(os.path.join(out_dir, "fused.png"))
        for k in range(per_head.shape[0]):
            apply_colormap(per_head[k], "inferno", out_size).save(os.path.join(out_dir, f"head{k}.png"))
        # Uncertainty maps are scaled to their own frame maximum for display
        # (the raw values are tiny); the hover grid carries the raw values.
        apply_colormap(between, "viridis", out_size, vmax=float(between.max())).save(
            os.path.join(out_dir, "disagreement.png"))
        apply_colormap(within, "viridis", out_size, vmax=float(within.max())).save(
            os.path.join(out_dir, "within.png"))
        Image.fromarray(np.where(anomaly, 255, 0).astype(np.uint8)).resize(
            out_size, Image.NEAREST).save(os.path.join(out_dir, "ground_truth.png"))
        det, boxes, on_object = draw_detections(raw_image, fused, anomaly, threshold, out_size)
        det.save(os.path.join(out_dir, "detection.png"))

        grids = {"fused": fused, "disagreement": between, "within": within,
                 **{f"head{k}": per_head[k] for k in range(per_head.shape[0])}}
        for name, arr in grids.items():
            save_grid(arr, os.path.join(out_dir, f"{name}_grid.json"))

        hist = ScoreHistogram()
        hist.update(fused[valid], anomaly[valid].astype(np.int64))
        objects, _, _ = ubq_local((fused >= threshold) & valid, anomaly, valid=valid)
        found = [o for o in objects if o["detected"]]

        frames.append({
            "id": frame_id,
            "title": os.path.basename(image_path),
            "selection": selection,
            "anomaly_pixels": int(anomaly.sum()),
            "score_inside": round(float(fused[anomaly & valid].mean()), 4),
            "score_outside": round(float(fused[~anomaly & valid].mean()), 4),
            "frame_auroc": round(hist.auroc(), 4),
            "frame_ap": round(hist.average_precision(), 4),
            "boxes_drawn": boxes,
            "boxes_on_object": on_object,
            "objects_total": len(objects),
            "objects_found": len(found),
            "boundary_f1": round(float(np.mean([o["boundary_f1"] for o in found])), 3) if found else None,
            "peak_score": round(peak, 4),
            "safety_triggered": peak >= config.SAFETY_TRIGGER_THRESHOLD,
            "latency_continuous_ms": round(cont_ms, 1),
            "latency_safety_ms": round(safety_ms, 1),
            "panels": {name: f"/static/generated/{frame_id}/{name}.png" for name in
                       ["raw", "detection", "segmentation", "fused", "ground_truth",
                        "disagreement", "within"] + [f"head{k}" for k in range(per_head.shape[0])]},
            "grids": {name: f"/static/generated/{frame_id}/{name}_grid.json" for name in grids},
        })
        print(f"[demo backend]   {frame_id} [{selection}] {os.path.basename(image_path)} "
              f"sep={separation:.3f} AP={hist.average_precision():.3f} "
              f"found {len(found)}/{len(objects)} boxes {on_object}/{boxes} "
              f"latency {cont_ms:.0f}/{safety_ms:.0f} ms")

    return {
        "checkpoint": CHECKPOINT,
        "checkpoint_mtime": os.path.getmtime(CHECKPOINT),
        "cache_version": CACHE_VERSION,
        "threshold": round(threshold, 4),
        "trigger_threshold": config.SAFETY_TRIGGER_THRESHOLD,
        "mc_passes": config.MC_DROPOUT_PASSES,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "test_frames_ranked": n_ranked,
        "frames": frames,
    }


def build_training_samples():
    """Real CutMix composites from the training pipeline: Cityscapes frames
    with COCO objects pasted on the road, and the mask the heads were told is
    anomalous. Drawn from the val split with a fixed seed so the gallery is
    reproducible; the training split is composited the same way."""
    from data.cityscapes_dataset import CityscapesDataset
    from data.cutmix import CutMixAugmentedDataset

    random.seed(config.GLOBAL_SEED)
    np.random.seed(config.GLOBAL_SEED)
    ds = CutMixAugmentedDataset(CityscapesDataset(split="val", normalize=False),
                                p=1.0, degradation_prob=0.0)
    out_root = os.path.join(GENERATED_DIR, "training")
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(config.GLOBAL_SEED)

    samples = []
    for idx, i in enumerate(rng.choice(len(ds), NUM_TRAINING_SAMPLES, replace=False)):
        image_t, _, ood, _ = ds[int(i)]
        rgb = (denormalize_imagenet(image_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = ood.numpy() > 0.5
        overlay = rgb.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array(BOX_COLOR)).astype(np.uint8)
        sample_id = f"sample_{idx:02d}"
        out_dir = os.path.join(out_root, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        size = (360, int(rgb.shape[0] * 360 / rgb.shape[1]))
        Image.fromarray(rgb).resize(size, Image.BILINEAR).save(os.path.join(out_dir, "raw.png"))
        Image.fromarray(overlay).resize(size, Image.BILINEAR).save(os.path.join(out_dir, "overlay.png"))
        n_obj = int(ndimage.label(mask)[1])
        samples.append({
            "id": sample_id,
            "source_file": f"Cityscapes val #{int(i)} + {n_obj} COCO object(s), "
                           f"{mask.mean():.2%} of pixels",
            "panels": {"raw": f"/static/generated/training/{sample_id}/raw.png",
                       "overlay": f"/static/generated/training/{sample_id}/overlay.png"},
        })
    return samples


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="TwinGuard Demo")
_manifest = None
_training_samples = None


def _cache_is_current(manifest):
    return (manifest.get("cache_version") == CACHE_VERSION
            and manifest.get("checkpoint") == CHECKPOINT
            and abs(manifest.get("checkpoint_mtime", 0) - os.path.getmtime(CHECKPOINT)) < 1)


@app.on_event("startup")
def startup():
    global _manifest, _training_samples
    if not os.path.exists(CHECKPOINT):
        raise RuntimeError(f"no checkpoint at {CHECKPOINT} -- train first or set TWINGUARD_DEMO_CHECKPOINT")
    os.makedirs(GENERATED_DIR, exist_ok=True)

    manifest_path = os.path.join(GENERATED_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            cached = json.load(f)
        if isinstance(cached, dict) and _cache_is_current(cached):
            print(f"[demo backend] reusing precomputed panels for {CHECKPOINT}")
            _manifest = cached
    if _manifest is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[demo backend] building panels for {CHECKPOINT} on {device} (one time)...")
        model = load_trained_model(CHECKPOINT, device)
        _manifest = build_manifest(model, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with open(manifest_path, "w") as f:
            json.dump(_manifest, f, indent=2)

    samples_path = os.path.join(GENERATED_DIR, "training", "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            _training_samples = json.load(f)
    else:
        _training_samples = build_training_samples()
        with open(samples_path, "w") as f:
            json.dump(_training_samples, f, indent=2)
    print("[demo backend] ready -> http://127.0.0.1:8000")


@app.get("/api/manifest")
def get_manifest():
    return _manifest


@app.get("/api/history")
def get_history():
    return EXPERIMENT_HISTORY


@app.get("/api/training-samples")
def get_training_samples():
    return _training_samples


@app.get("/api/explain")
def get_explain():
    """Explainability figures + aggregate numbers, written by explain.py
    (`python explain.py --checkpoint <ckpt> --demo`). Not built by the server:
    they take a few minutes of GPU and are produced on demand. Until they exist
    the page shows a clearly labelled "pending" notice."""
    path = os.path.join(GENERATED_DIR, "explain", "explain.json")
    if not os.path.exists(path):
        return {"status": "pending",
                "how": "python explain.py --checkpoint <ckpt>   then   python explain.py --checkpoint <ckpt> --demo"}
    with open(path) as f:
        # parse_constant: an older explain.json may hold a bare NaN, which is
        # not valid JSON and which FastAPI refuses to send. Serve it as null.
        return json.load(f, parse_constant=lambda _: None)


os.makedirs(GENERATED_DIR, exist_ok=True)   # StaticFiles refuses a missing directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.isdir(os.path.join(REACT_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(REACT_DIR, "assets")), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    """The built React app for every non-API route (/, /demo, /runs, ...)."""
    if path.startswith("api/"):
        raise HTTPException(status_code=404)
    index = os.path.join(REACT_DIR, "index.html")
    candidate = os.path.join(REACT_DIR, path)
    if path and os.path.isfile(candidate):
        return FileResponse(candidate)
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=503, detail="frontend not built: cd frontend && npm install && npm run build")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# TwinGuard

Out-of-distribution (OOD) detection for autonomous-vehicle perception. Instead of a segmentation model confidently mislabeling an object it's never seen (debris, an animal, fallen cargo), TwinGuard flags it as unknown — with a confidence score designed to actually be trustworthy, not just high.

Frozen SegFormer backbone + segmentation head + 3 independently-seeded OOD detection heads. Head disagreement is the uncertainty signal. Planned dual-mode inference: a fast deterministic pass normally, a slower MC-Dropout re-check only when something looks anomalous.

## Current status

**Experiment A (baseline, complete)** — off-the-shelf `nvidia/segformer-b5-finetuned-cityscapes`, no training, scored via Max Softmax Probability. Real result on Fishyscapes Lost&Found:

| AUROC | ECE | FPR@95 |
|---|---|---|
| 0.8304 | 0.0154 | 0.6802 |

**Experiment B (TwinGuard, RunPod mit-b5 runs complete)** — after fixing an OOD-head collapse bug (severe class imbalance letting the model learn "predict nothing is anomalous"), two real RunPod A40 runs on the full `mit-b5` backbone:

| Run | Config | Best AUROC |
|---|---|---|
| Checkpoint A | flat LR (`USE_OOD_HEAD_LR_SPLIT=False`) | 0.5727 |
| Checkpoint B | OOD heads at 10x lower LR (`USE_OOD_HEAD_LR_SPLIT=True`) | **0.6190** (epoch 1; degrades with further training) |

Neither run clears the 0.75 pre-calibration gate yet. Checkpoint B's epoch-1 weights are the current live-demo checkpoint (see `checkpoints/`, loaded by `server.py`). Full per-epoch logs for both runs are in `server.py`'s `EXPERIMENT_HISTORY` and rendered on the demo's "Training Runs" page.

Numeric gates: **0.75 AUROC** is the pre-calibration gate to proceed to Phase 2b (`L_calib`); **0.83** (Experiment A's baseline) remains the actual post-calibration target — the bar hasn't moved, calibration is a stated, accounted-for tradeoff (see `config.py`'s `CALIBRATION_TRADEOFF_NOTE`).

## Setup

```
pip install -r requirements.txt          # local (Python 3.8-constrained versions)
pip install -r requirements-runpod.txt   # cloud (RunPod, modern Python, skips torch reinstall)
```

Dataset paths are hardcoded in `config.py` (`CITYSCAPES_IMAGES_ROOT`, `CITYSCAPES_LABELS_ROOT`, `FISHYSCAPES_LABELS_DIR`, `FISHYSCAPES_IMAGES_ROOT`) — update these four lines to wherever Cityscapes and Fishyscapes Lost&Found actually live on your machine. CARLA-generated anomaly data lives at `data/images/` and `data/masks/`, produced by `generate_anomalies.py` (requires a running CARLA server).

## Running things

```
python experiment_a_eval.py     # real Fishyscapes evaluation for the baseline
python train.py                 # train TwinGuard (Experiment B)
python check_collapse.py        # mean/std diagnostic on the trained OOD heads
python compare_results.py       # pull all MLflow runs, compare metrics
python check_epoch_trend.py     # full per-epoch history for the latest run
python check_cutmix_overlay.py  # visual sanity check of CutMix pasting
```

## Live demo (React frontend + FastAPI backend)

A click-through demo of real inference on the current checkpoint — bounding-box detection,
per-head OOD heatmaps, head-disagreement uncertainty, and the full training-run history.
Inference runs once at backend startup and is cached as static images, so the actual
demo click-through doesn't depend on live GPU inference.

```
python server.py            # backend, http://127.0.0.1:8000
cd frontend && npm install && npm run dev   # frontend, http://localhost:5173
```

The frontend dev server proxies `/api` and `/static` to the backend (`frontend/vite.config.js`),
so both processes need to be running. To force the backend to rebuild its cached demo
output (e.g. after changing `server.py`'s panel/score logic), delete `static/generated/`
and restart it.

## Cloud training (RunPod)

See `RunPod_Setup_Guide.html` for the full step-by-step — pod creation, SSH setup, uploading code+data, and running training on an A40 GPU. `requirements-runpod.txt` is the cloud-specific dependency list (deliberately doesn't reinstall torch, since the pod template ships a CUDA-matched build already).

## Key planning documents

- `TwinGuard_Workflow_For_ClaudeCode.html` — the original architecture/spec
- `TwinGuard_Full_Plan_Updated (1).html` — current novelty positioning, checkpoint plan, and literature grounding
- `TwinGuard_Progress_Dashboard.html`, `Experiment_B_v2_Update.html` — result snapshots from earlier in the project

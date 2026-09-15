# TwinGuard

Out-of-distribution (OOD) detection for autonomous-vehicle perception. Instead of a segmentation model confidently mislabeling an object it's never seen (debris, an animal, fallen cargo), TwinGuard flags it as unknown — with a confidence score designed to actually be trustworthy, not just high.

Frozen SegFormer backbone + segmentation head + 3 independently-seeded OOD detection heads. Head disagreement is the uncertainty signal. Planned dual-mode inference: a fast deterministic pass normally, a slower MC-Dropout re-check only when something looks anomalous.

## Current status

**Experiment A (baseline, complete)** — off-the-shelf `nvidia/segformer-b5-finetuned-cityscapes`, no training, scored via Max Softmax Probability. Real result on Fishyscapes Lost&Found:

| AUROC | ECE | FPR@95 |
|---|---|---|
| 0.8304 | 0.0154 | 0.6802 |

**Experiment B (TwinGuard, in progress)** — best result so far is **0.6282 AUROC**, still below the baseline. A collapse bug (severe class imbalance letting the model learn "predict nothing is anomalous") was found and partially fixed; current leading hypothesis is that the local dev backbone (`mit-b2`, generic ImageNet-only) is undertrained relative to Experiment A's Cityscapes-finetuned `mit-b5`. Currently testing the real `mit-b5` backbone on a RunPod A40 GPU.

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

## Cloud training (RunPod)

See `RunPod_Setup_Guide.html` for the full step-by-step — pod creation, SSH setup, uploading code+data, and running training on an A40 GPU. `requirements-runpod.txt` is the cloud-specific dependency list (deliberately doesn't reinstall torch, since the pod template ships a CUDA-matched build already).

## Key planning documents

- `TwinGuard_Workflow_For_ClaudeCode.html` — the original architecture/spec
- `TwinGuard_Full_Plan_Updated (1).html` — current novelty positioning, checkpoint plan, and literature grounding
- `TwinGuard_Progress_Dashboard.html`, `Experiment_B_v2_Update.html` — result snapshots from earlier in the project

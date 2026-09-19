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

Dataset paths default to `config.py`'s `_DEFAULT_DATA_ROOT` (one unified root) but each of
the four can be overridden individually by its own env var — `CITYSCAPES_IMAGES_ROOT`,
`CITYSCAPES_LABELS_ROOT`, `FISHYSCAPES_LABELS_DIR`, `FISHYSCAPES_IMAGES_ROOT` — for a machine
where the data doesn't live under one shared folder (see `config.py`'s comment block at the
top for the exact mechanism). Run `python preflight.py` after setting these — it fails fast
with a clear message if a path is wrong, rather than an obscure `FileNotFoundError` mid-epoch.

Anomaly (outlier-exposure) objects for CutMix come from two sources behind one interface
(`data/anomaly_sources.py`), selected via `config.ANOMALY_SOURCE = "carla" | "coco" | "both"`:
- **CARLA** — `data/images/` + `data/masks/`, produced by `generate_anomalies.py` (needs a
  running CARLA server). 45 curated objects (5 miscategorized/oversized props excluded — see
  `PLAN.md`'s curation log; excluded originals kept at `data/_excluded_objects/`, not deleted).
- **COCO** — `data/coco_objects/`, produced by `python download_coco_anomalies.py` (one-time
  ~1GB download of COCO val2017; every Cityscapes-overlapping category is excluded via
  `config.COCO_EXCLUDED_CATEGORIES`, self-verified at the end of the script).
- **`"both"`** pools the two curated banks into one, sampled uniformly per paste — the
  current default, so the model doesn't lock onto either source's own low-level statistical
  signature (CG-render tells vs. COCO-photo tells) as a shortcut.

## Running things

```
python preflight.py             # 13-check pipeline sanity gate -- run before any GPU spend
python experiment_a.py          # baseline: real Fishyscapes eval via Max Softmax Probability
python train.py                 # train TwinGuard (Experiment B)
python check_data.py            # CutMix composites + object-size stats for the active bank
python check_data.py --object N # inspect one raw object from that bank
python check_runs.py            # pull all MLflow runs, compare metrics
python check_runs.py --trend    # per-epoch history for the latest run
python check_collapse.py        # separation diagnostic on the trained OOD heads
python validate_metrics.py      # self-check of metrics.py against reference values
python make_upload.py           # package code (no data) into twinguard_code.zip for RunPod
```

Everything above runs from the repo root. `generate_anomalies.py` (CARLA object
rendering) and `download_coco_anomalies.py` (COCO bank) are the two data-producing
scripts; `server.py` is the demo backend (see below).

### `legacy/` — superseded, kept for reference

`legacy/` holds the pre-merge versions of scripts that the audited pipeline replaced,
plus three one-off CARLA semantic-tag probes. Nothing in the pipeline imports them and
none of them are packaged by `make_upload.py`. Each file's docstring names its
replacement:

| legacy | use instead |
|---|---|
| `legacy/experiment_a_baseline.py`, `legacy/experiment_a_eval.py` | `experiment_a.py` |
| `legacy/compare_results.py`, `legacy/check_epoch_trend.py` | `check_runs.py` (`--trend`) |
| `legacy/check_overlay.py`, `legacy/check_cutmix_overlay.py` | `check_data.py` |
| `legacy/check_tag*.py` | — one-off CARLA probes, superseded by `generate_anomalies.py` |

`legacy/check_cutmix_overlay.py` no longer runs at all against the current data
classes (it predates ImageNet normalisation) — that is why it is here rather than in
the list above.

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

See `RUNPOD_GUIDE.md` for the current step-by-step (pod creation, SSH setup, `setup_env.sh`,
`make_upload.py` for packaging code+data, `pod_survey.sh` for a pre-flight hardware check, and
running training on an A40 GPU). The older `RunPod_Setup_Guide.html` still exists but predates
these scripts. `requirements-runpod.txt` is the cloud-specific dependency list (deliberately
doesn't reinstall torch, since the pod template ships a CUDA-matched build already).

## Repository layout

```
config.py losses.py metrics.py train.py utils.py   core pipeline
data/            datasets, CutMix, anomaly banks (images/, masks/, coco_objects/
                 are generated and gitignored -- see "Setup")
model/           SegFormer wrapper + OOD heads
preflight.py validate_metrics.py check_*.py        diagnostics, run from repo root
experiment_a.py train.py                           the two experiments
setup_env.sh pod_survey.sh make_upload.py          RunPod tooling (RUNPOD_GUIDE.md)
server.py static/ frontend/                        live demo (backend, assets, React app)
generate_anomalies.py download_coco_anomalies.py   data producers
legacy/          superseded scripts, see above
```

Scripts stay at the repo root rather than under a `scripts/` folder on purpose: every
one of them does `import config` / `from data...`, and Python puts the *script's* own
directory on `sys.path` (not the working directory), so a nested script only works with
an explicit path bootstrap. `legacy/` carries that bootstrap because nothing there is
run routinely; the scripts you actually run every day do not need it.

## Key planning documents

These are HTML files at the repo root and are **gitignored** (generated planning/report
docs, not source) — a fresh clone will not have them. Ask for them directly if you need
them.

- `TwinGuard_Workflow_For_ClaudeCode.html` — the original architecture/spec
- `TwinGuard_Full_Plan_Updated (1).html` — current novelty positioning, checkpoint plan, and literature grounding
- `TwinGuard_Progress_Dashboard.html`, `Experiment_B_v2_Update.html` — result snapshots from earlier in the project

# TwinGuard demo — frontend

React + Vite dashboard for the live demo. It has no model code of its own. Everything it
shows comes from `server.py` (FastAPI) in the repository root, which runs the trained
checkpoint and serves:

| Endpoint | Content |
|---|---|
| `/api/manifest` | the demo frames: panel image URLs, hover-grid URLs, per-frame metrics, latency |
| `/api/history` | every training run, in order, with its status |
| `/api/training-samples` | real CutMix training samples |
| `/static/generated/...` | the panel PNGs and value grids (built by `server.py` on first start) |

## Requirements

- **Node.js ≥ 20.19** (Vite 8). Checked with 22.12.
- The Python side set up as in the main README, with a trained 3-head checkpoint at
  `checkpoints/model_3head_best.pth` (or `TWINGUARD_DEMO_CHECKPOINT` pointing at one).
  Ask a teammate for it; it isn't in git.

## Build once, then run from Python (the normal way)

```bash
cd frontend
npm ci              # installs exactly what package-lock.json pins
npm run build       # writes ../static_react/
cd ..
python server.py    # http://127.0.0.1:8000 serves the built app + API
```

The first `python server.py` renders every panel with the real model (a few minutes on a
GPU) and caches them in `static/generated/`. Later starts take seconds.

## Editing with live reload

```bash
python server.py                 # terminal 1: backend on :8000
cd frontend && npm run dev       # terminal 2: open the URL it prints
```

`vite.config.js` forwards `/api` and `/static` to port 8000. After editing, run
`npm run build` again so `python server.py` alone serves the new version.

## Pages

| Route | File | Shows |
|---|---|---|
| `/` | `src/pages/HomePage.jsx` | headline test-half numbers, how it works, calibration comparison |
| `/demo` | `src/pages/DemoPage.jsx` | 6 held-out test frames (3 clearest + 3 typical, labelled): detections vs true outline, dual-mode latency + both uncertainty maps, per-head heatmaps with hover values |
| `/runs` | `src/pages/HistoryPage.jsx` | findings table + every run, including failed ones |
| `/training` | `src/pages/TrainingPage.jsx` | what the model trained on (Cityscapes + pasted COCO objects) |

The numbers hard-coded in `HomePage.jsx` / `HistoryPage.jsx` and in `EXPERIMENT_HISTORY` in
`server.py` are copied from `runs/RESULTS.md`. Update all three together if a new run
replaces the reported one.

`npm run lint` runs oxlint. There's one known warning, `set-state-in-effect` in
`DemoPage.jsx`. That's intentional: it clears the previous frame's hover values before the
new ones load.

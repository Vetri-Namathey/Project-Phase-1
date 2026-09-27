# TwinGuard

**Out-of-distribution (OOD) detection for autonomous-vehicle perception.**

A normal segmentation model knows 19 things — road, car, person, traffic light, and so on. Show it a mattress that fell off a truck and it has no "mattress" option, so it picks the nearest label it does have and reports high confidence. TwinGuard adds a second opinion: it flags the pixels it does not recognise, and attaches an uncertainty estimate that is meant to be *trustworthy* rather than merely confident-looking.

> **Branches.** `exp_v2` (this one) is the current, complete pipeline: Phase 2a training,
> Phase 2b calibration, evaluation (UBQ, second benchmark, ablations) and the live demo.
> `exp_coco` is the earlier COCO-only pipeline this branch was built from, and `main` holds
> the CARLA+COCO work and the original demo. Both are kept unchanged for reference.
>
> **What is not in the repository:** datasets, trained checkpoints, MLflow history and
> training logs. They are too large or not ours to redistribute. See
> [Get the data](#2-get-the-datasets) and [Get a checkpoint](#6-get-a-trained-checkpoint-optional).
> Every result quoted here is in [runs/RESULTS.md](runs/RESULTS.md), with how it was produced.

---

## Results

Evaluated on **Fishyscapes Lost & Found**, on a held-out 50-image test half that is never used for training or checkpoint selection.

| Method | AUROC ↑ | AP ↑ | FPR@95 ↓ | Reports ECE? |
|---|---|---|---|---|
| Experiment A — off-the-shelf SegFormer-B5 + MSP (no training) | 0.8372 | 0.0114 | 1.0000 | — |
| DenseHybrid (ECCV 2022) | — | 0.4390 | 0.0620 | no |
| PEBAL (ECCV 2022) | 0.9896 | 0.5881 | 0.0477 | no |
| RbA (ICCV 2023) | 0.9862 | 0.7081 | 0.0630 | no |
| **TwinGuard (this repo, `phase2a_coco_run1`)** | **0.9910** | **0.7499** | **0.0209** | **yes — 0.0003** |

> **These rows are not a like-for-like leaderboard.** The published figures are copied from
> each paper's own results table and were not re-run here. Depending on the paper, they're
> measured on the full public 100-image Fishyscapes Lost & Found set or on the official
> hidden test set, with each paper's own model and training data. TwinGuard and
> Experiment A are measured on a fixed 50-image half of the public set (the other half
> selects the checkpoint). The numbers show TwinGuard is in the same range as published
> methods. They do **not** establish that it beats them. That needs the same images and
> protocol, ideally a submission to the official benchmark.
>
> An earlier pod run (run 3) scored AP 0.7917 / FPR@95 0.0168; its checkpoint was lost, and
> the local reproduction above is the reportable one. Two epochs with near-equal validation
> AP differed by 0.04 test AP, so treat about ±0.04 AP as noise on 50 images.

Supporting numbers for the same checkpoint:

| | |
|---|---|
| **mIoU** on Cityscapes val (all 500 images) | **0.7700**: segmentation is not degraded by the OOD heads (1-head model: 0.7681; no class moves by more than 0.02) |
| **Head-disagreement AUROC** | **0.9845**: disagreement between the three heads, scored as a detector *on its own* |
| Selected epoch | 8 of 8, by best validation AP |
| Training time | ~80 min (8 epochs × ~10 min) on an RTX 3050 Ti laptop GPU |

All local results, including the ones that didn't work, are in [runs/RESULTS.md](runs/RESULTS.md).

**Why the disagreement number matters.** The architecture's premise is that three independently-initialised heads disagreeing is a real signal of "the model does not know." Measured alone, with no other information, that disagreement detects anomalies at 0.9845 AUROC. It is evidence, not an assumption. The 1-head ablation shows what it buys: one head detects just as well, but gets a comparable uncertainty signal (MC-Dropout, 0.968) only by running 10 extra passes. Three heads give it for free in the single continuous-mode pass.

**Staging gates.** 0.75 AUROC is the pre-calibration gate to begin Phase 2b (`L_calib`); **0.83** — Experiment A's baseline — remains the real post-calibration target. Both are cleared. The bar did not move: calibration is a stated, accounted-for tradeoff, not a lowered goal (see `CALIBRATION_TRADEOFF_NOTE` in `config.py`).

---

## Quick start

For a developer who has never seen this repo.

### 1. Install

```bash
pip install -r requirements.txt
```

Two version constraints in that file are load-bearing and explained inline — do not loosen them without reading the comments. `preflight.py` checks both and prints the fix if either is wrong.

### 2. Get the datasets

**No dataset is in this repository.** Ask a teammate who already has them before
downloading anything: the full set is on the project lead's machine and on the shared
RunPod volume (`/workspace/data`). Copying them saves hours and guarantees the same files.
The public sources are listed below so each one can be traced.

Put them under one root, with exactly these folder names:

```
<DATA_ROOT>/
├── leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/<city>/   Cityscapes images   (required)
├── gtFine_trainvaltest/gtFine/{train,val,test}/<city>/             Cityscapes labels   (required)
├── fishyscapes_lostandfound/                                       100 anomaly label PNGs (required)
├── leftImg8bit/leftImg8bit/{train,test}/<city>/                    Lost & Found photos (required)
└── road_anomaly21/dataset_AnomalyTrack/{images,labels_masks}/      RoadAnomaly21       (optional)
```

| Dataset | Used for | Source | Check |
|---|---|---|---|
| **Cityscapes** (`leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`) | training + mIoU | free account at <https://www.cityscapes-dataset.com>. Its licence does not allow public redistribution, so never upload it anywhere public | 2975 train / 500 val images |
| **Fishyscapes Lost & Found** labels | evaluation only (never trained on) | <https://fishyscapes.com> | 100 label PNGs |
| **Lost & Found** photos | the images those labels belong to | <http://www.6d-vision.com/lostandfounddataset> | the code pairs them by filename; 100/100 must match |
| **RoadAnomaly21** (optional) | second evaluation benchmark, 10 labelled images | `dataset_AnomalyTrack.zip`, <https://zenodo.org/records/5270237>, md5 `231bf79ed58924bcd33d9cbe22e61076` | 10 validation images with labels |

Fishyscapes labels and Lost & Found photos come from different sources and don't share a
folder tree. The code pairs them by filename, and `preflight.py` fails if any pair is missing.

Point the code at the root:

```bash
export TWINGUARD_DATA_ROOT=/path/to/your/data          # Linux / Git Bash
$env:TWINGUARD_DATA_ROOT = "D:\path\to\your\data"      # Windows PowerShell
```

Each path can also be overridden on its own with an environment variable named after its
`config.py` constant (`CITYSCAPES_IMAGES_ROOT`, `FISHYSCAPES_LABELS_DIR`, `ROAD_ANOMALY21_DIR`, …).
The default in `config.py` is one developer's Windows folder, a fallback only.

### 3. Build the anomaly bank

```bash
python download_coco_anomalies.py
```

~1GB download, a few minutes. It writes `data/coco_objects/` (3000 cutouts, 68 categories),
which is ignored by git: every machine builds its own, or copies it from a teammate. See
[Anomaly source](#anomaly-source) for what this is and why.

### 4. Verify before you train

```bash
python preflight.py          # must end: ALL CHECKS PASSED
python validate_metrics.py   # must end: ALL METRICS MATCH SKLEARN
```

**Do not skip this.** `preflight.py` runs 16 checks:
- library versions, dataset paths, the anomaly bank and the category exclusion list,
- input normalization and CutMix output statistics,
- model wiring and the global RNG,
- degradations, dual-mode inference, a real backward pass,
- the L_calib loss and the metric estimators.

Every one of them corresponds to a bug that actually happened in this project. It's two
minutes against hours of wasted GPU time. It also warns if your GPU is too small for the
batch size (see step 5).

### 5. Run

```bash
python experiment_a.py   # untrained baseline, ~10 min
python train.py          # TwinGuard, 8 epochs: ~20 min on an A40, ~80 min on a 4 GB laptop GPU
python check_runs.py     # results table from MLflow
```

**GPU with less than ~6 GB?** Set `TWINGUARD_BATCH_SIZE=2` and `TWINGUARD_GRAD_ACCUM=2` first.
Batch 4 needs 4.8 GB, and on Windows an overflow doesn't error. It silently pages into
system RAM and runs 8× slower. The pair keeps the effective batch at 4. Also set
`TWINGUARD_NUM_WORKERS=4` (or 8+ on a server). Full walkthroughs:
[LOCAL_GUIDE.md](LOCAL_GUIDE.md) (own GPU), [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) (shared cloud pod).

Training prints the selected checkpoint's numbers at the end. **Those are the ones to
quote.** Per-epoch lines describe models that were not kept.

Everything after training is evaluation only, a few minutes each:

```bash
python calibrate.py --checkpoint checkpoints/model_3head_best.pth --temp-only   # temperature baselines
python eval_spatial.py --dataset fishyscapes --raw checkpoints/model_3head_best.pth   # edge ECE + UBQ
python check_uncertainty_split.py   # Novelty 7: disagreement vs MC-Dropout under degradations
python eval_road_anomaly.py --with-baseline   # RoadAnomaly21 (needs the optional dataset)
TWINGUARD_NUM_HEADS=1 python train.py && python check_ablation.py   # 1-head vs 3-head table
python server.py                    # live demo, http://127.0.0.1:8000 (build frontend first)
```

### 6. Get a trained checkpoint (optional)

Checkpoints aren't in git (327 MB each, over GitHub's 100 MB limit). To evaluate or run the
demo without training, ask a teammate for these files and put them at the same paths:

| File | What it is |
|---|---|
| `runs/phase2a_coco_run1/model_3head_best.pth` | **the reported TwinGuard model** (copy it to `checkpoints/model_3head_best.pth` too; that's the scripts' default) |
| `runs/phase2a_1head_run1/model_1head_best.pth` | 1-head ablation |
| `runs/phase2b_calib2/model_3head_calib_best.pth` | best L_calib attempt (a finding, not an improvement) |

A checkpoint only loads with `transformers < 5`, because 5.x renamed the SegFormer weight
keys. If you train your own, your numbers will be close to but not identical to
`runs/RESULTS.md`: pasting is random, and on 50 test images about ±0.04 AP is noise.

---

## How it works

```
input image (512×1024, ImageNet-normalized)
        │
        ▼
┌──────────────────────────┐
│  FROZEN SegFormer-B5     │  Cityscapes-finetuned. Never updated.
│  encoder (81.4M params)  │  4 multi-scale feature maps out.
└──────────┬───────────────┘
           │
    ┌──────┴────────┬──────────────┬──────────────┐
    ▼               ▼              ▼              ▼
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ seg head│   │ OOD head │   │ OOD head │   │ OOD head │   4.1M trainable
│19 classes│   │  seed 42 │   │ seed 123 │   │  seed 7  │   total
└─────────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
                   └──── logits ──┴──────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        mean → OOD score              std → uncertainty
                                    (head disagreement)
```

- The encoder is **frozen** — `requires_grad=False` on every parameter, forward under `no_grad()`. Two assertions enforce this at startup.
- The three OOD heads share **zero trainable weights**. Their disagreement is genuine independence, not an artefact of a shared trunk.
- Heads emit **logits**, not probabilities. This keeps the loss numerically stable and makes the planned temperature-scaling ablation possible at all — temperature scaling operates on logits.
- Loss: `L_total = L_seg + α · L_OOD_total` in Phase 2a (`train.py`), with `L_OOD_total` the sum over the three heads as in the paper. Phase 2b (`calibrate.py`) adds `β · L_calib`, a soft-binning ECE surrogate on the pixels around each object's edge. It didn't beat temperature scaling, and why is a reported finding (see Results / `runs/RESULTS.md`).
- **Dual-mode inference** (`model.predict_dual_mode`): continuous mode is one encoder pass plus one head pass. The safety mode reuses the cached encoder features and runs each head 10× with dropout on (MC-Dropout), so the expensive backbone never runs twice.

### The strict dataset rule

Training only ever sees **Cityscapes** (normal street scenes) plus **pasted anomaly objects**. Fishyscapes Lost & Found is evaluation-only and appears in no training path. The 100 evaluation images are further split 50/50: the best epoch is chosen on one half, results are reported on the other. Selecting on the same images you report is model selection on the test set — it inflates every number even when training never touched them.

---

## Anomaly source

The model needs examples of "weird stuff on the road." Set by `config.ANOMALY_SOURCE`:

- **`"carla"`** — the project's canonical source. Objects rendered in the CARLA driving simulator by `generate_anomalies.py`. Requires a running CARLA server.
- **`"coco"`** — real-photo object cutouts from COCO, with every Cityscapes-overlapping category removed. This is the standard outlier exposure used by PEBAL, DenseHybrid and Mask2Anomaly. **All results above use this.**

Both hand CutMix the same `(RGB crop, binary mask)` pair through `data/anomaly_sources.py`, so nothing downstream changes when the flag flips, and the value is logged to MLflow on every run.

> ### ⚠️ The exclusion list is the one thing that must not be wrong
>
> COCO contains cars, people and bicycles — all of which Cityscapes already knows. Pasting one and labelling it "anomaly" would teach the model that **cars are anomalous**, and the whole run would be worthless. `COCO_EXCLUDED_CATEGORIES` in `config.py` removes them. `download_coco_anomalies.py` fails loudly on an unrecognised category name and asserts no excluded category reached the bank; `preflight.py` re-checks the built bank independently. Do not edit that list casually.

When CARLA data is available again, running both with everything else fixed gives a real synthetic-vs-real outlier-exposure ablation.

---

## The metrics, in plain terms

| Metric | What it asks | Good |
|---|---|---|
| **AUROC** | Pick one anomalous pixel and one normal pixel at random — are they ranked correctly? | → 1.0 |
| **AP** | Of the pixels flagged most confidently, how many are genuinely anomalous? | → 1.0 |
| **FPR@95** | To catch 95% of real anomalies, how many false alarms must you accept? | → 0.0 |
| **ECE** | When the model says "90% sure", is it right 90% of the time? | → 0.0 |
| **mIoU** | Did normal segmentation get worse while learning to spot anomalies? | stays flat |

Two things worth knowing before quoting any of these:

**AUROC saturates.** Once the model works, AUROC barely moves — across a full 15-epoch run it varied by 0.012 while AP varied by 0.116. At that point AUROC is mostly noise, so **checkpoint selection uses AP** (`SELECTION_METRIC`), which is also what the Fishyscapes benchmark ranks on. Selecting on AUROC once picked the epoch with the *worst* AP of the entire run.

**A low ECE alone means nothing here.** The anomaly rate is 0.24%, so a model that outputs "0.003 everywhere" is almost perfectly calibrated while being completely useless. ECE is only meaningful next to an AUROC showing the model discriminates at all. Always report them together.

---

## Repository layout

```
config.py                   every path and hyperparameter, with the reasoning inline
utils.py                    device / MLflow / checkpoint helpers
losses.py                   L_seg + α·L_OOD_total, SoftECELoss (L_calib), edge-band helper
metrics.py                  AUROC / AP / FPR@95 / ECE / mIoU (histogram-based),
                            edge bands, UBQ (per-object, local) and the max-F1 threshold

train.py                    Phase 2a: trains TwinGuard (TWINGUARD_NUM_HEADS=1 for the ablation)
calibrate.py                Phase 2b: temperature baselines + L_calib fine-tune + comparison
experiment_a.py             Experiment A: the untrained baseline (--image for one frame)

eval_spatial.py             edge ECE at r=4/8/16, UBQ threshold sweep, bootstrap CIs
eval_road_anomaly.py        RoadAnomaly21 (second benchmark, 10 labelled images)
check_uncertainty_split.py  Novelty 7: head disagreement vs MC-Dropout, clean vs degraded
check_ablation.py           Novelty 5: 1-head vs 3-head vs calibrated, mIoU per class
check_dual_mode.py          latency of both modes + safety-trigger threshold sweep
check_detections.py         boxes the model draws on real frames (check_detections.png)

preflight.py                16 checks; run before every training run
validate_metrics.py         checks every metric against sklearn / brute force

check_runs.py               results from MLflow  (--trend for per-epoch history)
check_collapse.py           per-head class separation on a saved checkpoint
check_data.py               visual + statistical check of the anomaly training data

download_coco_anomalies.py  builds the COCO object bank
generate_anomalies.py       builds the CARLA object bank (needs a CARLA server; see caveats)

server.py                   live demo backend (FastAPI), serves the React build
frontend/                   live demo dashboard (React + Vite), see frontend/README.md

make_upload.py              packages source for upload to a cloud GPU
setup_env.sh                makes env vars survive a pod restart
pod_survey.sh               read-only survey of a shared GPU pod

data/    transforms · cityscapes_dataset · anomaly_sources · cutmix · degradations · fishyscapes_dataset
model/   encoder · seg_head · ood_head · twinguard_model (incl. dual-mode inference)
runs/    RESULTS.md: every local run and evaluation, with its numbers (checkpoints/logs stay local)
PLAN.md  the original planning checklist from main, with a note on what exp_v2 supersedes
```

**Running on a rented GPU?** See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) — it assumes a shared pod and covers not breaking someone else's work on the same machine.

**Running on your own laptop GPU?** See [LOCAL_GUIDE.md](LOCAL_GUIDE.md). The whole pipeline fits in 4 GB of VRAM because the encoder is frozen, as long as it runs at batch 2 with 2-step gradient accumulation (same effective batch of 4). Batch 4 silently spills into system RAM and runs 8× slower. A full Phase 2a run takes ~1.5 h.

**Live demo.** `python server.py`, then open http://127.0.0.1:8000. That's a FastAPI backend plus
a React dashboard (`frontend/`, build once with `npm ci && npm run build`). It shows real
output of the trained checkpoint on held-out Fishyscapes test frames: clearest *and* typical
ones, labelled, with detection boxes against the true outline, dual-mode latency, and
head-disagreement and MC-Dropout maps. Setup in [LOCAL_GUIDE.md](LOCAL_GUIDE.md) §9c. Every
result from the local runs, with its log, is in [runs/RESULTS.md](runs/RESULTS.md).

---

## What changed, and why it mattered

The pipeline previously scored **0.6282 AUROC — below the untrained baseline**. An audit found this was not one bug but several compounding ones. Each item below was measured, not guessed. In rough order of impact.

### 1. The backbone had never seen a street

`nvidia/mit-b5` is an *ImageNet classifier* — 1000 labels, trained on internet photos of everyday objects. It has never looked at a road. That was exactly the criticism that had been correctly made of the previous `mit-b2`, so swapping b2 → b5 would have measured backbone **size**, not road-adaptation.

The road-adapted encoder was inside the checkpoint Experiment A already used. Identical architecture (`hidden_sizes [64,128,320,512]`, `depths [3,6,40,3]`), but Cityscapes-trained. Experiment B now sits on **the same features that give the baseline its 0.83**, which makes the comparison fair: the question becomes "do three trained heads beat max-softmax on identical features?"

### 2. Images were fed in the wrong number range

*In plain terms:* every pretrained vision model expects its input shifted and scaled a particular way — like a camera expecting a specific exposure. The code was handing over raw pixel values. Because the encoder is frozen, it could not adapt; it simply produced poor descriptions of every image.

Worse, Experiment A normalized correctly (via the HuggingFace processor) while Experiment B did not, so the two experiments were never being shown the same thing.

Measured in isolation — same model, same images, normalization the only variable: **MSP AUROC 0.8812 → 0.7329**.

### 3. Training objects were the wrong size

Measured across all 188 annotated Fishyscapes objects, longest side as a fraction of the image short side: **median 0.043**, p75 0.072.

`CUTMIX_SCALE` was **0.30–0.55**. Not approximately wrong — *zero overlap*. Every training object was larger than about 96% of real test objects. The model had never once seen an anomaly the size of the ones it was graded on.

Now 0.012–0.20, sampled log-uniformly, 1–3 objects per image, producing a 0.24% anomaly pixel rate against the real 0.28%.

This also explains the debugging history: raising the scale from 0.15–0.35 to 0.30–0.55 is the change that took AUROC 0.6282 → 0.6193. Three successive "give the heads more anomaly pixels" fixes all moved training *further* from the test distribution.

### 4. Two-thirds of objects were pasted where anomalies cannot be

Under uniform placement only **33%** of paste centres landed on road — the rest floated in sky, sat inside buildings, or perched on car roofs. Real debris is on the drivable surface. Placement is now restricted to road/sidewalk: **96%**.

### 5. The loss form made recovery impossible

The heads applied `sigmoid`, and the loss then called `binary_cross_entropy` on the result. *In plain terms:* once the model starts answering "definitely not an anomaly" to everything, this particular arrangement leaves it no gradient to climb back out — it gets stuck. That is the measured collapse signature (`mean ~0.001, std ~0.01`, unmoved by any data change).

Heads now emit logits and the loss uses `binary_cross_entropy_with_logits` with a fixed positive-class weight. The old weight was recomputed per batch and swung by two orders of magnitude between steps.

### 6. The best model was chosen using the test set

The best epoch was picked by score on all 100 Fishyscapes images. Fishyscapes never entered training — but using it for *selection* still biases every reported number. Now a deterministic 50/50 split: selection reads the val half, the test half is reported.

### 7–11. The rest

- **Segmentation labels are set to `ignore` under pasted objects**, instead of training the head that a pasted bench is "road".
- **Pasted objects are photometrically harmonized and edge-feathered**, so the heads cannot learn "spot the paste seam" as a shortcut for "unfamiliar object". Mask fragments under 32px are dropped — COCO's multi-polygon annotations otherwise leave 1-pixel specks labelled as anomalies, which nothing can detect and which are pure label noise.
- **AP and mIoU are computed.** Both are named as core metrics in the project plan; neither existed. AP is what Fishyscapes actually ranks on.
- **Metrics are histogram-based and validated.** Evaluation touches 170.6M valid pixels, and the old code sorted all of them four times per epoch. `metrics.py` accumulates in a single pass; `validate_metrics.py` checks every estimator against sklearn including a deliberately collapsed-model case.
- **The collapse diagnostic measures class separation, not output spread.** It previously tested global output std < 0.05 — which fires on a *healthy* model, because at a 0.24% positive rate nearly every pixel is a correctly-near-zero negative. It once flagged a model scoring 0.99 AUROC as collapsed. It now reports mean score on anomalous pixels minus mean on normal ones, and warns only when that is under 0.01 **and** AUROC is under 0.70.
- **Head disagreement is computed and its own AUROC logged.** The project's core uncertainty claim previously had nothing reading it.
- **OOD head construction no longer reseeds the global RNG**, which had been silently overriding `GLOBAL_SEED` for data shuffling and dropout.

---

## Known caveats

State these before a reviewer finds them.

**COCO outlier exposure is well-matched to Lost & Found.** Both are real photographs of everyday objects on roads. That similarity is part of why the results are strong. It is the standard method in this literature (PEBAL, DenseHybrid, Mask2Anomaly all do it), so it is defensible — but it should be said out loud, and it is exactly why the CARLA-vs-COCO ablation is worth running.

**Not a like-for-like comparison with published methods.** See the note under Results. TwinGuard is in the same range as PEBAL / RbA, but it isn't evaluated on their exact images and protocol. Don't claim it beats them.

**Experiment A's FPR@95 is 1.0000, and that's real.** The baseline is fully confident that more than 5% of anomaly pixels are a known class, so it can't reach 95% recall without flagging everything. An earlier project report of 0.6802 came from `roc_curve(drop_intermediate=True)`, which sparsifies the curve. Don't quote it.

**Small test set, single seed.** 50 test images and one seed per configuration. Two epochs of one run with nearly equal validation AP differed by 0.04 test AP, so don't over-read differences smaller than that. Validation AP peaked at the final epoch (8 of 8), so a slightly longer run might gain a little.

**Negative results, reported as findings (details in `runs/RESULTS.md`):**
- **L_calib (Phase 2b)** doesn't beat temperature scaling at object edges (edge ECE 0.189 vs raw 0.195, CI includes 0). Calibration learned on pasted training objects doesn't transfer to real ones: the model is ~99% sure of pastes and 50–69% sure of real objects.
- **Head disagreement vs MC-Dropout** don't separate "degraded camera" from "novel object". Both rise together under noise and blur.
- **The safety trigger** doesn't yet meet <5% false / >90% true triggers (best 20% / 47.5%). Most normal frames contain some patch scoring above 0.3.
- **RoadAnomaly21** (10 labelled images): AUROC 0.72, below the baseline's 0.87. Large close-up anomalies fail, most likely because pasted training objects are sized for Lost & Found's small debris.
- **Robustness:** camera noise drops AP 0.75 → 0.55 and motion blur → 0.57; fog barely matters.

**CARLA frames are not used.** In the 45 frames from `generate_anomalies.py`, the RGB and mask were captured asynchronously, so 23 of 45 masks don't line up with the object. Pooling them with COCO (the `main` branch) cost 0.17 AP. The script needs synchronous capture before CARLA data is usable.

---

## Status and what's next

Done on `exp_v2`, with results in `runs/RESULTS.md`:
- Phase 2a training, and the 1-head ablation (Novelty 5).
- The Experiment A baseline.
- Phase 2b: plain-NLL temperature baselines, and edge-band L_calib with an anchor.
- Edge ECE, and per-object UBQ with a threshold sweep (Novelty 4/6).
- The between- vs within-head uncertainty test (Novelty 7).
- The dual-mode latency and trigger sweep (Novelty 2).
- RoadAnomaly21.
- The live demo.

Still open:
1. **CARLA:** fix synchronous capture in `generate_anomalies.py` (needs the CARLA machine), re-render, then run CARLA-vs-COCO. This also unblocks Novelty 8 (frame-to-frame flicker needs a recorded sequence) and CARLA playback in the demo.
2. **Write-up:** including the DUDES comparison. Their paper doesn't report on Fishyscapes, so it has to be a qualitative comparison, stated as such.
3. **Possible improvements, not yet tried:**
   - a wider pasted-object size range, for the large RoadAnomaly21 objects;
   - a better trigger statistic for dual-mode;
   - calibrating on held-out anomalies the detector never trained on.

---

## Contributing

Before opening a PR:

```bash
python preflight.py           # must end: ALL CHECKS PASSED
python validate_metrics.py    # must end: ALL METRICS MATCH SKLEARN
```

Never commit checkpoints, datasets, `mlflow.db` or logs — `.gitignore` covers them, but a single `.pth` is 327MB and GitHub's per-file limit is 100MB. Once a large blob is in history it stays there.

If you change anything about how scores are produced or measured, add a check to `preflight.py` or a case to `validate_metrics.py`. Nearly every bug in the audit above was one that a two-minute check would have caught before a multi-hour GPU run.

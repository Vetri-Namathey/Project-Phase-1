# TwinGuard

**Out-of-distribution (OOD) detection for autonomous-vehicle perception.**

A normal segmentation model knows 19 things — road, car, person, traffic light, and so on. Show it a mattress that fell off a truck and it has no "mattress" option, so it picks the nearest label it does have and reports high confidence. TwinGuard adds a second opinion: it flags the pixels it does not recognise, and attaches an uncertainty estimate that is meant to be *trustworthy* rather than merely confident-looking.

---

## Results

Evaluated on **Fishyscapes Lost & Found**, on a held-out 50-image test half that is never used for training or checkpoint selection.

| Method | AUROC ↑ | AP ↑ | FPR@95 ↓ | Reports ECE? |
|---|---|---|---|---|
| Experiment A — off-the-shelf SegFormer-B5 + MSP (no training) | 0.8372 | 0.0114 | 1.0000 | — |
| DenseHybrid (ECCV 2022) | — | 0.4390 | 0.0620 | no |
| PEBAL (ECCV 2022) | 0.9896 | 0.5881 | 0.0477 | no |
| RbA (ICCV 2023) | 0.9862 | 0.7081 | 0.0630 | no |
| **TwinGuard (this repo)** | **0.9905** | **0.7917** | **0.0168** | **yes — 0.0003** |

Supporting numbers for the same checkpoint:

| | |
|---|---|
| **mIoU** on Cityscapes val | **0.7696** — segmentation is not degraded by the OOD heads |
| **Head-disagreement AUROC** | **0.9842** — disagreement between the three heads, scored as a detector *on its own* |
| Selected epoch | 8 of 8, by best validation AP |
| Training time | 21 min (8 epochs × 160s) on one A40 |

Published figures come from each paper's own results table. TwinGuard's row is the selected checkpoint's test-half score, reproducible with the commands below.

**Why the disagreement number matters.** The architecture's premise is that three independently-initialised heads disagreeing is a real signal of "the model does not know." Measured alone, with no other information, that disagreement detects anomalies at 0.9842 AUROC. It is evidence, not an assumption.

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

Four directories are needed. Put them under one root:

```
<DATA_ROOT>/
├── leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/<city>/   Cityscapes images
├── gtFine_trainvaltest/gtFine/{train,val,test}/<city>/             Cityscapes labels
├── fishyscapes_lostandfound/                                       100 OOD label PNGs
└── leftImg8bit/leftImg8bit/{train,test}/<city>/                    Lost&Found photos
```

- **Cityscapes** — free account at <https://www.cityscapes-dataset.com>; download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
- **Fishyscapes Lost & Found** — labels from Zenodo, images from the Lost&Found dataset. They come from different sources and do not share a folder tree; the code pairs them by filename.

Point the code at it:

```bash
export TWINGUARD_DATA_ROOT=/path/to/your/data
```

Each of the four paths can also be overridden individually by an environment variable of the same name as its `config.py` constant.

### 3. Build the anomaly bank

```bash
python download_coco_anomalies.py
```

~1GB download, a few minutes. See [Anomaly source](#anomaly-source) for what this is and why.

### 4. Verify before you train

```bash
python preflight.py          # must end: ALL CHECKS PASSED
python validate_metrics.py   # must end: ALL METRICS MATCH SKLEARN
```

**Do not skip this.** `preflight.py` runs 12 checks covering library versions, dataset paths, the anomaly bank, the category exclusion list, input normalization, CutMix output statistics, model wiring, a real backward pass, and the metric estimators. Every one of them corresponds to a bug that actually happened in this project. It is two minutes against hours of wasted GPU time.

### 5. Run

```bash
python experiment_a.py   # baseline, ~10 min -- run this FIRST
python train.py          # TwinGuard, ~21 min on an A40
python check_runs.py     # results table
```

Run the baseline before training. Its published 0.8304 was measured over all 100 images; you need the **test-half** number for a fair comparison against Experiment B.

Training prints the selected checkpoint's numbers at the end. **Those are the ones to quote** — per-epoch lines describe models that were not kept.

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
- Loss: `L_total = L_seg + α · L_OOD`. `L_calib` is Phase 2b and is deliberately not implemented yet.

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
losses.py                   L_seg + α · L_OOD   (L_calib is Phase 2b, not here)
metrics.py                  AUROC / AP / FPR@95 / ECE / mIoU, histogram-based

train.py                    Experiment B — trains TwinGuard
experiment_a.py             Experiment A — the untrained baseline (--image for one frame)

preflight.py                12 checks; run before every training run
validate_metrics.py         checks every metric estimator against sklearn

check_runs.py               results from MLflow  (--trend for per-epoch history)
check_collapse.py           per-head class separation on a saved checkpoint
check_data.py               visual + statistical check of the anomaly training data

download_coco_anomalies.py  builds the COCO object bank
generate_anomalies.py       builds the CARLA object bank (needs a CARLA server)

make_upload.py              packages source for upload to a cloud GPU
setup_env.sh                makes env vars survive a pod restart
pod_survey.sh               read-only survey of a shared GPU pod

data/    transforms · cityscapes_dataset · anomaly_sources · cutmix · fishyscapes_dataset
model/   encoder · seg_head · ood_head · twinguard_model
```

**Running on a rented GPU?** See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) — it assumes a shared pod and covers not breaking someone else's work on the same machine.

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

**Experiment A's FPR@95 needs re-confirming.** It reads **1.0000** under the current sklearn-validated metric code, where an earlier run of the project reported 0.6802. The old number came from `roc_curve(drop_intermediate=True)`, which sparsifies the curve. Resolve this before quoting either figure.

**Convergence was not formally demonstrated.** Validation AP peaked at the final epoch (8 of 8), so the run was still improving when it stopped. A 15-epoch run reached a best test AP of 0.7958 versus 0.7917 here — a 0.004 difference, and epochs 9–15 of that run *averaged* 0.7535, below this result. Practically converged, but `EPOCHS = 12` would settle it for ~11 more minutes.

**Single seed.** All results are one seed. Small differences between configurations should not be over-read until confirmed on a second.

---

## What's next

1. **Phase 2b — `L_calib`.** A differentiable soft-binning ECE surrogate trained jointly with OOD detection, staged so the two objectives do not fight. This is the actual research contribution and it is now unblocked.
2. **Temperature-scaling ablation.** The obvious reviewer question: "why not just apply the simple post-hoc fix?" Now implementable, because the heads emit logits.
3. **CARLA-vs-COCO ablation.** Same config, `ANOMALY_SOURCE` flipped.
4. **UBQ** — the spatial uncertainty metric, validated against exact simulator ground truth.
5. **Dual-mode inference** — measuring the trigger rate that the latency claim depends on.

---

## Contributing

Before opening a PR:

```bash
python preflight.py           # must end: ALL CHECKS PASSED
python validate_metrics.py    # must end: ALL METRICS MATCH SKLEARN
```

Never commit checkpoints, datasets, `mlflow.db` or logs — `.gitignore` covers them, but a single `.pth` is 327MB and GitHub's per-file limit is 100MB. Once a large blob is in history it stays there.

If you change anything about how scores are produced or measured, add a check to `preflight.py` or a case to `validate_metrics.py`. Nearly every bug in the audit above was one that a two-minute check would have caught before a multi-hour GPU run.

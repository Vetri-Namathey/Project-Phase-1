# Local GPU guide — TwinGuard on your own laptop

Written for: running the whole pipeline (Phase 2a training, Phase 2b L_calib, evaluation,
demo) on a Windows machine with a small NVIDIA GPU, without RunPod.

Every number here was **measured on one machine**: an RTX 3050 Ti laptop GPU (4 GB) with
16 GB RAM. On a different GPU the timings will differ. The memory rule in Section 0 applies
to any GPU under ~6 GB, and `preflight.py` tells you which case you're in. Commands are
PowerShell. Paths like `D:\Academics (D)\...` are that machine's; use your own clone's folder.

**Before starting:** you need the datasets (README, "2. Get the datasets"). None are in the
repository, so ask a teammate for them. You also need a checkpoint (README, "6. Get a trained
checkpoint") if you want to skip training.

---

## The short version

```
conda activate twinguard                 # the env from Section 1
cd <your clone>\TwinGuard_CARLA
$env:TWINGUARD_DATA_ROOT = "<folder holding the datasets>"   # README, "2. Get the datasets"
$env:TWINGUARD_NUM_WORKERS = "4"; $env:TWINGUARD_BATCH_SIZE = "2"; $env:TWINGUARD_GRAD_ACCUM = "2"
python preflight.py                                   # must end in ALL CHECKS PASSED
python -u train.py > runs\train.log 2>&1              # Phase 2a, ~1.5 h
python calibrate.py --checkpoint checkpoints\model_3head_best.pth --temp-only   # baselines, ~10 min
python calibrate.py --checkpoint checkpoints\model_3head_best.pth --out checkpoints\model_3head_calib_best.pth   # Phase 2b, ~1 h
python eval_spatial.py --dataset fishyscapes --raw checkpoints\model_3head_best.pth --calib checkpoints\model_3head_calib_best.pth   # the proof, ~15 min
```

The rest of this file explains each line: what it does, what healthy output looks like, and
how to tell whether the result is good.

---

## 0. What your laptop can and cannot do

| Measured on the RTX 3050 Ti (full training step, 512×1024, bf16) | batch 2 | batch 4 |
|---|---|---|
| GPU memory PyTorch reserves | **2.61 GB**, fits | **4.81 GB**, doesn't fit in 4 GB |
| Time per image | **0.18 s** | 1.43 s (8× slower) |

**Always run at batch 2 with 2-step gradient accumulation on this laptop:**

```
$env:TWINGUARD_BATCH_SIZE = "2"; $env:TWINGUARD_GRAD_ACCUM = "2"
```

Why: when a batch doesn't fit, Windows does **not** give an out-of-memory error. It quietly
pages GPU memory into normal RAM, and training crawls. A first attempt at batch 4 ran at
3 s/step and was heading for 5+ hours before it was stopped. Gradient accumulation adds up
two batches of 2 before each weight update, so the model learns exactly as it would from
batches of 4: the same recipe as the reported run, just split in two. `preflight.py` warns
if the env vars are missing.

A GPU with 8 GB or more can use the defaults (batch 4, no accumulation). Leave those two
variables unset.

| Timing at batch 2 × 2 | Value |
|---|---|
| DataLoader, 4 workers | faster than the GPU needs, so the GPU is the bottleneck (what you want) |
| One epoch (1488 steps + evaluation) | ~10–12 min |
| Phase 2a, 8 epochs | ~1.5 h |
| Phase 2b fine-tune, 5 epochs | ~1 h |

It fits at all because the big SegFormer-B5 encoder is **frozen**: it runs forward only and
stores nothing for backprop.

---

## 1. One-time setup

Create a separate environment for this project. Don't install into an env another
project uses, because upgrading a package there can break it.

```
conda create -n twinguard python=3.10 -y
conda activate twinguard
# PyTorch with CUDA: take the exact command for your CUDA version from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "transformers>=4.46.3,<4.51"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU's name.

The reference machine used: Python 3.9, torch 2.5.1 (CUDA), transformers 4.49.0,
numpy 1.23.5, mlflow 3.1.4 (there, the env was a clone of an existing GPU env plus MLflow).

> **transformers must stay below 4.51 when torch is older than 2.6.** Newer versions refuse
> to load the SegFormer weights (security check CVE-2025-32434), and transformers 5.x
> renames the model keys, so saved checkpoints stop loading. `preflight.py` checks this. If
> anything upgrades it: `pip install "transformers>=4.46.3,<4.51"`.

If `conda activate` doesn't work in your shell, call the env's Python directly by its full
path (`conda env list` shows where it is), e.g.:

```
& "<conda folder>\envs\twinguard\python.exe" preflight.py
```

---

## 2. Before every long run: the laptop checklist

A laptop kills long runs in ways a pod does not. Two minutes here saves a lost hour.

1. **Plug in the charger.** On battery, the GPU drops to a low power state and runs 2–3× slower.
2. **Stop the laptop sleeping.** Settings → System → Power → *Screen and sleep* → "When plugged
   in, put my device to sleep after" → **Never**. Sleep kills the run silently; the log just stops.
   Or, only for the duration of the run:
   ```
   powercfg /change standby-timeout-ac 0
   ```
   (Afterwards: `powercfg /change standby-timeout-ac 30` restores 30 minutes.)
3. **Don't close the lid** unless lid-close is set to "Do nothing" in the same power settings.
4. **Close other GPU users**: games, video editors, and browser tabs playing video.
   Check that nothing else holds GPU memory:
   ```
   nvidia-smi
   ```
   The `Memory-Usage` line should be under ~500 MiB before you start. The desktop itself uses
   a little; that's fine.
5. **Windows Update** can reboot overnight. Pause updates for a day if running overnight.

---

## 3. Preflight — always run first

```
conda activate twinguard
cd "D:\Academics (D)\SEM-7\PROJECTS\FinalYearProject\TwinGuard_CARLA"
python preflight.py
python validate_metrics.py
```

Expect:

- `preflight.py` → every line `[PASS]`, ending in **`ALL CHECKS PASSED -- ready to train.`**
  (16 checks, including `L_calib loss (Phase 2b)` and `dual-mode inference`).
- `validate_metrics.py` → ends in **`ALL METRICS MATCH SKLEARN`**.

| If you see | Meaning / fix |
|---|---|
| `dataset root(s) not found` | The datasets moved. Paths are at the top of `config.py`, or set `$env:TWINGUARD_DATA_ROOT` |
| `no cutouts found` | `data/coco_objects/` is missing: `python download_coco_anomalies.py` |
| `Cityscapes-known categories in bank` | A car/person leaked into the COCO bank. Delete `data/coco_objects` and re-download |
| `transformers ... CVE` or `SegformerModel requires the PyTorch library` | transformers got upgraded; see Section 1 |
| `compute device` WARN | You are not on the `twinguard` env, or the GPU is busy |

---

## 4. Phase 2a — train the detector (≈1.5 h)

This trains the segmentation head and the 3 OOD heads on Cityscapes with COCO objects pasted
in (CutMix). The encoder stays frozen. The anomaly source is set by `ANOMALY_SOURCE = "coco"`
in `config.py`.

```
$env:TWINGUARD_NUM_WORKERS = "4"; $env:TWINGUARD_BATCH_SIZE = "2"; $env:TWINGUARD_GRAD_ACCUM = "2"
New-Item -ItemType Directory -Force runs | Out-Null
python -u train.py > runs\train.log 2>&1
```

- `TWINGUARD_BATCH_SIZE=2` + `TWINGUARD_GRAD_ACCUM=2`: see Section 0. Without them it's 8× slower.

- `TWINGUARD_NUM_WORKERS=4`: 4 CPU processes prepare images while the GPU trains. With 0 the
  GPU sits waiting. Don't go above ~6 on 16 GB RAM; each worker holds its own copy of the data
  pipeline.
- `-u` + `> runs\train.log` writes the log live, so you can watch it from another window.
- The env var only lasts for that PowerShell window. Set it again in a new window.

**Watch it** from a second PowerShell window:

```
Get-Content runs\train.log -Wait -Tail 20
```

`Ctrl+C` there stops *watching*, not training. To stop training, press `Ctrl+C` in the
window running `train.py`.

### What healthy output looks like

At startup:

```
pre-flight: encoder frozen, heads independent, encoder=nvidia/segformer-b5-finetuned-cityscapes-1024-1024
anomaly source: coco (3000 objects)
dataloader: 1488 steps/epoch, num_workers=4, batch 2 x 2 accumulation = effective 4
fishyscapes: 50 val (selection) / 50 test (reported)
```

Then every 100 steps, a `loss=` line that trends down, and every epoch:

```
epoch 3/8 done in 470.2s -- loss=0.1234 | VAL ap=0.7xxx auroc=0.99xx | TEST auroc=0.99xx ap=0.7xxx ece=0.000x fpr95=0.0xxx | mIoU=0.76xx
  head separation (mean score on anomalous - on normal): h0=+0.4xxx  h1=+0.4xxx  h2=+0.4xxx
  -> new best VAL ap 0.7xxx (test ap 0.7xxx), checkpoint saved
```

How to read it:

- **VAL** = the 50 Fishyscapes images used to *pick* the best epoch. **TEST** = the other 50,
  the numbers you *report*. They are never mixed.
- **head separation**: positive and growing is healthy. A `WARNING ... collapse` line means
  the heads can't tell anomaly from road. Stop the run; more epochs won't fix it.
- **mIoU** should rise and then hold around 0.76–0.77. If it falls while AP rises, the OOD heads
  are hurting segmentation.
- **TEST numbers only print on epochs where VAL improved** (plus the last epoch). That's on
  purpose.

### At the end

```
training done. selected epoch N (best val ap=...)
  SELECTED CHECKPOINT on the test half:
    AUROC=...  AP=...  FPR@95=...  ECE=...
```

**Quote these numbers.** They describe the `.pth` file that was saved, not the last epoch.

### Is it a good run?

Target: reproduce the earlier COCO-only run (run 3, whose checkpoint was lost).

| Metric (test half) | Good | Earlier COCO-only run 3 | Friend's CARLA+COCO |
|---|---|---|---|
| AUROC | ≥ 0.99 | 0.9905 | 0.9920 |
| **AP** (the main one) | **≥ 0.75** | 0.7917 | 0.6218 |
| FPR@95 | ≤ 0.03 | 0.0168 | 0.0287 |
| mIoU | ≥ 0.75 | 0.7696 | — |

Runs are not bit-identical (the random pasting differs), so AP within about ±0.03 of 0.79
counts as a reproduction.

Output: `checkpoints\model_3head_best.pth` (~330 MB). This is a different file from
`model_3head_best.pth` at the repo root, which is the friend's CARLA+COCO checkpoint. Leave
that one alone; it is the comparison point.

---

## 5. Save every run you want to keep

`train.py` and `calibrate.py` **overwrite** their output checkpoint on the next run. Copy a good
run into its own folder straight away:

```
$name = "phase2a_coco_run1"
New-Item -ItemType Directory -Force runs\$name | Out-Null
Copy-Item checkpoints\model_3head_best.pth runs\$name\
Copy-Item runs\train.log runs\$name\
```

Then point later commands at the saved copy, e.g.
`--checkpoint runs\phase2a_coco_run1\model_3head_best.pth`.

`runs\` can hold GBs of checkpoints. `*.pth` and `*.log` are already in `.gitignore`, so
they will never be committed by accident. **Never commit a checkpoint**: it's over GitHub's
100 MB limit and stays in the history for good.

See **`runs\RESULTS.md`** for the log of every run done so far and which checkpoint is the
current best.

---

## 6. Baselines — temperature scaling (≈10 min)

Before L_calib can claim anything, it needs something to beat. Temperature scaling divides
every score by one number T, fitted on the val half:

```
python calibrate.py --checkpoint checkpoints\model_3head_best.pth --temp-only
```

Expected output:

```
fitting temperature scaling baselines (plain NLL, Fishyscapes val half)...
  whole  T=1.2xxx  NLL ... -> ...
  band   T=4.xxxx  NLL ... -> ...
model          AUROC      AP  FPR@95      ECE   band-ECE r=8
raw           ...
temp(whole)   ...
temp(band)    ...
```

What the columns mean in plain terms:

- **ECE** (whole image): how far the model's confidence is from reality, averaged over *every*
  pixel. It's tiny for any model, because 99.7% of pixels are obvious road and sky.
- **band-ECE r=8**: the same, but only for pixels within 8 px of an anomaly's edge. This is
  where the model is actually wrong about its own confidence, and it's the project's
  Novelty-6 claim.
- **temp(whole)**: T fitted on all pixels. **temp(band)**: T fitted on edge pixels only, the
  fair opponent for an edge claim.

Reference numbers (friend's CARLA+COCO checkpoint, measured 2026-09-26):

| | ECE | band-ECE r=8 |
|---|---|---|
| raw | 0.00044 | 0.2274 |
| temp(whole), T=1.22 | 0.00013 | 0.2096 |
| temp(band), T=4.37 | **0.0422** | **0.0938** |

The key fact: **one temperature cannot fix both.** Tuned for the edges, it makes the
whole-image error 100× worse. That gap is exactly what L_calib has to close.

> The old "T=1.7142" and "temp-scaled band-ECE 0.1845" numbers in `PLAN.md` came from a fit
> against the training loss (pos_weight=20). That fit is biased; ignore those numbers.

---

## 7. Phase 2b — L_calib fine-tune (≈1 h)

This continues training the Phase 2a checkpoint with a calibration loss added:

L_total = L_seg + α·L_OOD + **β·L_calib**

L_calib is a smooth version of ECE, computed on the **edge band** of each pasted object
(`CALIB_LOSS_REGION = "band"`). 30% of training images get fog, noise or motion blur
(`CALIB_DEGRADATION_PROB = 0.3`), so the model is penalised for staying confident on bad
inputs, as the paper describes.

```
$env:TWINGUARD_NUM_WORKERS = "4"; $env:TWINGUARD_BATCH_SIZE = "2"; $env:TWINGUARD_GRAD_ACCUM = "2"
python calibrate.py --checkpoint checkpoints\model_3head_best.pth --out checkpoints\model_3head_calib_best.pth
```

(Same three env vars as Phase 2a. This trains too, so it has the same memory limit.)

Startup prints the starting point and the bar it has to clear:

```
base checkpoint val: AUROC=... AP=... ECE=... band-ECE=...
an epoch is saved only if val band-ECE < 0.xxxxx, with AUROC drop <= 0.03 and AP drop <= 0.03
```

Each epoch:

```
calib epoch 2/5 done -- val AUROC=0.99xx (-0.00xx) AP=0.7xxx (-0.0xxx) ECE=0.000xx band-ECE=0.1xxx avg L_calib=... degraded=30%
  -> new best val band-ECE 0.1xxx, saved checkpoints\model_3head_calib_best.pth
```

Three possible endings, all of them legitimate results:

| Ending | Meaning |
|---|---|
| `L_calib fine-tune done. selected epoch N` + comparison table | Improved on the starting checkpoint. Go to Section 8. |
| `no epoch beat the base checkpoint's val band-ECE ... nothing saved` | L_calib didn't help with these settings. Not a crash. Tune (below). |
| `STOPPING: detection fell past its limit` | Calibration was costing too much detection (AP or AUROC fell > 0.03). Lower β or the learning rate. |

### The knobs (all in `config.py`, `Phase 2b` block)

| Setting | Default | Raise it when | Lower it when |
|---|---|---|---|
| `BETA_CALIB` | 1.0 | band-ECE barely moves | AP drops / STOPPING fires |
| `CALIB_LEARNING_RATE` | 1e-5 | nothing changes at all | results jump around between epochs |
| `CALIB_EPOCHS` | 5 | still improving at the last epoch | — |
| `CALIB_BAND_RADIUS_PX` | 8 | — | keep at 8, the headline metric uses r=8 |
| `CALIB_DEGRADATION_PROB` | 0.3 | — | AP drops even at low β |
| `CALIB_ANCHOR_WEIGHT` | 100 | — | 0 turns it off (plain L_calib) |

**Why the anchor exists:** without it, L_calib made real edges *worse* (val edge ECE
0.153 → 0.29). Pasted objects and real ones are miscalibrated in opposite directions at
the edge, and the anchor stops L_calib dragging the average edge score the wrong way. The
full history is in `runs\RESULTS.md`.

**The bigger lever is Phase 2a, not these knobs.** If the model is ~100% sure of every pasted
object, L_calib has nothing realistic to learn from. `CUTMIX_HARMONIZE_STRENGTH` and
`CUTMIX_OBJECT_BLUR_SIGMA` (config.py, CutMix section) make pastes blend in more. Changing
them means retraining Phase 2a.

Change **one** knob per run and write down what you changed. Otherwise you can't tell which
change did what.

### Is it a good result?

On the **test** half, compared with `temp(band)` from the same checkpoint:

1. **band-ECE r=8 lower than temp(band)'s**, with the bootstrap CI excluding 0 (Section 8).
2. **whole-image ECE stays small** (≤ ~0.001), which temp(band) cannot manage (0.042).
3. **AP within 0.03** of the Phase 2a checkpoint.

All three together are the Novelty-6 claim. Point 1 alone is not enough, because temperature
scaling already gets band-ECE down if you ignore the rest of the image.

---

## 8. The proof — eval_spatial (≈15 min)

```
python eval_spatial.py --dataset fishyscapes --raw checkpoints\model_3head_best.pth --calib checkpoints\model_3head_calib_best.pth
```

(Add `--temp-whole 1.xxxx --temp-band 4.xxxx` with the values from Section 6 to skip re-fitting.)

It prints each model at band widths r = 4, 8 and 16 px, then **paired bootstrap** confidence
intervals:

```
paired bootstrap, 1000 resamples: band-ECE(L_calib) - band-ECE(temp(band))
  r=8  diff=-0.0xxx  95% CI [-0.0xxx, -0.0xxx]  (CI excludes 0)
```

- **diff negative + "CI excludes 0"** → L_calib is better, and it isn't luck from which 50
  images happened to be in the test half. This is the line for the report.
- **"CI includes 0"** → no measurable difference.
- **diff positive** → L_calib is worse.

Each model block also prints a **UBQ table**: how tightly the flagged region hugs each real
object's outline. It uses a threshold fitted on the val half (best F1), then a sweep over
0.1–0.9:

| Column | Plain meaning | Better when |
|---|---|---|
| objects found | real objects with anything flagged near them (within 32 px) | higher |
| spill px | how far the flagged area bleeds past the object's edge | lower |
| miss px | how much of the object is left unflagged | lower |
| bnd-F1 | share of the outline matching within 4 px (0–1) | higher |
| far-FP | share of flagged pixels nowhere near a real object | lower |

Spill, miss and bnd-F1 are averaged over *found* objects only, so a missed object lowers
"objects found" and nothing else. Run 1 at the fitted threshold: 40/85 found, spill 2.5 px,
miss 6.7 px, boundary F1 0.57.

---

## 9. Pictures and the dual-mode check

```
python check_detections.py --checkpoint checkpoints\model_3head_best.pth --n 6 --threshold 0.30
python check_data.py --n 6
python check_dual_mode.py --checkpoint checkpoints\model_3head_best.pth --n-images 40
```

| Script | Writes | Shows |
|---|---|---|
| `check_detections.py` | `check_detections.png` | Real Fishyscapes photos. Green boxes are ground truth, red boxes are what the model flags. The `on-obj` column counts red boxes that actually hit the object. |
| `check_data.py` | `check_data_cutmix.png` | What the training images look like with pasted objects |
| `check_dual_mode.py` | text | Latency of continuous vs. safety mode on *your* GPU, and the lowest `SAFETY_TRIGGER_THRESHOLD` meeting "<5% normal frames trigger, >90% anomalous frames trigger" |

The images open directly from the project folder. They match `check_*.png` in
`.gitignore`, so they won't get committed.

Laptop latency numbers are **not** the paper's real-time numbers. A 3050 Ti is several times
slower than an automotive GPU. What matters is the *ratio* between the two modes, not the
absolute milliseconds.

---

## 9a. The 1-head ablation (Novelty 5)

To train the single-head comparison model, use the same command as Section 4 with one
more env var:

```
$env:TWINGUARD_NUM_HEADS = "1"
python -u train.py > runs\train_1head.log 2>&1
```

It writes `checkpoints\model_1head_best.pth` (not the 3-head file), takes ~8 min/epoch
(~1.1 h), and its "head-disagreement AUROC" prints 0.5000. That's expected: one head has
nothing to disagree with. Remove the variable (`Remove-Item Env:TWINGUARD_NUM_HEADS`) or
open a new window before training the 3-head model again.

Then build the comparison table for all three models (~10 min, evaluation only):

```
python check_ablation.py
```

It prints detection, the two uncertainty signals and mIoU on all 500 Cityscapes val
images, plus per-class IoU with any class that moves by more than 0.02 marked.

---

## 9b. Baseline, uncertainty split, second benchmark

All three are evaluation only (no training), a few minutes each, on the saved checkpoint:

```
python experiment_a.py
python check_uncertainty_split.py --checkpoint runs\phase2a_coco_run1\model_3head_best.pth
python eval_road_anomaly.py --checkpoint runs\phase2a_coco_run1\model_3head_best.pth --with-baseline
```

| Script | Answers | Run-1 result |
|---|---|---|
| `experiment_a.py` | Does TwinGuard beat the untrained off-the-shelf model on the same 50 test images? | Yes: AP 0.750 vs 0.011 |
| `check_uncertainty_split.py` | Novelty 7: does "heads disagree" track unknown objects while "one head wobbles" tracks bad camera input? | No, both rise together (report as a finding). It also shows noise is the main weakness (AP 0.75 → 0.55) |
| `eval_road_anomaly.py` | Does it work on a different real dataset (RoadAnomaly21)? | Worse than the baseline (AUROC 0.72 vs 0.87). Large close-up anomalies are the problem |

`eval_road_anomaly.py` needs the RoadAnomaly21 archive unpacked at
`<data root>\road_anomaly21\dataset_AnomalyTrack` (52.6 MB, from
`zenodo.org/records/5270237`; md5 `231bf79ed58924bcd33d9cbe22e61076`). Only its 10 validation
images have labels, so treat it as a sanity check, not a benchmark score.

`check_uncertainty_split.py` runs in full fp32 and pushes the GPU to ~86 °C for ~3 minutes.
That's normal, but don't queue several long GPU jobs back to back without a pause.

---

## 9c. The live demo (Phase 3)

A FastAPI backend (`server.py`) and a React dashboard (`frontend/`), built on the team's
original version.

**One-time setup** (Node.js 20.19+; this laptop has 22.12):

```
cd frontend
npm ci
npm run build          # writes ..\static_react\
cd ..
```

**Run it:**

```
conda activate twinguard
python server.py
```

Open <http://127.0.0.1:8000>. The **first** start runs the real model once to render every
panel (~5 min on the GPU):
- fits the detection threshold on the 50 val frames,
- ranks the 50 test frames,
- renders 6 of them with latency and MC-Dropout,
- builds the training gallery.

Later starts reuse `static\generated\` and are ready in seconds. The cache records which
checkpoint made it and rebuilds automatically if the checkpoint changes. To force a
rebuild, delete `static\generated\`.

A different checkpoint: `$env:TWINGUARD_DEMO_CHECKPOINT = "runs\phase2a_1head_run1\model_1head_best.pth"`.
Only 3-head checkpoints show head disagreement.

| Page | Shows |
|---|---|
| Home | headline numbers, how it works, calibration panel (raw / temperature / L_calib, whole and edge ECE) |
| Detection Demo | 6 **test-half** frames: 3 clearest + 3 typical, labelled. Per frame: detection boxes (blue) against the real outline (green) at the val-fitted threshold, frame AP, objects found; dual-mode latency and trigger; head-disagreement and MC-Dropout maps; segmentation, fused score, each head. Hover any heatmap for the value |
| Training Runs | findings table, then every run in order with its status |
| Training Data | 8 real CutMix samples (Cityscapes + COCO) with the anomaly mask |

**Editing the frontend with live reload:** keep `python server.py` running, and in a second
window run `cd frontend; npm run dev`, then open the address it prints (Vite forwards `/api`
and `/static` to port 8000). Run `npm run build` again afterwards, so `python server.py`
alone serves the new version.

Notes:
- The coloured strip along the top of some frames is in the original Lost & Found photos.
  Fishyscapes marks that border as "void", and every metric ignores it.
- Latency is measured on this laptop in bf16. It isn't the deployment GPU the paper's
  targets refer to.

---

## 9d. Explainability: *why* does it flag an object? (evaluation only)

`explain.py` answers "what is the model actually looking at?" by editing the picture and
re-running the model. No training; ~15 min on the reference laptop, all in fp32.

```
python validate_explain.py                    # first: the building blocks are correct (~1 min)
python explain.py --checkpoint runs\phase2a_coco_run1\model_3head_best.pth       # the full study
python explain.py --checkpoint runs\phase2a_coco_run1\model_3head_best.pth --demo   # figures for /demo (needs server.py run once)
```

| Step | What it does | Read it as |
|---|---|---|
| **X1 removal** | Repaints one part of an object (the object, its edge band, its core, its surroundings, or just blurs the band), re-runs the model, measures the peak score near the object | A big drop = the model relies on that part. "Detection lost" is the share of objects the model no longer flags. |
| **X1 controls** | Applies the same edit to a patch of clean road | If the control itself raises the score, the repaint is leaving an artefact. Judge a removal result *against* its control. |
| **X2 stages** | Which of the encoder's 4 stages (fine → coarse) the score depends on: gradient × activation, plus a faithful check that removes the object's features at one stage | Gradient share = where the score is sensitive; the removal column = what it needs. When they differ, trust the removal. |
| **X3 uncertainty** | Splits the 3 heads' uncertainty into disagreement between heads (epistemic) and each head's own (aleatoric) by region | Whether the signal says anything about *where* or *which direction* the edge is wrong. |

Everything is done twice: on **real** Fishyscapes test objects and on **pasted** CutMix objects
(what the heads trained on). Comparing the two shows whether the model treats pastes
differently. Confidence intervals resample whole *images*, because objects in one image are
not independent.

Outputs: the printed report (save it with `| Tee-Object runs\explain_run1.log`),
`runs\explain\summary.json`, `runs\explain\x1_objects.csv` (one row per object per edit).
`--demo` writes `static\generated\explain\`, which the **Detection Demo** page shows; until
you run it, that page says "pending" in plain words. The numbers are copied into
`runs\RESULTS.md`.

Design choices worth knowing (each measured, see RESULTS.md Part D):
- A plain copied road patch fires a false detection on clean road 71% of the time, so the
  patch fill is **Poisson-blended** (0% on the same control).
- Replacing a whole encoder stage by its global average was rejected: it makes the heads
  see impossible features everywhere. The ablation is *local* to the object.
- Edits are made at the model's 1024×512 input size; "edge band" = 4 px there, the same
  physical band as the r = 8 px edge-ECE at the 2048×1024 labels.

`record_route.py` (CARLA recording) is unrelated to this and needs a machine with the CARLA
simulator; see its header and the README.

---

## 10. MLflow — every number from every run

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://localhost:5000>. Every `train.py` and `calibrate.py` run is there with its
settings and per-epoch metrics. `Ctrl+C` stops the viewer.

Or from the terminal:

```
python check_runs.py           # all runs with the gates
python check_runs.py --trend   # per-epoch history
```

`mlflow.db` is local to this folder. It doesn't contain the friend's pod runs, and
it's git-ignored.

---

## Troubleshooting

**Training is several times slower than Section 0 says, with no error**
Almost always GPU memory spilling into system RAM. In `nvidia-smi`, `Memory-Usage` sits
right at the limit (~3950 / 4096 MiB). Check that `train.py` printed `batch 2 x 2 accumulation`.
If it says `batch 4 x 1`, the env vars weren't set in that window. Also close anything else
using the GPU.

**`CUDA out of memory`**
Something else is on the GPU (`nvidia-smi` shows what). Close it. As a last resort use
`TWINGUARD_BATCH_SIZE=1` with `TWINGUARD_GRAD_ACCUM=4` (same effective batch, slower).

**The log stopped updating, but no error**
The laptop slept, or Windows rebooted. Check `Get-Date` against the log's last line. See
Section 2. Restart the run; it doesn't resume mid-way.

**Very slow: epochs over 15 min**
1. On battery? Plug in.
2. Did `train.py` print `num_workers=4`? If it printed `0`, the env var was set in a different
   window.
3. `nvidia-smi -l 2`: GPU-Util should sit around 90–100%. Low utilisation with high CPU means
   data loading is the bottleneck.

**`BrokenPipeError` / `DataLoader worker exited unexpectedly`**
Windows starts workers as fresh processes, which costs RAM. Try `$env:TWINGUARD_NUM_WORKERS = "2"`.
`0` always works, just slowly.

**`no checkpoint at checkpoints\model_3head_best.pth`**
Phase 2a hasn't been run (or finished) yet, or you ran from the wrong folder. Always run
from `TwinGuard_CARLA\`.

**`no MLflow experiment named ...`**
Wrong folder. `mlflow.db` is a relative path.

**Checkpoint won't load: `Missing key(s)` / `Unexpected key(s)`**
transformers ≥ 5 renamed the SegFormer keys. Go back to `<4.51` (Section 1).

**Hundreds of `INFO [alembic.runtime.migration]` lines at startup**
Harmless. MLflow is creating a new `mlflow.db` the first time it runs in this folder.

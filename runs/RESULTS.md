# Local run log (exp_v2)

Every local run, in order. All numbers are from the Fishyscapes **test** half (50 images)
unless marked val.

**Only this file is in git.** The checkpoints (`runs/<name>/*.pth`) and logs (`runs/*.log`)
mentioned below were produced on the project lead's laptop and are git-ignored (a
checkpoint is 327 MB). If you need one, ask for it; don't retrain just to get it. Your
own training will land close to but not exactly on these numbers, because pasting is
random and ±0.04 AP is noise on 50 images.

**Current best Phase 2a checkpoint:** `runs/phase2a_coco_run1/model_3head_best.pth` (epoch 8, val-selected)
**1-head ablation checkpoint:** `runs/phase2a_1head_run1/model_1head_best.pth` (epoch 7, val-selected)
**Current best Phase 2b checkpoint:** none meets the target. `runs/phase2b_calib2/` is the best attempt (band-ECE equal to raw within CI, worse than temp(band)).

## Part A: evaluation-only steps (2026-09-27, all on phase2a_coco_run1, no training)

### A1: Experiment A baseline, same 50/50 split (`experiment_a.py`, log `runs/experiment_a.log`)

| test half | AUROC | AP | FPR@95 | ECE* |
|---|---|---|---|---|
| Experiment A (SegFormer-B5 MSP, untrained) | 0.8372 | 0.0114 | 1.0000 | 0.0148 |
| TwinGuard run 1 | **0.9910** | **0.7499** | **0.0209** | 0.0003 |

*Experiment A's ECE is over MSP, not a probability of anomaly, so it is not comparable.
FPR@95 = 1.0 means the baseline is fully confident that more than 5% of anomaly pixels
are a known class, so it can't reach 95% recall without flagging everything. Val half:
0.8170 / 0.0176 / 1.0000.

### A2: UBQ, per-object and local (`eval_spatial.py` + `metrics.ubq_local`, log `runs/ubq_run1.log`)

The old UBQ used a TPR-95 threshold (3% of the frame flagged) and the frame-wide worst
distance (~1080 px everywhere), so it measured far false alarms, not outlines. The new
version scores each true object on its own (ROI 32 px), counts missed objects separately,
reports far false alarms apart, and adds the standard boundary F1 (4 px tolerance).
Validated against brute force and sklearn in `validate_metrics.py`.

| raw, test half | objects found | spill px | miss px | boundary F1 | far-FP | flagged |
|---|---|---|---|---|---|---|
| **max-F1 (val) t=0.507** | 40/85 | 2.51 | 6.67 | 0.567 | 13.6% | 0.182% |
| t=0.1 | 51/85 | 5.11 | 3.28 | 0.574 | 43.8% | 0.441% |
| t=0.3 | 45/85 | 3.09 | 4.57 | 0.614 | 24.0% | 0.259% |
| t=0.5 | 40/85 | 2.55 | 6.56 | 0.570 | 13.9% | 0.185% |
| t=0.7 | 33/85 | 1.86 | 9.14 | 0.531 | 7.6% | 0.132% |
| t=0.9 | 25/85 | 0.77 | 11.79 | 0.349 | 2.6% | 0.073% |

- **Stable (Novelty 4):** boundary F1 stays at 0.53–0.61 over t=0.3–0.7, with spill and miss
  trading off smoothly. It degrades at t=0.9, when only 25 objects remain.
- **Temperature scaling leaves UBQ unchanged at the fitted operating point:** raw,
  temp(whole) and temp(band) give identical rows at max-F1. A monotone global rescale only
  moves where the threshold falls, never the outline, which is the plan's Novelty 1/4
  argument, now measured.
- Object recall is 47% at the operating point. The misses are mostly small or distant
  objects: a detection limit, not a boundary one.

### A3: Novelty 7, between-head vs within-head (`check_uncertainty_split.py`, log `runs/uncertainty_split_run1.log`)

50 test images, clean and under each training degradation at mid severity, 10 MC-Dropout
passes/head, fp32.

| condition | fused AUROC | fused AP | between-head AUROC | within-head AUROC | between ×clean (normal px) | within ×clean (normal px) |
|---|---|---|---|---|---|---|
| clean | 0.9910 | 0.7480 | 0.9892 | 0.9686 | — | — |
| noise | 0.9513 | 0.5459 | 0.7302 | 0.7019 | ×28.45 | ×30.54 |
| motion blur | 0.9776 | 0.5713 | 0.9640 | 0.9244 | ×2.51 | ×2.61 |
| fog | 0.9923 | 0.7351 | 0.9884 | 0.9660 | ×1.12 | ×1.08 |

**Split not supported.** Between-head is the (slightly) better novelty signal (0.989 vs
0.969), but the two signals rise by nearly identical factors under every degradation. They
are essentially the same quantity ("how unsure the heads are"), so this architecture can't
tell a degraded sensor from a novel object with them. Side findings: noise is the main
robustness weakness (AP 0.75 → 0.55), then motion blur (0.57). Fog barely matters.

### A4: RoadAnomaly21, second benchmark (`eval_road_anomaly.py`, log `runs/road_anomaly21_run1.log`)

Only the 10 validation images have public labels (the 100 test labels are withheld,
online submission only). Archive `dataset_AnomalyTrack.zip`, md5 verified, in
`<data root>/road_anomaly21/`. Web photos, 1280×720, anomalies 1.7–36.8% of the image
(Lost & Found: 0.28%).

| RoadAnomaly21 val (10 imgs) | AUROC | AP | FPR@95 | ECE |
|---|---|---|---|---|
| Experiment A (MSP) | **0.8706** | **0.4765** | 0.3500 | 0.0992 |
| TwinGuard run 1 | 0.7214 | 0.3849 | 1.0000 | 0.1266 |

**TwinGuard is worse than the baseline here.** Per image, small anomalies do well (1.7%
of the image → AUROC 0.998; 6.8% → 0.987) and large close-up ones badly (22% → 0.64;
37% → 0.69), with one exception (10.8% → 0.45). The likely cause is the CutMix scale range,
deliberately fitted to Lost & Found (object ≤ 20% of image height), so the heads have never
seen a large anomaly. With 10 images this is a strong hint, not a proof. Report as a
generalisation limitation.

## Part B: 1-head ablation + Novelty-5 table (2026-09-27)

**Training:** `phase2a_1head_run1`, the same recipe as run 1 (COCO, 8 epochs, batch 2 × 2
accum, 4 workers), with `TWINGUARD_NUM_HEADS=1`. The single head uses seed 42, the same as
head 0 of the 3-head model, so head count is the only difference. Total OOD loss weight is
kept equal (α = 1 for 1 head, 1/3 × 3 heads), though it barely matters under AdamW: the
encoder is frozen and the heads share no trainable weights with the seg head.
~8 min/epoch, GPU 85–87 °C, 2.3 GB.

| epoch | val AP | test AUROC | test AP | test FPR@95 | mIoU (100) |
|---|---|---|---|---|---|
| 1 | 0.6099 | 0.9878 | 0.7052 | 0.0488 | 0.7329 |
| 2 | 0.6483 | 0.9894 | 0.6988 | 0.0308 | 0.7508 |
| 3 | 0.6957 | 0.9917 | 0.7707 | 0.0197 | 0.7574 |
| 4 | 0.6676 | — | — | — | 0.7625 |
| 5 | 0.6576 | — | — | — | 0.7624 |
| 6 | 0.6529 | — | — | — | 0.7641 |
| **7** | **0.7049** | **0.9938** | **0.7770** | **0.0152** | 0.7672 |
| 8 | 0.6788 | 0.9881 | 0.7511 | 0.0196 | 0.7688 |

Selected epoch 7 (val). Checkpoint and log in `runs/phase2a_1head_run1/`.

**Novelty-5 table** (`check_ablation.py`, log `runs/ablation_table.log`; Fishyscapes test half,
Cityscapes val mIoU on all 500 images):

| model | AUROC | AP | FPR@95 | ECE | disagreement AUROC | MC within-head AUROC | mIoU |
|---|---|---|---|---|---|---|---|
| 1-head | 0.9938 | 0.7770 | 0.0152 | 0.0004 | — (one head) | 0.9676 | 0.7681 |
| 3-head (run 1) | 0.9910 | 0.7499 | 0.0209 | 0.0003 | **0.9845** | 0.9689 | 0.7700 |
| 3-head + L_calib (calib2) | 0.9910 | 0.7534 | 0.0203 | 0.0003 | 0.9830 | 0.9741 | 0.7704 |

**What it shows:**
- **Normal segmentation doesn't suffer.** mIoU is 0.768–0.770 for all three, and no class
  moves by more than 0.02 between them (largest: train 0.739 → 0.722 with L_calib, 0.019).
  This is the plan's Novelty-5 check, passed.
- **Three heads don't detect better than one.** The 1-head model is slightly ahead on the
  test half (AP 0.777 vs 0.750) and behind on the val half (0.705 vs 0.730). Differences this
  size flip between the two 50-image halves, so report them as equal detection quality.
- **What three heads add is uncertainty in a single pass.** Head disagreement separates
  anomalies at AUROC 0.985 and comes free with the one continuous-mode pass. A single head
  can get a similar signal from MC-Dropout (0.968), but only with 10 extra passes. That's the
  actual argument for the twin-head design: the dual-mode latency case, not detection accuracy.

## Part C: Phase 3 demo (2026-09-27)

Built on the teammate's `server.py` + `frontend/` from `main` (copied into exp_v2 with
`git archive`, main untouched). Run: `python server.py` → http://127.0.0.1:8000 (setup in
LOCAL_GUIDE.md §9c).

**Bugs fixed in the original backend (all four would have shown wrong numbers):**
1. No ImageNet normalisation on the input (the frozen encoder needs it). Now uses `load_image_tensor`.
2. Double sigmoid on scores that are already probabilities (squashed everything into 0.5–0.73).
3. Stale history (old failed runs shown as "current checkpoint, AUROC 0.619").
4. Training gallery showed the misaligned CARLA frames. Now real CutMix samples.

**Changed for honesty:** frames come from the test half only, 3 clearest + 3 typical
(median), labelled. Boxes use the val-fitted max-F1 threshold (0.507), not a per-image
top-3%. The true outline is drawn in green so false alarms are visible. The cache records
its checkpoint and rebuilds when it changes.

**Added:** a dual-mode section (measured latency, trigger state, head-disagreement vs
MC-Dropout maps), per-frame AP / objects found / boundary F1, and a findings table on the
Training Runs page.

**Verified:** frontend lint + build clean. All 11 routes (4 pages, 3 APIs, panels, grids,
JS bundle, 404) respond correctly; 96/96 panel and grid assets present. The demo
checkpoint is byte-identical to `runs/phase2a_coco_run1/`. Panels inspected by eye:

| frame | selection | frame AP | objects found | boxes on object | trigger | latency cont./safety |
|---|---|---|---|---|---|---|
| demo_00 | clearest | 0.848 | 1/1 | 1/1 | fires | 175 / 493 ms |
| demo_01 | clearest | 0.975 | 2/2 | 2/2 | fires | 176 / 497 ms |
| demo_02 | clearest | 0.996 | 2/2 | 2/2 | fires | 177 / 500 ms |
| demo_03 | typical | 0.666 | 0/1 | 0/0 | no (peak 0.45) | 179 / 501 ms |
| demo_04 | typical | 0.676 | 1/2 | 1/1 | fires | 177 / 503 ms |
| demo_05 | typical | 0.023 | 0/1 | 0/5 (all false alarms) | fires | 177 / 506 ms |

The head-disagreement map shows rings on object edges: the heads agree on object
interiors and disagree at outlines, which is the Phase-2b edge finding made visible.
Laptop latency (RTX 3050 Ti, bf16): continuous ≈ 177 ms, safety ≈ 500 ms. Not the paper's
deployment targets (<35 / <100 ms), which need a deployment GPU to test.

## Part D: explainability (X1–X4) and the `record_route.py` port (2026-10-05)

Built from the plan a teammate wrote on `main`; code, fills, statistics and tests are this
branch's own. `explain.py` (full study ~13 min, fp32), `validate_explain.py` (all checks pass),
logs `runs/explain_run2.log` (final) and `runs/explain_run1.log` (first run, flawed fill, kept as
a record). Data: 78 real objects in 45 Fishyscapes test images (≥12 px at 1024×512), of which
**39 are detected** (peak ≥ 0.536, the val-fitted threshold at this resolution); 61 pasted objects
in 30 CutMix composites (60 detected). Real objects are smaller: median 155 px vs 360 px, so the
real-vs-paste comparisons are also repeated size-matched (100–500 px: 18 real, 19 paste).
Confidence intervals resample whole images.

### The fill problem (the main method lesson)

Every removal needs something to paint with, and the fill can itself look anomalous. The
plan's advice (compare two fills, add a control) was necessary and not sufficient. Measured on 58
clean-road controls and 30 pasted objects:

| fill | false detections on clean road | pasted objects removed (mean score left in the object) |
|---|---|---|
| hard road-patch copy | 55% | 47% lost (0.094) |
| Poisson-blended patch | 10% | **0% lost (0.922)**: leaves a flat grey ghost of the object |
| Telea inpainting | 22% | 73% (0.120) |
| **feathered road patch (σ=2), used** | 22% | 80% (0.041) |

The Poisson blend looked the cleanest only because it removed nothing; the first full run
(`explain_run1.log`) used it and reported "pasted objects stay detected after removal",
which was wrong (only 1.7% lost, because the patch had left a ghost). With the feathered patch the two fills agree: whole-object removal loses
**54% vs 51%** of real detections and **70% vs 68%** of pasted ones. Even so, the fill alone makes the model fire on
10–27% of clean-road patches (by edit type), so "detection lost" is a **lower bound**, and the
"excess over fill artefact" column (post-edit peak minus the same edit's peak on clean road)
is the cleaner number.

### X1 counterfactual removal (detected objects; Telea; neighbourhood = object + 4 px)

| edit | real: peak before→after | real: detection lost [95% CI] | paste: detection lost [95% CI] | real: excess over artefact |
|---|---|---|---|---|
| whole object | 0.89 → 0.46 | **54%** [38, 70] | **70%** [60, 81] | +0.15 [+0.03, +0.27] |
| edge band only (core kept) | 0.89 → 0.81 | 18% [5, 33] | 28% [16, 41] | +0.59 |
| core only (edge band kept) | 0.94 → 0.96 | **0%** [0, 0] | 0% [0, 0] | +0.81 |
| surroundings (4–20 px ring) | 0.89 → 0.96 | 3% [0, 8] | 0% [0, 0] | n/a (no control) |
| edge band blurred | 0.89 → 0.87 | 13% [4, 25] | 12% [3, 21] | +0.63 |

What it shows:
- **The evidence is spread out.** Repainting the edge band while keeping the core, or the core while keeping the
  edge band, rarely removes the detection (0–28%). Only repainting the whole object does (about half of real, 70% of pasted objects).
  Paired peak drop, edge band − core: +0.012 [−0.014, +0.058] real, +0.039 [0.000, +0.103] paste, so no
  significant difference. The hypothesis "the model keys on the pasted edge" is **not supported**; neither is "content only".
- **Edge sharpness barely matters:** blurring the band loses 13% (real) vs 12% (paste); size-matched real−paste +0.020 [−0.080, +0.153].
- **The model does not need the surroundings:** repainting a ring 4–20 px around the object does not lower its score (it rises
  slightly, 0.89 → 0.96; the fill artefact is not controlled for this edit, so read it as "no dependence").
- **Pasted objects are fully removable, real ones leave a residue.** After whole-object removal, a pasted object is at the
  fill-artefact level (excess +0.05 [−0.05, +0.14] Telea, +0.03 [−0.09, +0.15] patch); a real object keeps +0.15 [+0.03, +0.27] /
  +0.17 [0.00, +0.34] above it. What the residue is (shadow, contact darkening, labels that under-cover the object) was **not tested**.
- Size-matched real−paste peak drop for whole-object removal: −0.289 [−0.470, −0.090] Telea, −0.177 [−0.426, +0.092] patch.

### X2 which encoder stage carries the score (detected objects)

| | stage 1 (1/4) | stage 2 (1/8) | stage 3 (1/16) | stage 4 (1/32) |
|---|---|---|---|---|
| gradient share, real | 12% | 26% | 30% | 32% |
| gradient share, paste | 17% | 31% | 26% | 25% |
| real − paste, size-matched (pts) | −5.9 [−7.4, −4.7] | −6.5 [−8.8, −4.4] | +5.4 [+3.5, +7.5] | +7.0 [+4.4, +9.6] |
| **removal**: score cut when this stage's features at the object are replaced by their surroundings, real | **34%** [24, 43] | **56%** [47, 66] | **64%** [51, 76] | 16% [5, 28] |
| removal, paste | 11% [7, 14] | 33% [26, 41] | 10% [6, 15] | 0% [−1, 0] |
| removal, real − paste, size-matched (pts) | +28.7 [18, 39] | +31.4 [14, 48] | +44.4 [22, 67] | −0.6 [−8, +5] |

- Gradient share and removal **disagree about stage 4**: it carries 25–32% of the gradient but removing its evidence barely changes
  the score (real 16%, paste 0%). The removal result is the faithful one: the evidence lives in the mid-scale stages 2–3.
- Real objects depend far more on each of stages 1–3 than pasted ones (size-matched +29 to +44 pts): a pasted object's score
  (saturated near 1.0) survives losing any single stage, a real object's does not. This fits the Phase 2b finding that the model is
  ~99% sure of pastes and only 50–69% sure of real objects, but is a consistency check, not proof of that explanation.
- A tested hypothesis that did **not** hold: pasted objects relying on the finest stage (the "paste artefact" shortcut). Stage 1 matters *less* for pastes (11% vs 34%).
- A first ablation (whole stage replaced by its global average) was rejected: it feeds the heads impossible features everywhere and
  raised scores up to 50×. The kept ablation is local to the object.
- Maps: stages 1–3 peak within about a cell of the object (mean offset −8 to +2 px); stage 4's median offset is 27 px, inside its 32 px cell,
  larger than most objects. Read the stage-4 map as "in this neighbourhood".

### X3 epistemic / aleatoric split of the 3 heads (real test objects, bits)

| region | mean score | total | aleatoric | epistemic (head disagreement) | epistemic share |
|---|---|---|---|---|---|
| object core | 0.708 | 0.516 | 0.501 | 0.0153 | 3.0% |
| inner edge band | 0.445 | 0.530 | 0.516 | 0.0148 | 2.8% |
| outer edge band | 0.102 | 0.256 | 0.249 | 0.0069 | 2.7% |
| background | 0.0009 | 0.0037 | 0.0036 | 0.0001 | 2.9% |

- Almost all uncertainty is "aleatoric" (the heads agree that it is uncertain); disagreement is ~3% everywhere. Its *location* signal is
  strong (epistemic ≈ 150× higher in the object core and inner edge than in the background, 2× higher inside the edge than just outside it).
- Mean uncertainty inside the ±4 px edge band by outcome: TP 0.57, **FN 0.50**, FP 0.74, TN 0.23 (pixels: TP 10,390; FN 14,276; FP 1,762; TN 28,260).
  The dominant edge error is the **miss** (8× more than false alarms), and misses look about as certain as hits.
- AUROC for "this edge pixel is wrong": total 0.665, aleatoric 0.663, epistemic 0.653, head spread 0.662, versus 0.638 for the naive
  "near the threshold" baseline. For telling the direction of an error (miss vs false alarm): 0.32–0.41 (below 0.5 means high uncertainty
  → more likely a false alarm). So uncertainty cannot flag the edge under-confidence found in Phase 2b. The 3 heads share one frozen
  encoder, so their disagreement understates true epistemic uncertainty.

### Limits

One checkpoint, 39 detected real objects, fp32 at 1024×512, edits at the model's input size. Edge band = 4 px there (= the r = 8 px band at
label resolution). The feathered fill's skirt bleeds ≤2 px into a kept core. Objects the model does not detect (39 of 78) are excluded from
the removal statistics by construction. Pasted objects come from the training bank. The label masks come from nearest-neighbour downscaling.

### `record_route.py` (ported from `main`, CARLA recording with synchronous capture)

Logic verified against a simulated CARLA (10 checks: tick sequence, stale-frame draining, BGRA decoding, mask = the object rendered in the
same frame's RGB for all 6 frames, files, cleanup). It has **not** been run against the real simulator (no CARLA on this laptop); its
first real run should be a small pilot whose frames are looked at. It is the intended fix for the `generate_anomalies.py` sync bug.

### Demo panel (X4)

`python explain.py --checkpoint <ckpt> --demo` writes `static/generated/explain/`; `/demo` shows the aggregate removal table and, per frame, the
before/after figure with the repaint outline and the stage bars. Until generated, the page says "pending". Rendered and inspected in headless
Edge; `/api/explain` serves 6 frames and 66 assets. A bug found by that end-to-end test (a bare `NaN` in the JSON made the endpoint fail) is fixed
at the source and covered by a test.

## Reference points (not run here)

| Checkpoint | AUROC | AP | FPR@95 | ECE | band-ECE r=8 | Note |
|---|---|---|---|---|---|---|
| COCO-only run 3 (pod, lost) | 0.9905 | 0.7917 | 0.0168 | 0.0003 | — | target to reproduce |
| friend's CARLA+COCO raw (`../model_3head_best.pth`) | 0.9920 | 0.6212 | 0.0292 | 0.00044 | 0.2274 | |
| same, temp(whole) T=1.2247 | 0.9923 | 0.6203 | 0.0290 | 0.00013 | 0.2096 | plain-NLL fit |
| same, temp(band) T=4.3681 | 0.9924 | 0.6170 | 0.0286 | 0.0422 | 0.0938 | edge-fitted T |
| friend's L_calib (`../model_3head_calib_best.pth`) | 0.9924 | 0.6020 | 0.0294 | 0.0005 | 0.2395 | worse than raw at r=8 (CI excludes 0) |

## Runs

| # | Name | Phase | What changed | AUROC | AP | FPR@95 | ECE | band-ECE r=8 | mIoU | Kept? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (aborted) | 2a | batch 4 on the 4 GB GPU | — | — | — | — | — | — | killed at step ~300: 3 s/step, VRAM spilled into system RAM |
| 1 | phase2a_coco_run1 | 2a | COCO, 8 epochs, batch 2 × 2 accum (effective 4), workers=4, 10 min/epoch | 0.9910 | 0.7499 | 0.0209 | 0.0003 | see calib1 | 0.7695 | **yes**, the reportable Phase 2a model |

Run 1 per epoch (test half is scored only when val improves):

| epoch | val AP | test AUROC | test AP | test FPR@95 | mIoU |
|---|---|---|---|---|---|
| 1 | 0.6039 | 0.9904 | 0.6290 | 0.0308 | 0.7338 |
| 2 | 0.6650 | 0.9886 | 0.7037 | 0.0355 | 0.7494 |
| 3 | 0.6477 | — | — | — | 0.7560 |
| 4 | 0.6502 | — | — | — | 0.7614 |
| 5 | 0.7241 | 0.9946 | 0.7882 | 0.0150 | 0.7600 |
| 6 | 0.6887 | — | — | — | 0.7629 |
| 7 | 0.7223 | — | — | — | 0.7699 |
| **8** | **0.7296** | **0.9910** | **0.7499** | **0.0209** | **0.7695** |

Head-disagreement AUROC (epoch 8): 0.9845 (run 3: 0.9842).

### Phase 2b on phase2a_coco_run1

Post-hoc baselines (test half; fitted on val only):

| calibration | AUROC | AP | ECE | band-ECE r=8 |
|---|---|---|---|---|
| raw | 0.9909 | 0.7491 | 0.00030 | 0.1950 |
| temp(whole) T=1.2206 | — | — | — | — |
| temp(band) T=3.6202 | 0.9938 | 0.7486 | 0.02034 | 0.1058 |
| prior shift −ln 20 (no fit) | 0.9824 | 0.7480 | 0.00174 | 0.3659 |
| Platt(whole) 0.710·z − 0.677 | 0.9938 | 0.7489 | 0.00047 | 0.2427 |
| **Platt(band) 0.350·z + 0.507** | 0.9938 | 0.7486 | 0.01603 | **0.0541** |

Every edge-fitted global map raises whole-image ECE 50–70×. That is the gap L_calib has to close.

| # | Name | What changed | val AP | val ECE | val band-ECE | Kept? |
|---|---|---|---|---|---|---|
| base | — | phase2a_coco_run1 | 0.7292 | 0.00082 | 0.1531 | — |
| c1 | phase2b_calib1 | β=1, band L_calib, degradation 0.3, lr 1e-5 | 0.7162 (ep2) | 0.00070 | 0.2466 (ep2) | no, stopped after epoch 2 (ep1 0.2673) |

**Why c1 failed (measured with 400-step probes, `finetune_probe.py`):**

| probe | pastes band-ECE | pastes conf / pos | real band-ECE | real conf / pos | real AP |
|---|---|---|---|---|---|
| before | 0.144 | 0.49 / 0.35 | 0.153 | 0.29 / 0.44 | 0.729 |
| A: β=1 | 0.017 | 0.36 / 0.35 | 0.291 | 0.15 / 0.44 | 0.724 |
| B: β=0 (control) | 0.137 | 0.49 / 0.35 | 0.153 | 0.30 / 0.44 | 0.732 |

At object edges the model is on average *over*-confident on synthetic pastes and
*under*-confident on real Fishyscapes anomalies (it misses parts of real objects). L_calib
fixes the pastes by lowering edge scores, which moves real edges the wrong way. The
control shows nothing else in the fine-tune does this. Both domains are too *sharp*
(band T > 1 on each), so the shared part is the scores being too extreme, not their average level.

**Anchor probes** (edge-band mean confidence held at the base model's level; 400 steps):

| probe | β | γ (anchor) | real edge conf (pos 0.441) | real band-ECE | real AP | real ECE |
|---|---|---|---|---|---|---|
| before | — | — | 0.294 | 0.1531 | 0.7292 | 0.00082 |
| A | 1 | 0 | 0.150 | 0.2911 | 0.7240 | 0.00144 |
| B | 0 | 0 | 0.295 | 0.1531 | 0.7323 | 0.00057 |
| C | 1 | 10 | 0.264 | 0.1768 | 0.7379 | 0.00052 |
| **D** | 1 | 100 | **0.302** | **0.1450** | 0.7388 | 0.00079 |
| E | 5 | 100 | 0.276 | 0.1681 | 0.7364 | 0.00060 |
| F | 3 | 300 | 0.288 | 0.1575 | 0.7299 | 0.00075 |

Across all six, real band-ECE tracks real edge confidence: real anomalies need edge scores
*raised* (the model misses parts of them), while pastes pull them down. D is the only
variant that doesn't lower them.

**Post-hoc maps that use only model-observable information don't help** (test band-ECE, raw 0.1950):
disagreement-conditioned Platt 0.2371; two-region Platt around the model's own predicted
edge 0.2387 (τ=0.5, R=8) / 0.2445 (τ=0.1, R=16). Only maps *fitted on the GT band* lower
it (Platt(band) 0.054), and they do so by pulling every score toward the band's ~44% base
rate, which costs 50–70× whole-image ECE. A simulation with a model calibrated by
construction gives GT-band ECE 0.005, so the metric itself is sound. The real model's
0.195 is mostly *localization* error: parts of real objects scored near 0 and spill just
outside scored high. No per-pixel monotone map can fix that without knowing the true edge.

**c2 (phase2b_calib2): c1 + anchor γ=100, 5 epochs.** Val band-ECE per epoch: 0.1573,
0.1468, 0.1512, 0.1499, **0.1451** (selected ep5). Test half:

| model | AUROC | AP | FPR@95 | ECE | band-ECE r=8 |
|---|---|---|---|---|---|
| raw | 0.9909 | 0.7491 | 0.0211 | 0.00030 | 0.1950 |
| temp(whole) T=1.2206 | 0.9937 | 0.7490 | 0.0211 | 0.00039 | 0.1844 |
| temp(band) T=3.6202 | 0.9938 | 0.7486 | 0.0210 | 0.02034 | 0.1058 |
| L_calib c2 | 0.9909 | 0.7527 | 0.0206 | 0.00030 | 0.1894 |

**Not a success.** c2 is 3% better than raw on band-ECE and keeps whole-image ECE, but
plain temp(whole) is better still. Kept (`runs/phase2b_calib2/`) only as the record of
this attempt.

**Root cause, measured (mean fused score, phase2a_coco_run1):**

| | inner edge (label 1) | outer edge (label 0) | deep interior (label 1) |
|---|---|---|---|
| real Fishyscapes (val) | 0.50 | 0.13 | 0.69 |
| pastes, run-1 settings | 0.95–0.98 | 0.23–0.28 | 0.99–1.00 |
| pastes, harmonize U(0.6,1.0) + blur U(0,1.5) | 0.73 | 0.13 | 0.78 |
| pastes, harmonize 1.0 + blur 2.5 | 0.37 | 0.09 | 0.51 |

Training pastes are far easier than real anomalies. The model is ~100% sure of every pasted
object, so whatever L_calib learns about confidence on them doesn't transfer. The fix goes in
the Phase 2a data, not the Phase 2b loss: `CUTMIX_HARMONIZE_STRENGTH` and
`CUTMIX_OBJECT_BLUR_SIGMA` in config.py.

| 2 | phase2a_coco_run2 | 2a | run 1 + harder pastes: harmonize U(0.6,1.0), object blur U(0,2.0) px; val-selected ep6 | 0.9821 | 0.7495 | 0.0787 | 0.0005 | 0.1947 | 0.7629 | **no**, checkpoint deleted; log in `runs/phase2a_coco_run2/` |

**Run 2 didn't work, and it shows the limit of this whole approach.** After training on
the harder pastes, the model is 97% sure of them again (inner edge 0.89, deep interior
0.97) while real anomalies stay at 0.48 / 0.62. Its band-ECE (0.1947) is unchanged from run 1
(0.1950), and detection is worse (FPR@95 0.079 vs 0.021, val AP 0.659 vs 0.730). Band
temperature T=4.24; temp(whole) 0.1904, temp(band) 0.0978 with whole ECE 0.059.

### Final evaluation of phase2a_coco_run1 + phase2b_calib2 (eval_spatial.py, test half)

| model | whole ECE | band-ECE r=4 | r=8 | r=16 |
|---|---|---|---|---|
| raw | 0.0003 | 0.2479 | 0.1950 | 0.1409 |
| temp(whole) T=1.2206 | 0.0004 | 0.2315 | 0.1844 | 0.1369 |
| temp(band) T=3.6202 | 0.0203 | 0.1332 | 0.1058 | 0.1021 |
| L_calib (c2) | 0.0003 | 0.2397 | 0.1894 | 0.1369 |

Paired bootstrap (1000 resamples, 95% CI):
- L_calib − raw: r=4 −0.008 [−0.017, +0.001], r=8 −0.006 [−0.014, +0.004], r=16 −0.004 [−0.010, +0.003]. **No measurable effect.**
- L_calib − temp(band): r=4 +0.107 [+0.081, +0.135], r=8 +0.084 [+0.041, +0.112]. **Worse, significant.** r=16 +0.035 [−0.015, +0.084].
- temp(band) − raw: r=4 −0.115, r=8 −0.089 (both CIs exclude 0).

UBQ remains saturated (pred_to_gt ≈ 1080 px, 3.4% of frame predicted positive at the TPR-95 threshold). Not usable yet.

### Dual-mode trigger (check_dual_mode.py, phase2a_coco_run1, laptop GPU)

Latency (RTX 3050 Ti laptop, not the deployment GPU): continuous 224 ms, safety 629 ms
(10 passes × 3 heads, one encoder pass). **No trigger meets <5% normal / >90% anomalous**:
peak-pixel at 0.5 gives 60% / 72.5%; best area trigger (≥10 px ≥ 0.9) gives 20% / 47.5%. 80%
of normal Cityscapes val frames have ≥10 pixels scoring ≥ 0.3, so this is a frame-level
detection limit.

`check_detections.py` at threshold 0.30: 9 of 33 boxes (27%) land on a real anomaly.

A model is always overconfident on the data it trains on, which is why calibration is
normally fitted on held-out data. Any synthetic anomaly the detector trains on becomes
easy, however it is made, so L_calib computed on those anomalies can't learn real-world
uncertainty. Config reverted to run 1's paste settings; run 1 restored as
`checkpoints/model_3head_best.pth`.

Epoch 5 had the higher *test* AP (0.788) and is kept as `runs/phase2a_coco_run1_ep5/`
for reference only. It is **not reportable**: val AP picked epoch 8, and switching to
epoch 5 because of its test score would be selecting on the test set. The ~0.04 AP gap
between two epochs with near-equal val AP (0.724 vs 0.730) shows how noisy AP is on 50
test images. Report it as such, not as a regression from run 3.

# TwinGuard: paper tables (P7 draft, 2026-10-06)

Every number below is copied from a log in the repo root (named in each caption) or from a
signed PLAN.md section. Conventions:
- **Model:** `model_3head_best.pth`, a frozen SegFormer-b5 encoder with 3 OOD heads, trained with
  CutMix pastes from a 2000-object bank (500 CARLA tiles from 45 objects + 1500 COCO cutouts).
  L_calib rows use `model_3head_calib_best.pth`.
- **Input:** 1024×512 (scale 1; full resolution gave no gain, P3).
- **Fishyscapes Lost & Found:** the 100 public val images are split 50/50 into val and test.
  Every calibrator and threshold is fitted on val and reported on test.
- **Metrics:** AP is reported first, because AUROC ≈ 0.99 is saturated. Absolute numbers only, no
  ratios. Intervals are 95% bootstrap CIs, paired over images; on video they use a 20-frame
  circular block bootstrap.
- **No comparison with published leaderboards:** our test set is a 50-image half of the public
  L&F val, so it is a different split.

## Table 1. Detection (Fishyscapes test half)

| Model | AP | AUROC | FPR@95 | Source |
|---|---|---|---|---|
| TwinGuard, 3-head mean (raw) | 0.6215 | 0.9920 | 0.0290 | `eval_fishyscapes_T122.log` |
| Head 0 alone (same network) | 0.5810 | n/r | n/r | `eval_fishyscapes_T122.log` |
| Δ head 0 − 3-head mean | −0.0406 [−0.0685, −0.0125] | | | paired over 50 images |
| RoadAnomaly21 val (10 images), TwinGuard 3-head | 0.3084 | 0.6550 | 1.0000 | `eval_road_anomaly21.log` |
| RoadAnomaly21 val (10 images), MSP baseline (Experiment A) | 0.4744 | 0.8705 | 0.3504 | `eval_road_anomaly21.log` |

**RoadAnomaly21 is a generalisation limitation.**
- Anomalies there are large and close: 14.81% of valid pixels, against 0.28% in Lost & Found. On
  those, TwinGuard falls below plain MSP.
- Per image it ranges from AP 0.098 / AUROC 0.477 (validation0005, 10.8% anomalous) to AP 0.813 /
  AUROC 0.972 (validation0006, 6.8%).
- The lead's separately trained COCO-only checkpoint showed the same pattern: AUROC 0.72 vs MSP
  0.87 (exp_v2 `runs/RESULTS.md`, labelled by checkpoint).
- Nothing was fitted or selected on these 10 images.
- The zip's md5 was not checked (UNVERIFIED); the image/label layout matched (10/10 pairs).

Head 0 is one member of a jointly trained ensemble at the epoch selected on the fused val AP.
The Δ is therefore the averaging gain at a fixed epoch, not "3 heads beat 1". (A separately
trained 1-head model on another checkpoint matched 3 heads; exp_v2, the lead's COCO-only run.)

## Table 2. Calibration (Fishyscapes test half; `eval_fishyscapes_T122.log`, temp whole ECE from `calibrate_compare_T122.log`)

| Method | AP | AUROC | FPR@95 | whole-image ECE | band-ECE r=8 |
|---|---|---|---|---|---|
| raw | 0.6215 | 0.9920 | 0.0290 | 0.0004 | 0.2273 |
| temperature, whole-image fit, T = 1.2251 | 0.6206 | 0.9924 | 0.0289 | 0.00012 | 0.2095 |
| temperature, 50/50 band/background fit, T = 3.1516 | 0.6177 | 0.9925 | 0.0286 | 0.0177 | 0.1277 |
| temperature, edge-band fit (C2), T = 4.3592 | 0.6173 | 0.9925 | 0.0286 | 0.0420 | 0.0942 |
| disagreement-conditioned T(d) (C3) | 0.6106 | 0.9923 | 0.0286 | 0.0193 | 0.1312 |
| neighbourhood-context calibrator (C5) | 0.4929 | 0.9707 | 0.0949 | 0.0264 | 0.0738 |
| L_calib (soft-ECE loss, CutMix-trained) | 0.6024 | 0.9924 | 0.0293 | 0.0005 | 0.2396 |

The r=8 band covers the pixels within 8 px of an object outline at label resolution. Raw
band-ECE at r=4/8/16 is 0.2848/0.2273/0.1748. At every radius the band is
**under-confident**: at r=8 the mean score is 0.2258 against a positive rate of 0.4421
(`eval_fishyscapes_c5.log`).

Paired differences (`eval_fishyscapes_T122.log`):
- C2 − temp(1.2251): band r=8 −0.1153 [−0.1378, −0.0732]; whole +0.0418 [+0.0390, +0.0443].
- C3 − T 3.1516 control: band r=8 +0.0035 [−0.0010, +0.0052]. The CI includes 0, so C3 is no
  better than a single T.
- C5 − temp(1.2251): band r=8 −0.1358 [−0.1608, −0.0993]; whole +0.0263 [+0.0240, +0.0278].
- L_calib − temp(1.2251): band r=8 +0.0301 [+0.0196, +0.0397]. L_calib − raw: +0.0123
  [+0.0017, +0.0258].

## Table 3. Outline quality (Fishyscapes test half; `eval_fishyscapes_T122.log`)

Each row is scored at its own val max-F1 threshold, with a 32 px ROI and 4 px tolerance.
Boundary F1, spill and miss are means over the objects found.

| Method | threshold | objects found (of 85) | boundary F1 | spill px | miss px | far-FP (% of flagged px) |
|---|---|---|---|---|---|---|
| raw | 0.4766 | 40 | 0.505 | 2.97 | 8.94 | 14.7% |
| temp T = 1.2251 | 0.4969 | 39 | 0.514 | 3.16 | 9.33 | 13.8% |
| C2 T = 4.3592 | 0.5011 | 39 | 0.509 | 3.13 | 9.47 | 13.3% |
| C3 | 0.5228 | 38 | 0.500 | 2.17 | 11.75 | 11.1% |
| C5 | 0.7217 | 59 | 0.584 | 5.37 | 4.50 | 36.1% |
| L_calib | 0.4612 | 43 | 0.537 | 2.57 | 7.62 | 19.4% |
| head 0 only | 0.3621 | 37 | 0.508 | 3.21 | 7.32 | 17.5% |

C5 − raw per-image boundary F1: +0.0950 [+0.0349, +0.1543], over the 24/50 images where both
find an object. This is bought with far false alarms (36.1% vs 14.7%) and AP (0.6215 → 0.4929).

## Table 4. Calibration learned on pastes moves edges in the paste direction (Finding 4)

| Data | edge direction (r=8 gap) | L_calib − raw, r=8 | L_calib − temp(1.2251), r=8 | Source |
|---|---|---|---|---|
| Fishyscapes test (real) | under-confident (−0.2163) | +0.0123 [+0.0017, +0.0258] (worse) | +0.0301 [+0.0196, +0.0397] (worse) | `eval_fishyscapes_T122.log` |
| Pasted CARLA route, 400 frames | over-confident (+0.1292) | −0.0162 [−0.0207, −0.0114] (better) | −0.0046 [−0.0104, +0.0016] (≈ 0) | `eval_video_town02_pasted_T122.log` (block CI) |

The CARLA route uses self-fitted thresholds (no val split) and shows pastes in a simulator,
not the real world. It also contains CARLA-bank pastes whose mask quality is under audit (A1).

## Table 5. Same-photo paste test, X5 (Fishyscapes test photos; `paste_test.log`)

Each photo gets one paste per forward pass. All three paste types in a photo share the same
centre and size, and every paste sits ≥ 48 px from the real object. 50 photos were placed and
no arm was dropped. In the original photo the paste pixels score 0.000 (the exact removal).

| Object | n | mean px | object score | band-ECE (±4 px at 1024×512) | edge gap (score − positive rate) |
|---|---|---|---|---|---|
| COCO paste, never in training | 50 | 915 | 0.939 | 0.3138 | +0.298 |
| COCO paste, in the training bank | 50 | 839 | 0.927 | 0.2837 | +0.292 |
| CARLA bank paste (A1-"aligned" objects) | 50 | 816 | 0.842 | 0.2652 | +0.234 |
| Real Fishyscapes object, same photos | 37 | 1392 | 0.347 | 0.2348 | −0.179 |

**Pre-registered contrasts:**
- **Primary.** Edge gap, unseen paste − real = **+0.468 [+0.401, +0.532]** (n = 37). This
  contrast is **formally confounded** under our rule. The unseen-paste forward pass moved the
  real object's score by −0.0032 [−0.0059, −0.0007], and that CI excludes 0. The other two
  drift checks passed: seen −0.0008 [−0.0036, +0.0027], CARLA −0.0010 [−0.0038, +0.0017].
  With the seen arm, whose drift check passes, the gap is +0.453 [+0.381, +0.523] (exploratory).
- **Secondary.** Seen − unseen object score = −0.011 [−0.053, +0.033]: no detectable
  memorisation. This stays provisional: whether the local COCO bank is the training copy is
  unverified.

## Table 6. Which encoder stage the score needs: X2 faithfulness (`explain_x2_faith.log`)

Detected objects only (peak ≥ 0.4766). "Ablation drop" = the relative fall in the object's
mean fused score when that stage's features at the object are replaced by the mean of a ring of
cells around it, in the stage's own grid.

| | stage 1 (1/4) | stage 2 (1/8) | stage 3 (1/16) | stage 4 (1/32) | top-stage agreement, gradient vs ablation |
|---|---|---|---|---|---|
| Real objects (Fishyscapes, n = 24): ablation drop | 27.8% | 51.7% | **74.4%** | 28.2% | 0.25 [0.08, 0.42]: **not validated** |
| Real objects: grad×act share | 0.16 | 0.21 | 0.29 | 0.33 | |
| Pasted objects (CARLA route, n = 75): ablation drop | 2.7% | 15.8% | **20.4%** | 12.4% | 0.56 [0.44, 0.67]: faithful at top-stage level |
| Pasted objects: grad×act share | 0.17 | 0.35 | 0.29 | 0.19 | |

The rule was fixed in advance: agreement > 0.5 with the CI above 0.25 (chance). Gradient
attribution therefore stays a footnote for real objects. On the lead's COCO-only checkpoint,
exp_v2 found the same gradient/ablation disagreement at stage 4 (exp_v2 `runs/RESULTS.md`
Part D, labelled by checkpoint).

## Table 7. Counterfactual removal and uncertainty split, X1/X1b/X3

| Test | Result | Source |
|---|---|---|
| X1: real objects, Telea inpainting removal (37 objects) | 0.347 → 0.091; drop +0.256 [+0.145, +0.362]. A lower bound, because the fill itself scores | `explain_x1.log` |
| X1b: pasted CARLA objects, exact counterfactual (75 frames) | with paste 0.951 → truly removed 0.016; Telea-removed 0.625 | `explain_x1b.log` |
| X1b: Telea fill bias (Telea after − true after) | +0.609 [+0.538, +0.681] | `explain_x1b.log` |
| X3: Fishyscapes edge band | aleatoric 0.236 vs epistemic 0.010 | `explain_x23.log` |
| X3: false-positive − true-positive head disagreement | +0.0067 [−0.0153, +0.0277], n = 9 (not supported) | `explain_x23.log` |

## Table 8. Driving video (pasted CARLA route, Town02, 400 frames; `eval_video_town02_pasted_T122.log`, `static/video/twinguard_video_town02_pasted_stats.json`)

| | value |
|---|---|
| Pixel AP / AUROC (raw, self-fit) | 0.6604 / 0.9799 |
| Objects boxed, 3-head mean (threshold 0.5, min box 150 px) | 862 / 1245 (69.2%), 3.585 false boxes per frame |
| Objects boxed, head 0 | 858 / 1245 (68.9%), 3.40 false boxes per frame |
| Whole-image ECE, raw → temp(1.2251) | 0.009205 → 0.010719 (a real-val T worsens this over-confident route) |

## Table 9. Why each calibration attempt failed

| Attempt | What it tried | Why it failed | Evidence |
|---|---|---|---|
| L_calib | fine-tune with a differentiable ECE loss on CutMix pastes | lowers edge scores, which is the wrong direction for real objects (under-confident edges) | Table 4 |
| L_calib, β = 50 | stronger loss weight | surrogate down, real ECE up: 0.0004 → 0.0014; AUROC 0.9920 → 0.9898 | `config.py` BETA_CALIB history (no log on disk; cite as "observed in development") |
| Whole-image T = 1.2251 | one T on real val | fits the background, which is >99% of pixels; edge ECE only 0.2273 → 0.2095 | Table 2 |
| Edge T = 4.3592 (C2) | T fitted on r=8 band pixels | one T cannot serve both regions; whole ECE 0.00012 → 0.0420 | Table 2 |
| Disagreement T(d) (C3) | T varies with head disagreement | disagreement marks *where* edges are, not which side is wrong | C3 − control CI includes 0 |
| Context calibrator (C5) | neighbourhood max/mean score | spreads anomaly score into nearby background | AP 0.6215 → 0.4929; far-FP 14.7% → 36.1% |

## Limitations (to state in the paper)

- One trained checkpoint; a 50-image test half; 85 objects, 37–40 of them used, depending on
  the analysis.
- The heads were trained on half-resolution features only. Finding 2 cannot separate "learned
  at the training resolution" from "the heads don't transfer across scale".
- The CARLA results are pastes rendered in a simulator, with self-fitted thresholds and one
  route. A1 found that 15–21 of the 45 CARLA bank objects were cut with misaligned masks:
  166–233 of the 500 CARLA tiles, or 8–12% of the 2000-object bank. That range is pending a
  visual check.
- The X5 primary contrast is formally confounded (see Table 5). Bank identity, U0-4, is
  unverified.
- RoadAnomaly21 has 10 labelled images: a sanity check, not a benchmark. On it, TwinGuard is
  below MSP (Table 1). The paste-trained heads do not generalise to large, close anomalies.

## References (checked 2026-10-06; the first two were confirmed online)

1. K. Vinogradova, A. Dibrov, G. Myers. Towards Interpretable Semantic Segmentation via
   Gradient-Weighted Class Activation Mapping (Student Abstract). AAAI 2020.
   doi:10.1609/aaai.v34i10.7244. arXiv:2002.11434. *(confirmed online)*
2. R. L. Draelos, L. Carin. Use HiResCAM instead of Grad-CAM for faithful explanations of
   convolutional neural networks. arXiv:2011.08891, 2020. *(confirmed online)*
3. V. Petsiuk, A. Das, K. Saenko. RISE: Randomized Input Sampling for Explanation of Black-box
   Models. BMVC 2018.
4. S. Depeweg, J. M. Hernández-Lobato, F. Doshi-Velez, S. Udluft. Decomposition of Uncertainty
   in Bayesian Deep Learning for Efficient and Risk-sensitive Learning. ICML 2018.
5. R. Chan et al. SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation. NeurIPS Datasets and
   Benchmarks 2021.
6. H. Blum et al. The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation.
   IJCV 2021.
7. P. Pinggera et al. Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles.
   IROS 2016.
8. K. Lis, K. Nakka, P. Fua, M. Salzmann. Detecting the Unexpected via Image Resynthesis.
   ICCV 2019.
9. E. Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with
   Transformers. NeurIPS 2021.
10. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks.
    ICML 2017.
11. A. Dosovitskiy et al. CARLA: An Open Urban Driving Simulator. CoRL 2017.
12. M. Cordts et al. The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR 2016.
13. T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
14. S. Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
    Features. ICCV 2019.
15. D. Hendrycks, K. Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution
    Examples in Neural Networks. ICLR 2017.
16. A. Telea. An Image Inpainting Technique Based on the Fast Marching Method. Journal of
    Graphics Tools, 2004.

References 3–16 are standard and were checked against memory, not online. Check each DOI
before submission.

## Title options (the user chooses)

1. "Calibrated Everywhere but the Edges: Boundary Miscalibration in Pixel-Level Anomaly
   Detection for Autonomous Driving" (the original recommendation)
2. "TwinGuard: Exposing Hidden Boundary Miscalibration in Road Anomaly Segmentation"
3. "Whole-Image ECE Is Not Enough: Boundary-Aware Calibration Analysis for Road Anomaly
   Detection"
4. "Why Synthetic Pastes Mis-Calibrate Real Anomalies: A Boundary Study with TwinGuard". X5
   now gives this direct same-photo evidence (Table 5), with the confound caveat.

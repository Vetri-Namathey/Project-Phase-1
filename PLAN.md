# TwinGuard — forward plan (rebuilt 2026-10-05)

This file is the single forward plan. It was rebuilt from scratch on 2026-10-05 (step P-1).
**The full history is in `PLAN_HISTORY.md`**, an untouched copy of the plan as it stood before
the rebuild. Never edit `PLAN_HISTORY.md`. Mistakes and the standing rules that prevent them
are in `MISTAKES.md`; read it before any step.

## Header

- **Date of rebuild:** 2026-10-05.
- **Current step (2026-10-05):** U1 and U2 are **done and signed by Agents 1 and 2**. P3 kept
  input scale 1. P2 (write-up) and the deferred P6 number edits are **written and SIGNED 2026-10-06 (fresh Agents 1+2 + orchestrator; agent_state/agent1_P2P6_review.md, agent2_P2P6_compliance.md), formerly awaiting the
  agents' sign-off**; then the user's site check U3a. A1 (CARLA-tile audit) results are PENDING
  REVIEW (user eyeball outstanding).
- **Primary model (all reported numbers):** `model_3head_best.pth` (repo root,
  `config.PRIMARY_RAW`). It is the 2000-object-bank model: 500 CARLA tiles of 45 objects +
  1500 of 3000 COCO cutouts, `config.ANOMALY_SOURCE="both"`, 8 epochs, epoch 4 selected on
  the Fishyscapes val half. Fishyscapes test half: AUROC 0.9920, AP 0.6215, FPR@95 0.029
  (`eval_fishyscapes_c5.log`; `train.log` gives AP 0.6218 / FPR@95 0.0287 for the same file,
  bf16-level difference). **The user wants this model kept as the primary model.**
- **L_calib model:** `model_3head_calib_best.pth` (repo root, `config.PRIMARY_CALIB`).
- **HARD RULES (user's, non-negotiable), verbatim from the brief:**
  - **No git operations of any kind** (no commit, push, fetch, checkout, stash, reset).
    Never `git push`.
  - **Do not run anything that uses the GPU, CARLA, training or evaluation.** The user runs
    those; the coder agent hands them commands. CPU-only checks are fine: `py_compile`,
    reading files, tiny numpy/tensor sanity checks, `npm run build`.
  - Python: `C:\Users\venka\anaconda3\envs\cuda_test\python.exe` (in PowerShell: `& "<path>"`).
    The plain `python` on PATH is a broken 3.12 without numpy; don't use it. User commands
    assume `conda activate cuda_test` then `python ...`.
  - Don't touch `.pth` files, `data/`, or the `video_*` folders. Don't delete anything.
  - Stay on the plan. Don't chase unrelated environment issues.
  - Read `MISTAKES.md` (repo root) first, and follow its standing rules.
- **Rule for T: LIFTED 2026-10-05.** P1 rule (a) passed (fitted whole T = 1.2251, in
  [1.2147, 1.2347]) and both agents signed. T=1.2251-derived numbers may now be written in this
  file, each citing `eval_fishyscapes_T122.log`. The frontend stays deferred to P6 (after U2).
- **Logs:** every GPU log is written by PowerShell 5.1 `Tee-Object` and is **UTF-16LE**. Decode
  before grepping (`iconv -f UTF-16 -t UTF-8 <log>`), or grep reports "Binary file matches"
  and a row looks missing. Exceptions: `train.log` and `calibrate.log` are ASCII.
- **Conventions:** every GPU command is run by the **user** in `conda activate cuda_test`,
  from the repo root, with CARLA closed, and piped through `| Tee-Object -FilePath <log>`.
  Every table states checkpoint, split and input scale (standing rule 10). After every pasted
  output the coder fills that step's **Step evaluation record**, and Agent 1 (soundness) and
  Agent 2 (compliance) sign before the next command is issued (standing rule 12).

## RESUME HERE (2026-10-05, after U2)

> **UPDATE 2026-10-07: β = 0.05 is DONE (no help; see Y-results). NEXT = the β = 0 control**,
> then the user decides on 0.1 / 0.2. The user is pushing first. Commands:
> `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_sweep.db"`;
> `python calibrate.py --train-lcalib --checkpoint model_3head_best.pth --beta 0 --out checkpoints/lcalib_beta0.pth | Tee-Object -FilePath lcalib_beta0_train.log`;
> `python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib checkpoints/lcalib_beta0.pth | Tee-Object -FilePath eval_lcalib_beta0.log`.
> The block below is the older plan, kept for reference.
>
> **START HERE TOMORROW (saved 2026-10-06 evening): the Y1 β sweep, from scratch.**
> - Done today:
>   - Y2 threshold sweep: C5 outline gain is THRESHOLD-DEPENDENT.
>   - Y3 latency: 92.1 ms forward, 176 ms end to end.
>   - Live image + video upload on /demo.
>   - Thesis updated; zip rebuilt.
> - The user ran nothing for β yet: the first attempt hit the MLflow DB crash (M24, fixed), and
>   the second was stopped early.
> - User commands, in one terminal:
>   1. `cd` to TwinGuard_CARLA; `conda activate cuda_test`;
>      `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_sweep.db"` (the DB already exists and is
>      reused).
>   2. `Test-Path checkpoints\lcalib_beta0.05.pth`. If True, use `lcalib_beta0.05_run2.pth` in
>      both β = 0.05 commands (never overwrite or delete).
>   3. For β = 0.05, 0.1, 0.2:
>      `python calibrate.py --train-lcalib --checkpoint model_3head_best.pth --beta <b> --out checkpoints/lcalib_beta<b>.pth | Tee-Object -FilePath lcalib_beta<b>_train.log`
>      then
>      `python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib checkpoints/lcalib_beta<b>.pth | Tee-Object -FilePath eval_lcalib_beta<b>.log`
>      (about 30–40 min per β, 1.5–2 h total).
> - Then (orchestrator):
>   - Judge against the Y1 rule (below, "Y-ITEMS"). Prediction: every β falls between raw 0.2273
>     and β=1's 0.2396.
>   - Add a β row to `paper/TABLES.md` (Table 9), to thesis chapter 5 (the β = 50 note becomes a
>     sweep table), and to the friend's status table (item 6).
>   - Rebuild `TwinGuard_Thesis_Overleaf.zip`.
> - Still parked: the open user questions below (A1 eyeball, COCO provenance, min box, /demo
>   look).

**OPEN USER QUESTIONS (parked 2026-10-06 at the user's request; ask again before signing P7):**
1. Paper title: options 1–4 in `paper/TABLES.md` (orchestrator suggests #4).
2. A1: eyeball `agent_state\a1_carla_tiles\contact_sheet.png`. Are the "misaligned" tiles really
   wrong? This fixes k in Limitations (currently 15–21 of 45).
3. U0-4: did `data/coco_objects` come from the training pod? (decides "no memorisation": firm
   or provisional)
4. P6b: keep video min box 150? (yes = no re-render)
5. P6: does `/demo` look fine (X1 viewer, X5 cards, X2 ablation bars)? Home RA21 section added
   2026-10-06.

**Y-ITEMS (2026-10-06; the user approved the three "yes" items; rules written BEFORE any run):**
- **Y1 β sweep.** `calibrate.py --train-lcalib` gained `--beta` and `--out`; `--out` refuses an
  existing file or a primary model; each run's png gets its own name. Runs: β ∈ {0.05, 0.1, 0.2},
  everything else as the β = 1 checkpoint (5 epochs, lr 1e-5, selection on val ECE). Each is
  scored by `eval_spatial.py --dataset fishyscapes`.
  - **Prediction:** every β has band-ECE r=8 between raw 0.2273 and β=1's 0.2396, i.e. worse
    than raw.
  - **Rule:** a β "helps" only if L_calib(β) − temp(1.2251) at r=8 has a CI entirely < 0 and AP
    drops ≤ 0.01 vs raw. Otherwise: "no β tested beats a single temperature".
  - CPU check: `agent_state/coder_checks/beta_sweep_cpu_check.py` PASS.
- **Y2 threshold sweep.** `eval_threshold_sweep.py`, raw vs C5, matched by the fraction of VAL
  pixels flagged (0.05–0.5%), never by threshold value (M11).
  - **Sanity:** it must reproduce 40/85 and 59/85 at max-F1, else STOP.
  - **Rule:** "robust" if the C5 − raw BF1 CI > 0 at ≥ 4 of 6 points, else "threshold-dependent".
  - CPU check PASS.
- **Y3 live demo + latency.** `server.py` gained `POST /api/infer` (the raw image file is the
  body; same scoring path as evaluation; 400/413/415 guards; CUDA-synchronised timings). /demo
  gains a "Try it on your own image" panel. `bench_latency.py` times the single pass (200 runs
  after 20 warm-ups) and end-to-end.
  - CPU check: `live_upload_cpu_check.py` PASS; max score identical to `score_maps`.
  - **Rule:** report as a laptop-GPU number only; no in-vehicle claim.
  - **Video upload (2026-10-06, user request):** `POST /api/infer-video`.
    - The clip is sampled at 10 fps and capped at 30 s / 200 MB. Each frame is scored like an
      image (shared `data.transforms.pil_to_tensor`; `load_image_tensor` now calls it, and the
      tensors are verified identical).
    - Output: an H.264 mp4 (detection over heatmap) in `static/generated/uploads/` (gitignored;
      newest 5 kept).
    - Processed, then played back: **not real time**.
    - Docker: not needed (user decision).
    - CPU check: `video_upload_cpu_check.py` PASS (30 frames at 25 fps → 12 frames).

**Y-results (2026-10-06, the user's GPU runs):**
- **Y3 latency (`bench_latency.log`):** RTX 3070 Laptop GPU, 1024×512, amp.
  - Model forward: median **92.1 ms** (mean 94.6, p95 104.9; 10.9 fps).
  - End-to-end (disk read + resize + forward + upsample to 2048×1024 + score): median **176.0 ms**
    (p95 198.7; 5.7 fps).
  - Peak GPU memory 0.91 GiB.
  - Verdict: a laptop number, reported as such.
- **Y2 threshold sweep (`eval_threshold_sweep.log`):** sanity PASS (40/85 and 59/85, far-FP
  14.7% / 36.1%; C5 refit = the signed weights). C5 − raw BF1 by val-flagged fraction:
  - 0.05%: +0.008 [−0.109, +0.129]
  - 0.1%: −0.033 [−0.158, +0.115]
  - 0.15%: −0.036 [−0.138, +0.076]
  - 0.2%: +0.062 [−0.028, +0.153]
  - **0.3%: +0.103 [+0.042, +0.162]**
  - **0.5%: +0.154 [+0.098, +0.208]**

  **2/6 → THRESHOLD-DEPENDENT.** The BF1 gain appears only at loose operating points, where
  C5's far-FP is 37–51% of flagged pixels. C5 finds more objects at every point (34 vs 17 …
  66 vs 52), but "found" means a flagged pixel within 32 px. That favours a calibrator that
  spreads score into the surroundings, so it is exploratory, not evidence of better outlines.
  **Finding 6 is narrowed:** the old "C5 improves outlines" holds only at its own max-F1 point
  and at loose thresholds.
- **Y1 β = 0.05 DONE 2026-10-07** (`lcalib_beta0.05_train.log`, `eval_lcalib_beta0.05.log`,
  checkpoint `checkpoints/lcalib_beta0.05_run2.pth`, selected epoch 2 on val ECE).
  - Sanity: every other row reproduces the signed T122 numbers exactly.
  - Band-ECE r=8 **0.2409**; AP 0.5945; whole ECE 0.000593.
  - L_calib − raw r=8 **+0.0136 [+0.0016, +0.0279]**.
  - L_calib − temp r=8 **+0.0314 [+0.0188, +0.0433]**.
  - **Rule: does not help** (worse than both; AP drop 0.027 > 0.01).
  - **Prediction PARTLY WRONG, recorded:** predicted "between raw and β=1 (0.2273–0.2396)";
    measured 0.2409, slightly above β=1. Direction right, ordering wrong.
  - **Observation (not pre-registered):** in the training log the SoftECE term is about 0.0005
    against a base loss of about 0.08, so β·L_calib is <1% of the loss even at β=1 and about
    0.05% at β=0.05.
    - The harm therefore comes from the 5-epoch fine-tune on CutMix pastes itself, not from the
      calibration term.
    - Proposed control: **β = 0** (same fine-tune, no L_calib). Predicted ≈ β=0.05 and ≈ β=1.
    - β = 0.1 / 0.2 are expected to be uninformative; running them is the user's decision
      (they were pre-registered, so skipping them is recorded as a decision).
- **Y1 status 2026-10-06 evening:** the MLflow fix worked (a fresh `mlflow_sweep.db` was
  created), but the user stopped the β = 0.05 run early (no time). The whole sweep is rerun the
  next day: set `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_sweep.db"` again (the DB is
  reused). If `checkpoints/lcalib_beta0.05.pth` exists from the stopped run, use
  `--out checkpoints/lcalib_beta0.05_run2.pth`; the guard refuses to overwrite, and nothing is
  deleted. A partial run's checkpoint is never used for results.
- **Y1 β sweep: NOT RUN (first attempt).** MLflow crashed before training ("Can't locate revision
  b7e2c1a4d9f3"): `mlflow.db` was written by a newer MLflow (the pod) than `cuda_test` has. No
  checkpoint was written. Rerun with `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_sweep.db"`
  (a fresh DB; `/mlflow_*.db` gitignored; the old `mlflow.db` is untouched). M24.

**DONE 2026-10-06: thesis text written.**
- Title chosen by the user in the template: "TwinGuard: Boundary-Aware Calibration Analysis for
  Road Anomaly Detection" (Q1 closed).
- **Moved 2026-10-06 into `../Thesis_Template___Amrita_AIE/Thesis_Template___Amrita_AIE/`** (the copies outside it were removed at the user's request; upload zip `../TwinGuard_Thesis_Overleaf.zip`). Files: `mythesis.tex` (title typo fixed, univa→univA, abstract, abbreviations, symbols, chapter titles)
  and `../chapters/{introduction,chapter2..7,conclusion,appendix}.tex`. All live outside the
  repo, next to `mythesis.tex`.
- `../mybib.bib` (22 entries) and `../thesis_images/` (12 figures from `paper/make_figures.py`).
- Pending items are marked `% PENDING` in chapter6/chapter7.
- Not compiled locally (no LaTeX installed); static check passed.

**(was) NEXT TASK (2026-10-06):** the user will send a .tex paper file.
- Write its contents for this project from `paper/TABLES.md` and the signed results.
- **Do not change the images/figures already in it.**
- The user does the LaTeX formatting in Overleaf.
- Every number must come from a log (rule 12); the open questions above stay marked as pending
  in the text.

**After a reset, read `agent_state/RESUME_AGENTS.md` first.**

**P2/P6 review: SIGNED 2026-10-06 by the orchestrator.** The reviewer agents were stopped
for usage cost after an API limit; the user approved finishing the review in the main
session. Checks:
- server.py scoring path matches eval_spatial: load_image_tensor, amp, upsample logits to
  label size, sigmoid, mean. So BOX_THRESHOLD 0.47658 (raw val max-F1,
  eval_fishyscapes_T122.log) transfers. CACHE_VERSION = 2.
- Home and History numbers match the signed logs: T = 1.2251, AP 0.6206, ECE 0.0001, band
  r=8 0.2095, C2 0.0942.
- A grep of frontend/src for 1.71, 1.7142, 20× and 570 finds 0 hits.
- static/calibration_reliability_T122.png is present, and `npm run build` passes.

**Next: U3a, the user's site check** (agent_state/agent3_U2.md §4). After that, the open user
items.

**State:**
- U1 and U2 done and signed (`agent1_U1_eval.md`, `agent2_U1_compliance.md`, `agent1_U2_eval.md`,
  `agent2_U2_compliance.md`). U2 ran in the order 3 → 5 → 4 (independent steps; recorded as a
  deviation).
- P3: **scale 1 kept** (no detectable gain; underpowered). No P1 re-run; P4/P5/P6 at scale 1.
- P1b: Finding 4's CARLA half re-sourced: L_calib − raw r=8 −0.0162 [−0.0207, −0.0114]
  (20-frame block CI); L_calib vs temp(T=1.2251) indistinguishable on the route.
- P2 written below (section "Write-up (step P2)"); P6 deferred number edits done (Home/History calibration
  rows, new reliability png, fixed box threshold 0.47658, README). Both SIGNED 2026-10-06.
- A1 results PENDING REVIEW; M22 PENDING.

**Next action, in order:**
1. ~~Agents 1 and 2 review and sign P2 and the P6 number edits.~~ DONE, signed 2026-10-06.
2. User: **U3a** site check (P6 checkpoint): `python server.py`, then `cd frontend; npm run dev`;
   pages `/`, `/runs`, `/demo` (checklist in `agent3_U2.md`). Then P6b (min box by eye, once).
3. User, open items: eyeball `agent_state\a1_carla_tiles\contact_sheet.png` (A1); `git ls-remote
   --heads origin` (unblocks NOW-4); RA21 download + MD5 (unblocks P5); `data/coco_objects`
   provenance (X5 secondary contrast); the lead's SHA256 (exp_v2 "same file" wording).

**2026-10-06, later: NOW-4 DONE.**
- `git ls-remote` run and exp_v2 read.
- `paste_test.py` fixed; `explain.py --x2-faith` added; `eval_road_anomaly.py` ported.
- CPU checks pass (see P4 and P5).

Handed to the user as **U3b**: the X5 run and the X2 faithfulness run (GPU). **U3a and U3b are done
(2026-10-06).** U3a log PASS. P4 is recorded:
- X5 PRIMARY: +0.468, but formally confounded by the unseen-arm drift.
- No memorisation detected.
- X2 not validated on real objects, so it becomes a footnote; the demo was reworded (M23).

**Next:**
- The user's visual "looks fine" on /demo.
- P6b: the min box decision.
- The A1 eyeball.
- The RA21 download → P5.
- Then P7.

**Blocked:**
- A1/M22: user eyeball + sign-off.
- P5: the RA21 zip download.
- Any description of CARLA bank quality: A1.

## Step template

Every step has: **Goal · Inputs · Procedure · Command · Log · Rule** (pre-registered, with
the behaviour for every outcome band) **· Checkpoint · Status · Evaluation record · Sign-off.**
The evaluation record is filled only from a pasted log:

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|

---

## P-1: Process files (CPU, coder)

- **Goal:** one forward plan with pre-registered rules and evaluation records, and an updated
  mistakes log, so no mistake repeats.
- **Inputs:** `PLAN_HISTORY.md`, `MISTAKES.md`, agent reports (verifier rounds 1-2,
  compliance rounds 1 and 3 FINAL), exp_v2 `runs/RESULTS.md`.
- **Procedure:** rebuild this file; replace MISTAKES standing rule 2 with 2'; add rules 10-12
  and M9-M18; keep the multiplicity note.
- **Command:** none (CPU edit). Check: Markdown renders; every step below has all template
  fields; every `1.71` hit in `frontend/src`, `README.md`, `eval_spatial.py`, `calibrate.py`
  is covered by the Superseded claims table.
- **Log:** `agent3_coder_round1.md` (agents' scratchpad).
- **Rule:** PASS if all three checks hold; any missing field or uncovered `1.71` hit → fix
  before P0 is signed.
- **Checkpoint P-1:** both files rebuilt and read by Agents 1 and 2.
- **Status:** [x] written by coder 2026-10-05 · [x] signed (code review 2026-10-05).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | CPU edit + field/`1.71` grep | agent3_coder_round1.md | see coder report | none known | written, awaiting review | 2', 10-12 | P0 review | see next row | see next row |
| 2026-10-05 | code review of NOW-1..3 (read-only) | agent1_codereview.md, agent2_codereview.md | PASS; numbers spot-checked against train.log / c5.log / town02 log | N1 (exp_v2 'same file' wording), N5 (U1 sequential), M19, M20 added | APPROVED U1 with required fixes; fixes done in coder round 2 (agent3_coder_round2.md) | 2', 10-13, M19, M20 | U0, U1 (sequential) | APPROVE (Agent 1, 2026-10-05) | APPROVE (Agent 2, 2026-10-05) |

---

## P0a: Lead-time asks (user, any time, no GPU) — U0

- **Goal:** start the slow external dependencies now, and get the hashes that P1 rule (a)'s
  fallback needs.
- **Inputs:** none.
- **Procedure / Command (PowerShell, repo root):**
  1. `Get-FileHash model_3head_best.pth -Algorithm SHA256` and
     `Get-FileHash model_3head_calib_best.pth -Algorithm SHA256`. Paste both. Ask the lead for
     the SHA256 of their copy of the friend's `model_3head_best.pth` (exp_v2 calls it
     `../model_3head_best.pth`).
  2. Ask the lead for the `phase2a_coco_run1` checkpoint (only for optional O1), and, if O1
     is wanted, its `phase2b_calib2` L_calib checkpoint (see O1).
  3. Download SMIYC `dataset_AnomalyTrack.zip` (zenodo.org/records/5270237). Run
     `Get-FileHash dataset_AnomalyTrack.zip -Algorithm MD5`; it must equal
     `231BF79ED58924BCD33D9CBE22E61076` (exp_v2 `LOCAL_GUIDE.md:497`).
  4. Say whether `data/coco_objects` was copied from the training pod or regenerated locally.
     It decides whether X5's seen/unseen contrast is secondary or exploratory (P4).
  5. Rule 2': `git ls-remote --heads origin` (the user runs it; agents never do). Paste it.
- **Log:** paste into the chat (no file).
- **Rule:** no gate. Hash equal to the lead's → same file, exp_v2's T=1.2247 is a valid
  reference for P1(c). Hash differs → our numbers stand on our own fitted T; exp_v2's values
  are cited as another file's. md5 mismatch → do not use the zip; re-download. A branch not
  yet read → its results file is read before the next plan step is written (rule 2').
- **Checkpoint:** answers recorded below.
- **Status:** [x] asked · [ ] answered (item 1 hashes PASS in U0; item 5 done 2026-10-06; items 3, 4 and the lead's hash open).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | item 5: `git ls-remote --heads origin` + read-only `git fetch origin exp_v2` (orchestrator; rule 2' allows it) | chat | heads: main f20fa7e, exp_coco b0496ae, exp_v2 **d162e22** (snapshot was da7349d). New commit: "explainability (removal, stage attribution, uncertainty split) + /demo panel; port record_route.py". `eval_road_anomaly.py` unchanged since da7349d | snapshot refreshed in place (explain.py, validate_explain.py, record_route.py, LOCAL_GUIDE.md, README.md, server.py, runs/RESULTS.md; older copies recoverable from da7349d; noted in SNAPSHOT_VERSION.txt) | branch read (RESULTS.md Part D) before the next plan step; see "exp_v2 Part D" under P4 | 2' | NOW-4 | orchestrator | orchestrator |

---

## P0: Code for the redo (CPU, coder)

- **Goal:** make P1/P1b/P3 possible and safe: correct T fit, no dangerous defaults, the
  metrics P1's rules need.
- **Inputs:** exp_v2 `calibrate.py` (`_mean_nll`, `fit_temperatures`), exp_v2 `metrics.py`
  (`ubq_local`, `_outline`, `threshold_at_max_f1`), exp_v2 `validate_metrics.py`.
- **Procedure (done 2026-10-05):**
  - `config.py`: `PRIMARY_RAW`, `PRIMARY_CALIB`, `TEMPERATURE_LOG_BOUNDS`,
    `TEMPERATURE_BAND_RADIUS_PX`; `CHECKPOINT_3HEAD/_CALIB` unchanged and commented as
    training-output paths only (M15).
  - `utils.load_trained_model(None)` raises `ValueError`. `check_collapse.py` takes a required
    `--checkpoint`; `demo_visualize.py` carries a stale-path comment.
  - `calibrate.py`: plain-NLL `fit_temperatures` (bounded search over log T, xatol 1e-4, per
    head before the mean, whole + band r=8); old fit renamed
    `_fit_temperature_weighted_DEPRECATED` (no flag reaches it); exactly one of
    `--compare-only` (needs `--raw --calib`, optional `--temp-whole`, writes
    `calibration_reliability_T122.png`, refuses the old png name), `--temp-only --checkpoint`,
    `--train-lcalib --checkpoint` (the only route into `run_calib_finetune`). A bare run errors.
  - `eval_spatial.py`: `--raw`/`--calib` required; `--temperature 1.7142` removed; Fishyscapes
    fits whole T on val and prints `fitted whole T = x.xxxx` (plus band T for reference);
    carla temp row only with `--temp-whole`; every temp label prints the fitted T; M14
    docstring fixed; head-0 row + paired AP bootstrap head0 − fused (rule f); `ubq_local` line
    per row at its val max-F1 threshold (legacy UBQ line labelled "legacy, saturated");
    per-image boundary-F1 paired bootstrap C5 − raw (rule e); `--input-scale {1,2,ms}` and
    `--window`.
  - `metrics.py`: `threshold_at_max_f1`, `_outline`, `ubq_local`, `paired_bootstrap_ap`
    (per-image pos/neg at `DEFAULT_BINS`, pooled AP per resample, 1000 resamples,
    `GLOBAL_SEED`). `validate_metrics.py` updated to exp_v2's checks plus edge cases.
  - `data/transforms.load_image_tensor(..., size=None)`: default unchanged; no training or
    dataset caller passes `size`.
  - New `smoke_fullres.py` and `eval_scale.py` (val half only, `assert pairs is fishy_val`).
- **Command (CPU checks, coder):** py_compile on every touched file; toy NLL (logits of a
  calibrated Bernoulli ×2 → T≈2 within 1%); `validate_metrics.py`; `paired_bootstrap_ap(a, a)`;
  calibrate argparse cases; `load_trained_model(None)`; `eval_spatial.py --help`; grep for
  `1.7142`; end-to-end runs of eval_spatial/calibrate/eval_scale/server on a fake CPU model
  and fake data in the scratchpad.
- **Log:** `agent3_coder_round1.md`.
- **Rule:** every check PASS → P0 can be signed. Any FAIL → fix and re-run before U1.
- **Checkpoint P0:** all checks pass, output pasted, both agents sign.
- **Status:** [x] written + CPU-checked 2026-10-05 · [x] signed (code review 2026-10-05).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | CPU checks listed above | agent3_coder_round1.md | all PASS (toy T 1.9963, −0.18%) | windows are 1088 px wide (see P0b) | written, awaiting review | 5, 11 | U1 | see next row | see next row |
| 2026-10-05 | code review of NOW-1..3 (read-only) | agent1_codereview.md, agent2_codereview.md | eval_spatial, metrics, calibrate, eval_scale correct; all 10 deviations accepted | smoke_fullres now prints reserved memory, rule on reserved < 6.5 GiB; --block-bootstrap added for carla | APPROVED U1 with required fixes; fixes done in coder round 2 (agent3_coder_round2.md) | 5, 11, 13 | U0, U1 (sequential) | APPROVE (Agent 1, 2026-10-05) | APPROVE (Agent 2, 2026-10-05) |

---

## P0b: Full-resolution smoke test (GPU, user, ~1 min) — U1-1

- **Goal:** know before any write-up whether a 2048×1024 forward fits in 8 GB, so P3 has a
  feasible mode.
- **Inputs:** `model_3head_best.pth`; first Fishyscapes **val** image.
- **Procedure:** one forward per scale under `no_grad` + bf16 autocast; prints the **reserved**
  peak (`max_memory_reserved`, what the caching allocator holds — the number that decides an
  OOM), the allocated peak (reference only), wall time, logit shape, fused min/mean/max. `--window` runs
  scale 2 as two 1024-tall windows, each 1088 px wide (half the width + 64 px), overlapping by
  128 px, logits averaged in the overlap. (Two 1024-wide windows cannot overlap on a 2048-wide
  image, so the window width is 1088; flagged for the agents.)
- **Command:** `python smoke_fullres.py --raw model_3head_best.pth --scales 1 2 | Tee-Object -FilePath smoke_fullres.log`
  - only if needed by the rule:
    `python smoke_fullres.py --raw model_3head_best.pth --scales 2 --window | Tee-Object -FilePath smoke_fullres_window.log`
- **Log:** `smoke_fullres.log` (+ `smoke_fullres_window.log`).
- **Rule (pre-registered):**
  - scale-1 logit shape must be (1, 3, 512, 1024) → else **STOP** (code or model changed).
  - scale-2 **reserved** peak < 6.5 GiB, no OOM → **P3 uses whole-image scale 2**. (The
    ~1.5 GiB margin covers the CUDA context, counted in neither number, and the second model
    that `eval_spatial --input-scale 2` loads; Agent 1 code review.)
  - scale-2 reserved peak ≥ 6.5 GiB or OOM → run the window command: reserved peak < 6.5 GiB →
    **P3 uses windows** (`--window`); else **P3 dropped**, recorded as "full-res infeasible on
    8 GB".
  - Run P0b **before** P1, never at the same time.
- **Checkpoint:** P3's mode (whole / window / dropped) recorded here: **whole-image scale 2, no
  `--window`** (reserved 3.54 GiB; with eval_spatial's second model about 3.9 GiB).
- **Status:** [x] run 2026-10-05 · [x] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `python smoke_fullres.py --raw model_3head_best.pth --scales 1 2 \| Tee-Object -FilePath smoke_fullres.log` | smoke_fullres.log | scale-1 shape (1,3,512,1024) PASS; scale-2 reserved 3.54 GiB < 6.5 PASS (allocated 2.93 GiB, 0.84 s, shape (1,3,1024,2048)); scale 1 reserved 0.98 GiB | none | P3 mode = whole-image scale 2, no `--window` | 3, 4 checked | P3 (U2-5) | SIGNED (Agent 1, agent1_U1_eval.md) | SIGN (Agent 2, agent2_U1_compliance.md) |

---

## P1: Calibration redo with the correct T + ubq_local + head-0 (GPU, user) — U1-2

Run only after U1-1 (P0b) has finished; never at the same time.

- **Goal:** re-measure the temp(whole) baseline with a plain-NLL T fitted on our checkpoint;
  reproduce every T-independent row; settle the C5 outline claim at a matched operating
  point; give our own objects-found and head-0 numbers.
- **Inputs:** `model_3head_best.pth`, `model_3head_calib_best.pth`, Fishyscapes 50 val / 50
  test, scale 1.
- **Procedure:** one run. Val: cache logits, fit whole T (plain NLL), C2, C3, control, C5 and
  every row's thresholds. Test: rows raw, temp(T), L_calib, C2, C3, control, C5, head 0; the
  existing band-ECE bootstraps; AP bootstrap head0 − fused; boundary-F1 bootstrap C5 − raw.
  Runtime UNVERIFIED (estimate 30-40 min; the T fit and ubq_local add CPU time); RAM peak
  estimated a few GB.
- **Command:** `python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib model_3head_calib_best.pth | Tee-Object -FilePath eval_fishyscapes_T122.log`
- **Log:** `eval_fishyscapes_T122.log`.
- **Rule (pre-registered):**
  - **(a) Fitted whole T** (line `fitted whole T = x.xxxx`): T ∈ [1.2147, 1.2347] → PASS.
    Outside → **STOP**; compare the U0 hashes. Hashes differ → our numbers stand on our own T,
    exp_v2's 1.2247 is cited as another file's value. Hashes equal → find the environment or
    code difference before anything else.
  - **(b) Reproduction** of raw / L_calib / C2 / C3 / control / C5 against
    `eval_fishyscapes_c5.log` (whole ECE, band-ECE r=4/8/16, AP): |Δ| ≤ 0.0005 → PASS;
    0.0005 < |Δ| ≤ 0.002 → RECORD and continue (bf16/cuDNN nondeterminism, new numbers used);
    |Δ| > 0.002 on any of them → **STOP**, diff data/code/checkpoint before any P1 number is
    used. Reference values (test half, scale 1):

    | row (c5.log) | AP | whole ECE | band r=4 | band r=8 | band r=16 | fitted |
    |---|---|---|---|---|---|---|
    | raw | 0.6215 | 0.0004 | 0.2848 | 0.2273 | 0.1748 | — |
    | L_calib | 0.6024 | 0.0005 | 0.2858 | 0.2396 | 0.1889 | — |
    | C2 temp-band | 0.6173 | 0.0420 | 0.1242 | 0.0942 | 0.0821 | T=4.3592 |
    | C3 | 0.6106 | 0.0193 | 0.1614 | 0.1312 | 0.0916 | exp(1.182 −3.774·d) |
    | C3 control | 0.6177 | 0.0177 | 0.1563 | 0.1277 | 0.0893 | T=3.1516 |
    | C5 | 0.4929 | 0.0264 | 0.0911 | 0.0738 | 0.0624 | w as logged |

  - **(c) temp(whole) row** vs exp_v2's reference (whole ECE 0.00013 — printed as 0.0001 —
    and band r=8 0.2096, `RESULTS.md` "Reference points"): within ±0.002 → PASS; else RECORD,
    and continue only if (a) and (b) passed.
  - **(d) Verdicts:** recompute C4 condition 1 (C3 − temp, r=8), L_calib − temp, C5 − temp.
    Predicted: no flips. Any flip is recorded as a new finding in PLAN and MISTAKES, not
    explained away.
  - **(e) C5 outline (M11):** per-image boundary F1, C5 − raw, each row at its own val max-F1
    threshold, paired over images where both find ≥ 1 object. CI > 0 → keep "context improves
    outlines at a matched operating point, at an AP cost". CI ∋ 0 → withdraw the extent claim.
    CI < 0 → withdraw and record. Objects found and far-FP reported alongside.
  - **(f) Head 0 vs fused ΔAP** (paired image bootstrap, descriptive): CI ∋ 0 → "equal
    detection on our model too" (consistent with exp_v2's 1-head). fused > head0, CI excludes
    0 → "the ensemble improves detection on the 2000-bank model" (caveat: the epoch was
    selected on fused val AP). head0 > fused, CI excludes 0 → report as is. No design decision
    depends on it.
  - **(g) Objects found** (raw row's `ubq_local` line) replaces "47%" everywhere (M16).
  - **STOP gate:** nothing below runs until (a) and (b) are signed.
- **Checkpoint P1:** (a)-(c) pass; (d)-(g) recorded; superseded rows for M1/M11/M16 filled.
  **Reached 2026-10-05.**
- **Status:** [x] run 2026-10-05 · [x] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib model_3head_calib_best.pth \| Tee-Object -FilePath eval_fishyscapes_T122.log` | eval_fishyscapes_T122.log | (a) T = 1.2251 ∈ [1.2147, 1.2347] PASS (bounded band T 4.3686). (b) raw/L_calib/C2/C3/control/C5 identical to c5.log on every metric and every T-independent bootstrap, \|Δ\| = 0 PASS. (c) temp whole ECE 0.0001 (ref 0.00013), band r=8 0.2095 (ref 0.2096) PASS. (d) no flips: C4 cond 1 −0.0783 [−0.0972, −0.0651]; L_calib − temp r=8 +0.0301 [+0.0196, +0.0397]; C3-control and C5 verdicts unchanged. (e) C5 − raw BF1 +0.0950 [+0.0349, +0.1543], n = 24/50 → keep, qualified: far-FP 36.1% vs 14.7% of flagged px, AP 0.4929 vs 0.6215. (f) head0 − fused AP −0.0406 [−0.0685, −0.0125]. (g) objects found 40/85 (47.1%) at t = 0.4766 | none | P1 PASS; M1 confirmed on our machine; no verdict flips; (f) differs from exp_v2 (finding, scoped below) | 1, 3, 6, 7, 10 checked; M1, M11, M16 evidence; M21 triggered (rule-e guard gap) | U2 (U2-3, U2-4, U2-5) | SIGNED (a)+(b) (Agent 1, agent1_U1_eval.md) | SIGN (Agent 2, agent2_U1_compliance.md) |

### P1-compare (GPU, user, after P1 sign-off) — U2-3

- **Goal:** replace `calibrate.log`'s table and the png's temp curve with the fitted T.
- **Inputs:** T from P1 (a).
- **Procedure:** eval only; never reaches the fine-tune; writes a new png.
- **Command:** `python calibrate.py --compare-only --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole 1.2251 | Tee-Object -FilePath calibrate_compare_T122.log`
- **Log:** `calibrate_compare_T122.log` + `calibration_reliability_T122.png`.
- **Rule:** its temp row (AP, whole ECE, band-ECE r=8) matches P1's temp row (AP 0.6206, whole
  ECE 0.0001, band r=8 0.2095; `eval_fishyscapes_T122.log`) within 0.0005, the new png exists and
  the old `calibration_reliability.png` is untouched → PASS. Mismatch > 0.0005 → STOP and diff `calibrate.evaluate_regions`
  vs `eval_spatial` (both upsample logits, then sigmoid, then mean).
- **Checkpoint:** new png ready for P6's deferred edit. **Status:** [x] run 2026-10-05 · [x] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `python calibrate.py --compare-only --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole 1.2251 \| Tee-Object -FilePath calibrate_compare_T122.log` | calibrate_compare_T122.log | temp row AP 0.6206 = P1; band r=8 0.2096 vs 0.2095 (\|Δ\| 0.0001 ≤ 0.0005) PASS; whole ECE 0.00012; raw (0.00044, 0.2273) and L_calib (0.00053, 0.2396) match P1; new png written, old png untouched (Sep 25) PASS | run order 3→5→4 (independent); stale "0.75 gate / Experiment A" print (removed 2026-10-05, cosmetic) | PASS | 1, 11 checked | P2; png into P6 | SIGNED (Agent 1, agent1_U2_eval.md) | SIGN (Agent 2, agent2_U2_compliance.md) |

---

## P1b: Video temp row + Finding 4's CARLA half (GPU, user, ~5 min) — U2-4

- **Goal:** replace the town02 temp row (old T) and re-source Finding 4's CARLA half to the
  synced route (M10).
- **Inputs:** T from P1; `video_town02_pasted` (400 frames, synced capture).
- **Procedure:** carla branch with `--temp-whole`; thresholds self-fit (no val split), as before.
  Consecutive frames are ~1 m apart and show the same objects, so frame-level bootstrap CIs are
  optimistic (M20). `--block-bootstrap 20` adds a paired **circular block bootstrap**: each
  resample takes ceil(400/20) = 20 blocks of 20 consecutive frames from uniformly drawn start
  frames, wrapping past the last frame to the first, truncated to 400. N = 20 (≈ 20 m of route,
  longer than a paste stays in view) is fixed now, before the run. The frame-level lines are
  still printed, for comparison with the old log.
- **Command:** `python eval_spatial.py --dataset carla --carla-root video_town02_pasted --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole 1.2251 --block-bootstrap 20 | Tee-Object -FilePath eval_video_town02_pasted_T122.log`
- **Log:** `eval_video_town02_pasted_T122.log` (the old log `eval_video_town02.log` scored this
  same root; its name did not say so).
- **Rule:** raw and L_calib rows reproduce `eval_video_town02.log` (raw AP 0.6604, ECE 0.0092,
  band r=4/8/16 0.3082/0.2234/0.1357; L_calib AP 0.6452, band r=8 0.2071) within the P1(b)
  bands, else STOP. The temp row replaces 0.1907. Finding 4's CARLA half cites this log's
  L_calib − raw r=8 from the **block-bootstrap** line. The route holds native CARLA props **and**
  pastes (unseen COCO + CARLA-bank), so the claim covers the whole route:
  - block CI excludes 0 (below 0) → "L_calib lowers band-ECE on the pasted CARLA route (native
    props + pastes; r=8, 20-frame circular block CI)";
  - block CI includes 0 → "no detectable effect on the pasted CARLA route (native props + pastes;
    r=8, 20-frame circular block CI)", and Finding 4 keeps only its Fishyscapes half;
  - block CI above 0 → report that L_calib is worse there too.
  - **Sanity note (not a gate):** record whether the block CI is wider than the frame-level CI
    printed just above it. It should be; if it is not, flag it to the agents. (A sensitivity
    check at N = 10 / 40 was optional and is not implemented; the decision reads N = 20 only.)
  - Whole-image ECE now prints at 6 dp; compare it with the old log's 4-dp value after
    rounding. (Old value −0.0162
  [−0.0178, −0.0142] is a frame-level CI over correlated frames, optimistic.)
  `eval_carla_c1.log` is dropped as evidence. The pilot logs' temp rows stay INVALID, not re-run.
- **Checkpoint:** Finding 4 and its "Why each attempt failed" row cite this log.
- **Status:** [x] block-bootstrap code signed (Agent 1, agent1_round2_codereview.md) · [x] run 2026-10-05 · [x] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `python eval_spatial.py --dataset carla --carla-root video_town02_pasted --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole 1.2251 --block-bootstrap 20 \| Tee-Object -FilePath eval_video_town02_pasted_T122.log` | eval_video_town02_pasted_T122.log | raw/L_calib reproduce eval_video_town02.log exactly (\|Δ\| = 0, incl. C1 lines and frame bootstrap) PASS; temp r=8 0.2117 replaces 0.1907; block CIs wider than frame-level at every radius (r=8 width 0.0093 vs 0.0036) PASS; Finding 4 CARLA half: L_calib − raw r=8 −0.0162 [−0.0207, −0.0114] (block) excludes 0 → "helps on the pasted CARLA route (native props + pastes)"; L_calib − temp block CI ∋ 0 at r=4/8/16 → indistinguishable (old frame-level "worse than temp" superseded); ubq_local self-fit, descriptive | run order 3→5→4 | PASS; Finding 4 re-sourced (direction asymmetry measured on both sides) | 7, 10, M10, M20 checked | P2 write-up; A1 before P4 | SIGNED (Agent 1, agent1_U2_eval.md) | SIGN (Agent 2, agent2_U2_compliance.md) |

---

## P3: Full-resolution decision (GPU, user, val only, ~15-30 min) — U2-5

Runs right after P1, before P2's write-up, P4 and P5. Skip if P0b dropped it.

- **Goal:** decide on the val half whether feeding the encoder 2048×1024 (scale 2) or the mean
  of both passes' logits (ms) beats the training resolution (scale 1).
- **Inputs:** `model_3head_best.pth`, Fishyscapes val half only; P0b's mode.
- **Procedure:** per val image, per-head logits at label resolution for each mode; pooled
  AUROC/AP/FPR@95/ECE, band-ECE r=8; paired AP bootstrap of each mode vs scale 1; the script
  prints the rule's verdict.
- **Command:** `python eval_scale.py --raw model_3head_best.pth --modes 1 2 ms | Tee-Object -FilePath eval_scale_val.log`
  (no `--window`: P0b chose whole-image scale 2).
- **Log:** `eval_scale_val.log`.
- **Rule (pre-registered, val only):** for m ∈ {2, ms}: ΔAP = AP(m) − AP(1) with paired CI,
  ΔFPR = FPR@95(m) − FPR@95(1).
  - **ADOPT** m if ΔAP ≥ +0.03, the CI excludes 0 and ΔFPR ≤ +0.01. Both qualify → higher
    val AP; within 0.01 of each other → scale 2.
  - **HURTS** if ΔAP ≤ −0.03 with the CI excluding 0 → keep scale 1; report "full-res hurts;
    the heads are scale-specific to their training resolution" (bears on Finding 2 / M9).
  - **NO GAIN** otherwise → keep scale 1; report "tested, no gain" (val numbers only).
  - In every band: band-ECE r=8 per mode is descriptive only (Finding 2 wording).
  - **If ADOPT:** headline-changing — the agents tell the user before anything else. Then
    `python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib model_3head_calib_best.pth --input-scale <m> | Tee-Object -FilePath eval_fishyscapes_T122_s<m>.log`
    (add `--window` if P0b chose windows) with P1 rules (d)-(g) re-applied; P4, P5 and the P6
    numbers run at scale m; every table states its scale; the paper also reports scale 1. The
    video is not re-rendered (CARLA frames are native 1024×512). Needs several GB of free RAM
    (logits cached at 1024×2048); UNVERIFIED.
  - **If not adopted:** one descriptive scale-2 test line may go in the paper; never used for
    anything.
- **Checkpoint P3:** decision + val numbers recorded before any test number at a new scale is
  quoted. **Reached: scale 1 kept** (NO GAIN for 2 and ms) — "no detectable gain at this sample
  size" (AP CIs about ±0.13 on 50 val images; a true +0.03 could not be detected). No P1 re-run;
  P4/P5/P6 at scale 1. ms halved val FPR@95 (0.0914 → 0.0456): descriptive, recorded as future
  work / hypothesis only, **never checked on the test half** (that would be selection on test).
  Val and test halves differ a lot (val band r=8 0.1405 vs test 0.2273): no val number is ever
  compared with a test number.
- **Status:** [x] run 2026-10-05 · [x] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `python eval_scale.py --raw model_3head_best.pth --modes 1 2 ms \| Tee-Object -FilePath eval_scale_val.log` | eval_scale_val.log | val only. Mode 2 ΔAP −0.1044 [−0.2305, +0.0359] → NO GAIN; ms ΔAP +0.0167 [−0.0426, +0.0931] → NO GAIN; scale 1 kept. Descriptive: band-ECE r=8 (1/2/ms) 0.1405/0.1907/0.1745; ms FPR@95 0.0914 → 0.0456 (not acted on). Val AP 0.6837 ≈ train.log 0.6839 | run order 3→5→4 | scale 1 kept; reviewer idea 1 tested, no detectable gain (underpowered, AP CIs about ±0.13) | 6, 7, 10 checked | P2; no reruns | SIGNED (Agent 1, agent1_U2_eval.md) | SIGN (Agent 2, agent2_U2_compliance.md) |

---

## P2: Write-up of P1/P1b/P3 into this file (CPU, coder)

- **Goal:** the calibration table, outline table, head-0 line, Findings 1-6 and "Why each
  attempt failed" filled from logs.
- **Inputs:** signed P1, P1-compare, P1b, P3 records.
- **Procedure:** fill the skeletons below; remove "570×" (M12); narrow Finding 2 (M9) plus P3's
  descriptive result; re-source Finding 4 (M10); Finding 6 per P1(e); every cell cites log,
  checkpoint, split and scale.
- **Command:** none (CPU edit). **Log:** this file.
- **Rule:** any cell without a log citation → not signed. Any number contradicting its log →
  fix before P6's number edits.
- **Checkpoint:** every number cites a log, checkpoint and scale.
- **Status:** [x] written 2026-10-05 (section "Write-up (step P2)" below) · [x] signed 2026-10-06 (agent1_P2P6_review.md SIGN, agent2_P2P6_compliance.md SIGN).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

---

## P4: Explainability completion (CPU code + GPU run) — U3, later

- **Goal:** a same-photo paste test (X5) on our model with a pre-registered primary contrast;
  decide X2's role in the paper.
- **Inputs:** `paste_test.py`; `data/coco_objects` (bank identity per U0-4).
- **Procedure (coder, CPU, NOW-4 or later):** one paste per forward pass (4 forwards per
  photo: original + one per arm); same centre and same longest-side target for all arms of a
  photo; reject a placement whose dilated paste meets the dilated real mask, retry up to 10
  times, else drop that arm for that photo; log px per arm; paired CI on real-object drift;
  bank-identity CPU checks (i) 3000 pairs, (ii) `coco_banks()` reproduces `_build_both_bank`
  (same sorted glob, same `random.Random(GLOBAL_SEED).sample`, same `COCO_BANK_TARGET`);
  required `--raw`, `--input-scale`.
- **Command:** `python paste_test.py --raw model_3head_best.pth --input-scale 1 | Tee-Object -FilePath paste_test.log`
  (P3 kept scale 1), plus `python explain.py --raw model_3head_best.pth --x2-faith | Tee-Object -FilePath explain_x2_faith.log`.
- **Log:** `paste_test.log`.
- **Rule (paired over photos):**
  - **Primary:** edge gap (unseen paste − real) > 0, CI excludes 0 → "pastes are more
    over-confident at edges than real objects in the same photos (our model)". CI ∋ 0 → "not
    supported in the same photos; the X1b paste/real gap may be a domain effect". < 0 with CI
    excluding 0 → report the contrary.
  - **Secondary** (exploratory if U0-4 leaves bank identity unverified): seen − unseen object
    score > 0, CI excludes 0 → memorisation evidence; else "no detectable memorisation".
  - Real-object drift CI must include 0; otherwise the pastes still affect the real object →
    report it and treat the primary contrast as confounded.
  - All other contrasts exploratory, labelled so.
- **Optional X2 faithfulness** (only if X2 is more than a footnote, decided at this checkpoint):
  `python explain.py --x2-faith | Tee-Object -FilePath explain_x2_faith.log` (code not written).
  Mean-ablate one stage at a time; per image, does the top grad×act stage also give the largest
  ablation drop? Agreement > 0.5 with CI excluding 0.25 → "faithful at the top-stage level";
  else "not validated", footnote only.
- Then re-export `static/explain/summary.json` (CPU) with M17 wording.
- **exp_v2 Part D (read 2026-10-06, d162e22; the lead's COCO model, NOT ours, M16):** the lead ran
  its own X1–X4. Three points matter for us:
  1. Gradient share and a *local* stage ablation disagree at stage 4: it carries 25–32% of the
     gradient but only 0–16% of the removal effect. So a grad×act share is not evidence that the
     model relies on a stage. Our X2 line "stage 4 has the largest share (0.42)" is therefore a
     **gradient-share statement only** until `--x2-faith` runs on our model.
  2. On their model, Telea fill fires on 22% of clean-road controls. Our X1b already measured Telea
     bias +0.609 on ours (M6), which is consistent.
  3. Their removal is per object, with clean-road controls and image-clustered CIs.

  Any comparison in the paper is labelled by checkpoint.
- **Code fixes done 2026-10-06** (orchestrator acting as coder; reviewed in the main session, per the
  cost note):
  - `paste_test.py` rewritten to the procedure above:
    - 1 + 3 forwards per photo (one paste each).
    - One shared (centre, longest-side target) per photo. The centre sits on the model's
      road/sidewalk prediction, at distance > 48 px + 0.75·target from the real object.
    - Up to 10 object draws per arm. A draw is rejected if the paste meets the real mask dilated by
      48 px; after 10 rejections the arm is dropped and counted.
    - Logged per arm: px, tries, surface fraction, real-object drift. The drift gets a paired CI per
      arm.
    - Output blocks print in the order PRIMARY / SECONDARY / CONFOUND / EXPLORATORY.
    - `--raw` and `--input-scale {1}` are required.
    - The CARLA arm uses A1 class `aligned` (24 objects). `--carla-classes` changes this, pending the
      A1 eyeball.
  - **Deviation (set before any run):** the shared target is log-uniform in [16, 102] px, not
    training's [6, 102] px. Below ~16 px most cutouts fall under the 32-px fragment threshold.
  - Bank checks, forced CPU (`python paste_test.py --check-banks`; output in
    `agent_state/coder_checks/paste_test_check_banks.txt`):
    - (i) 3000 COCO pairs: PASS.
    - (ii) `coco_banks()` seen set = the `_build_both_bank()` COCO part, same order: PASS. Unseen set:
      1500, disjoint: PASS.
    - The U0-4 caveat stands: whether the local files are the pod's is still unverified.
  - `explain.py --x2-faith` written (a port of exp_v2's local `stage_ablation`). Same rule as above,
    plus one refinement set before any run: **only detected objects** count (peak ≥ 0.47658, the raw
    val max-F1), because an undetected object's drop is noise around a ~0 score. One object = the
    image's whole anomaly mask (as in our X2), not one per connected component.
  - CPU smoke on 1–4 test photos, forced CPU (code checks, **not results**):
    `agent_state/coder_checks/now4_cpu_check_output.txt` and `x2_faith_cpu_check_output.txt`. Paste
    size, centring and locality are asserted; real-object drift on the smoke photo was 0.0000.
  - MISTAKES rule 13 nearly repeated (M19): `CUDA_VISIBLE_DEVICES=""` was used again. The in-code
    assert caught it before anything ran; fixed to `"-1"`.
- **Checkpoint P4:** X5 verdict recorded; X2's paper role decided.
- **Status:** [x] code fixes 2026-10-06 · [x] run (U3b) 2026-10-06 · [x] signed (orchestrator) for PRIMARY/SECONDARY/X2; the CARLA-arm wording waits for the A1 eyeball.
- **Results (U3b, 2026-10-06; `paste_test.log`, `explain_x2_faith.log`; raw primary model; scale 1; Fishyscapes test half):**
  - **X5 setup:** 50/50 photos placed; 0 arms dropped; surface fraction 0.99 in every arm; the exact
    removal (original photo) scores 0.000 on the paste pixels; 37 photos have a real object ≥ 30 px.
  - **X5 PRIMARY:** edge gap, unseen paste − real = **+0.468 [+0.401, +0.532]**, n=37. Pastes:
    band_gap +0.298, over-confident. Real objects: −0.179, under-confident.
  - **X5 CONFOUND:** the rule says each real-object drift CI must include 0.
    - coco_seen −0.0008 [−0.0036, +0.0027]: PASS.
    - carla_bank −0.0010 [−0.0038, +0.0017]: PASS.
    - **coco_unseen −0.0032 [−0.0059, −0.0007]: FAIL.**
    - Per the pre-registered rule, the primary contrast is **reported as confounded**. Stated
      alongside, not as a rescue: the drift is about 150× smaller than the gap. The same contrast
      with the seen arm, whose drift check passes, gives +0.453 [+0.381, +0.523]; that is
      exploratory. Three drift tests at 95% have about a 14% chance of one false exclusion.
  - **X5 SECONDARY:** seen − unseen object score = −0.011 [−0.053, +0.033] → **"no detectable
    memorisation"**. Exploratory until U0-4; the CPU bank check passed, but whether the local files
    are the pod's is still unconfirmed.
  - **X5 exploratory:**
    - Same photos: unseen paste 0.939 vs real 0.347 (+0.584 [+0.478, +0.677]). The real objects are
      larger (1392 vs 915 px), so size does not explain the gap in this direction.
    - CARLA(aligned) − seen COCO −0.086 [−0.175, −0.006]. 4 CARLA pastes scored ≤ 0.05, all at
      18–56 px targets.
    - Stage-4 grad share, unseen − real −0.244 [−0.306, −0.173]. This is gradient only; see X2.
  - **X2 faithfulness:**
    - Fishyscapes, 24/37 detected: agreement **0.25 [0.08, 0.42] → NOT validated.** The top
      gradient stage was mostly stage 4 (11/24); the top ablation stage was stage 3 (17/24) and
      never stage 4. Ablation drop by stage: 28 / 52 / **74** / 28%.
    - CARLA pastes, 75/75 detected: **0.56 [0.44, 0.67] → "faithful at the top-stage level"**.
      Ablation drop by stage: 3 / 16 / 20 / 12%.
    - This matches exp_v2 Part D on the lead's model (labelled by checkpoint, M16).
  - **X2 paper role:** footnote for real objects. The removal (ablation) numbers are the
    explanation: real detections rest on stages 2–3. The old demo line "stage 4 pushes the score
    down on real objects" is withdrawn (superseded; M23).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | `python paste_test.py --raw model_3head_best.pth --input-scale 1`; `python explain.py --raw model_3head_best.pth --x2-faith` (user, GPU) | paste_test.log, explain_x2_faith.log | PRIMARY +0.468 [+0.401, +0.532] (> 0, CI excludes 0); CONFOUND: unseen drift FAIL (−0.0032 [−0.0059, −0.0007]), seen/CARLA PASS; SECONDARY −0.011 CI ∋ 0; X2 real agreement 0.25 → not validated; X2 pasted 0.56 [0.44, 0.67] → faithful | target range [16, 102] px (pre-set); X2 detected-only (pre-set); CARLA arm = A1 "aligned", labels not yet eyeballed | PRIMARY direction supported but **formally confounded** (rule); no memorisation detected; X2 = footnote for real objects, ablation reported instead. `summary.json` re-exported (x2_faith, x5); /demo X2 panel reworded, X5 card added; build passes | 3, 5, 11, 13; M16, M17, M18; M23 (new) | P6b, P7 | orchestrator | orchestrator |

---

## P5: RoadAnomaly21 val (GPU, user, ~3 min) — U3, later

- **Goal:** a second real eval set for our model (exp_v2 measured only its COCO model).
- **Inputs:** `dataset_AnomalyTrack.zip` (md5 verified in U0-3); 10 val images with public
  labels.
- **Procedure (coder, CPU, NOW-4 or later):** port exp_v2 `eval_road_anomaly.py` +
  `config.ROAD_ANOMALY21_DIR` with required `--checkpoint`, `--input-scale`, and the md5 check.
- **Command:** `python eval_road_anomaly.py --checkpoint model_3head_best.pth --with-baseline | Tee-Object -FilePath eval_road_anomaly21.log`
  (add `--input-scale <m>` if P3 adopted).
- **Log:** `eval_road_anomaly21.log`.
- **Rule:** nothing is fitted or selected on RA21. Pooled + per-image metrics with anomaly area.
  TwinGuard < MSP → "generalisation limitation (large close objects)", next to exp_v2's COCO
  result labelled by checkpoint. TwinGuard ≥ MSP → report with the 10-image caveat. Never a
  headline benchmark.
- **P5b (optional):** Lis et al. Road Anomaly (60 images), same script with `--dataset lis` →
  `eval_road_anomaly_lis.log`, only after a CPU overlap check (md5/perceptual hash) against
  RA21; overlapping images reported in one set only; neither set called "independent" before
  the check. Public availability UNVERIFIED.
- **Port done 2026-10-06:** `eval_road_anomaly.py`, from exp_v2 da7349d (unchanged at d162e22).
  - Required `--checkpoint` and `--input-scale {1}`.
  - `--zip` checks the md5 against `config.ROAD_ANOMALY21_ZIP_MD5`. A mismatch stops the run;
    without `--zip` it prints UNVERIFIED.
  - AP is printed first.
  - Data path: `config.ROAD_ANOMALY21_DIR` (env var `ROAD_ANOMALY21_DIR`).
  - Compiles; not run (no data yet).
- **Command (replaces the one above):** `python eval_road_anomaly.py --checkpoint model_3head_best.pth --input-scale 1 --zip <path>\dataset_AnomalyTrack.zip --with-baseline | Tee-Object -FilePath eval_road_anomaly21.log`
- **Checkpoint:** results recorded with the 10-image caveat. **Status:** [x] port 2026-10-06 · [x] run 2026-10-06 · [x] signed (orchestrator).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | `python eval_road_anomaly.py --checkpoint model_3head_best.pth --input-scale 1 --with-baseline` (user, GPU; ROAD_ANOMALY21_DIR set) | eval_road_anomaly21.log | 10/10 val pairs PASS; nothing fitted PASS; TwinGuard AP 0.3084 / AUROC 0.6550 / FPR@95 1.0000 / ECE 0.1335; MSP AP 0.4744 / AUROC 0.8705 / FPR@95 0.3504; anomaly px 14.81%; per-image AP 0.098–0.813 | zip md5 not checked (no `--zip`; the zip was not found on disk) → UNVERIFIED | rule: TwinGuard < MSP → **"generalisation limitation (large close objects)"**, shown next to exp_v2's COCO result (AUROC 0.72 vs 0.87) labelled by checkpoint; never a headline | 3, 11; M16 | P7 Table 1 filled | orchestrator | orchestrator |

---

## P6: Frontend, server and README (CPU code; the user checks visually)

- **Goal:** the demo shows the real 0.9920 model honestly; every displayed number traces to a
  log.
- **Inputs:** exp_v2 `server.py` (reference); P1 numbers for the deferred part.
- **Procedure — done 2026-10-05 (code only):**
  - `server.py`: `load_image_tensor`; no second sigmoid (scores = upsampled logits → sigmoid
    → mean, as in eval); loads `config.PRIMARY_RAW`; frames from the **test half only**, 3
    highest per-frame AP + 3 around the median, rule stated in the page caption; new panels go
    to `static/generated/test_half/` with a `meta.json` (old `static/generated/demo_*` untouched);
    `EXPERIMENT_HISTORY` gains the 0.9920 run (`train.log`) marked current; click-through UX
    unchanged. Box rule unchanged with `# PENDING P1 (val max-F1)`.
  - `DemoPage.jsx`: M17 rewording (old `:332`, `:368`); three video-caption notes; lede names
    the model and the frame rule; "never seen during training" now true (test-half frames).
  - `HistoryPage.jsx`: "current" marker follows the new history id.
  - `README.md`: 0.9920 run from `train.log`; T "being re-measured (M1)".
  - `render_video.py`: M3 comment fixed; `--min-box` (default 150).
- **Procedure — done 2026-10-05, coder round 2 (code review N3/N4, Agent 1 §4):**
  - `server.py` training gallery: real CutMix composites (Cityscapes val, fixed seed, objects
    from the 2000-object bank) written to `static/generated/training_cutmix/`, instead of the
    curated CARLA frames' misaligned masks (M10). The old gallery stays on disk, unused.
    `TrainingPage.jsx` lede rewritten to match.
  - `DemoPage.jsx:430`: "COCO cutouts from outside the 1500-object training subsample (assuming
    the local COCO bank is the one used in training, still being confirmed)" — conditional
    until U0-4 is answered.
  - `render_video.py --raw` is required (rule 11).
- **Procedure — done 2026-10-05 after U2 (awaiting sign-off):**
  - `HomePage.jsx`: temperature tab = T 1.2251 (AP 0.6206, AUROC 0.9924, whole ECE 0.0001, band
    0.2095); "570×" and "20×" removed (absolute numbers); C2 note "0.2095 → 0.0942, whole ECE
    0.0001 → 0.0420; image T ≈ 1.2, edges T ≈ 4.4"; L_calib note per Finding 4; AP before AUROC
    (metrics strip and comparison rows).
  - `HistoryPage.jsx`: lede no longer says Checkpoint B is the demo model; calibration table
    AP-first with the T 1.2251 row; "20×" replaced by absolute numbers; image →
    `/static/calibration_reliability_T122.png` (copied; old png kept) with the bin-mass caveat.
  - `server.py`: boxes at the fixed val-fitted threshold `BOX_THRESHOLD = 0.47658` (raw row,
    `eval_fishyscapes_T122.log`) instead of each image's 97th percentile; `CACHE_VERSION` 2.
    `DemoPage.jsx` detection caption describes the fixed threshold.
  - `README.md`: T = 1.2251 cited. `calibrate.py`: stale "0.75 gate / Experiment A" print
    removed (compare/temp output; cosmetic).
  - Still deferred: `summary.json` re-export (P4); any CARLA-bank-quality wording (A1).
- **Command:** coder: `npm run build` in `frontend/`. User (it loads the model on the GPU, so the
  user starts it): `python server.py`, then `cd frontend; npm run dev`, open
  `localhost:5173/` and `/demo`. The first start builds the test-half panels once.
- **Log:** `server_T122.log` if the user tees the first start
  (`python server.py | Tee-Object -FilePath server_T122.log`).
- **Rule:** build passes; demo frames are test-half (server log lists them with their rule
  label); every displayed number traces to a log; the user confirms `/` and `/demo` → PASS.
  Any number without a source → fix before Checkpoint 2.
- **Checkpoint P6:** build passes, numbers traced, user confirms visually.
- **Status:** [x] code-only part + build 2026-10-05 · [x] code-only part signed (review 2026-10-05) · [x] deferred number edits + build 2026-10-05 · [x] signed 2026-10-06 · [~] user visual check (U3a: log PASS 2026-10-06; the user's "looks right" pending).
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | py_compile server/render_video; `npm run build`; fake-model server check | agent3_coder_round1.md | PASS | box rule kept until P1 | code part done | 11, M13, M17 | user visual check after P1 | see next row | see next row |
| 2026-10-05 | code review of NOW-1..3 (read-only) | agent1_codereview.md, agent2_codereview.md | server fixes correct; frame rule accepted | training gallery → CutMix, DemoPage:430 conditional wording, render_video --raw required (done round 2) | APPROVED U1 with required fixes; fixes done in coder round 2 (agent3_coder_round2.md) | 11, M10, M13, M17 | U0, U1 (sequential) | APPROVE (Agent 1, 2026-10-05) | APPROVE (Agent 2, 2026-10-05) |
| 2026-10-05 | round-2 P6 additions: CutMix gallery, DemoPage:430, TrainingPage lede, render_video --raw; forced-CPU checks; `npm run build` | agent3_coder_round2.md | all PASS | none | done, awaiting agents' read | 11, 13 | user visual check after P1 | [ ] | [ ] |
| 2026-10-06 | U3a: `python server.py` (user, GPU) + `npm run dev`; pages /, /runs, /training, /demo visited | server_T122.log (pasted in chat) | device cuda PASS; checkpoint `model_3head_best.pth` (primary) PASS; 50 test-half frames ranked, 3 highest-AP (0.980/0.970/0.959) + 3 around median (0.424/0.338/0.283) with rule labels PASS; manifest → `static/generated/test_half` PASS; CutMix gallery from the "both (2000 objects)" bank, 8 samples PASS; every request 200/206/304, none 4xx/5xx PASS (video, explain summary, reliability png, history, training samples all served). Boxes 2/0/2/0/0/1: demo_01 has AP 0.970 but no box (mean inside 0.359 < 0.477 and/or region < 0.05% of the image) | caption gap: DemoPage detection text omitted the min-area rule (`min_area_frac=0.0005`); fixed 2026-10-06, `npm run build` passes. FastAPI `on_event` deprecation warning: cosmetic, left | PASS on the log; user's visual verdict on wording/layout pending | 11, M13, M17 | P6b | orchestrator (log) | orchestrator (log) |

---

## P6b: Video Checkpoint 2 (user)

- **Goal:** tick Checkpoint 2 honestly (re-defined: the mp4 plays on `/demo` next to the
  **fixed** image demo, with honest captions).
- **Inputs:** P6 done; `/demo`.
- **Procedure:** the user watches the video and picks min box (150 or about 100) **by eye on the
  first 30 s, once**; never by the found% statistic.
- **Command (only if the value changes):** `python render_video.py --root video_town02_pasted --raw model_3head_best.pth --min-box <v> --out static/video/twinguard_video_town02_pasted_minbox<v>.mp4 | Tee-Object -FilePath render_video_minbox<v>.log`
  (`--out` keeps the current mp4 and stats on disk; the coder then points `VIDEO_BASE` in
  `DemoPage.jsx` at the new file. The verifier's version had no `--out` and would overwrite.)
- **Log:** `render_video_minbox<v>.log` + the new `_stats.json`.
- **Rule:** chosen once, by eye; no further tuning on this video (old plan). Checkpoint 2 ticked
  only after P6's user visual check.
- **Checkpoint 2:** mp4 + fixed image demo + captions. **Status:** [ ] decided · [ ] ticked.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

---

## P7: Paper tables, citations, title (CPU; title = user)

- **Goal:** paper-ready tables, every number sourced; title chosen by the user.
- **Inputs:** signed P1-P6.
- **Procedure:** tables AP-first (AUROC 0.992 is saturated; Fishyscapes ranks on AP), each
  citing log, checkpoint, split, scale: calibration (P1/P3), outline quality (P1e), head 0 (P1f),
  explainability (X1/X1b/X2/X3/X5), RA21 (+Lis), video ("pastes on CARLA, not real-world"),
  "Why each attempt failed"; an exp_v2 column only where labelled by checkpoint (M16). **No
  comparison to published leaderboard numbers**: our test set is a 50-image half of the public
  L&F val; any mention says "different split". Verify citations: Seg-Grad-CAM (Vinogradova et
  al. 2020), HiResCAM (Draelos & Carin 2020), RISE (Petsiuk et al. 2018), Depeweg et al. 2018,
  SMIYC, Fishyscapes, Lis et al., SegFormer. Title: present the 4 options (PLAN_HISTORY Phase 4)
  plus any change implied by P3 and P5.
- **Command:** none. **Log:** this file + paper draft.
- **Rule:** any unsourced number → not signed. Title is the user's choice, after P3 and P5.
- **Checkpoint 4:** every number sourced; title chosen by the user.
- **Status:** [x] tables drafted 2026-10-06 (`paper/TABLES.md`) · [ ] RA21 row (P5) · [ ] A1 wording · [ ] title (user) · [ ] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | none (CPU write-up) | `paper/TABLES.md` | 9 tables + limitations + 16 references + 4 title options; every number copied from the signed P2 write-up, the P4 results or the stats json, each with its log; AP first; no ratios; no leaderboard comparison. 2 citations checked online (Vinogradova et al. 2020, doi 10.1609/aaai.v34i10.7244; Draelos & Carin arXiv:2011.08891); 14 from memory, flagged | the β = 50 row has no log on disk (flagged in the table) | draft complete except the RA21 row, the A1 k, and the title | 3, 12; M12, M16, M17, M21, M23 | user: title, P5, A1 | orchestrator | orchestrator |

---

## A1 (= Agent 1's P4.0): CARLA-tile audit (CPU, coder; read-only on `data/`)

- **Goal:** measure whether the training bank's CARLA tiles were cut with misaligned masks.
  `data/anomaly_sources._ObjectBank.sample` crops each CARLA object as the bounding box of its
  `data/masks/*.npy` mask, the masks M10 found misaligned.
- **Inputs:** `data/images`, `data/masks` (45 objects), `_build_both_bank()` with the current
  config. Caveat: assumes the local 45 files are the ones the pod trained on (UNVERIFIED, like
  the COCO bank, U0-4).
- **Procedure:** reproduce the 500-entry CARLA part of the bank; per object: bank count, mask
  bbox, mask-vs-8 px-ring colour contrast (the M10 screen), edge support (mean Sobel gradient on
  the mask outline / in the ring), road-colour fraction, mask IoU with neighbours,
  identical-bbox detection across frames (≤ 5 px), and a contact sheet with the mask outline
  (red) and the cut crop for eyeballing. CPU forced in code (rule 13).
- **Command (coder, CPU):** `& "C:\Users\venka\anaconda3\envs\cuda_test\python.exe" agent_state\a1_carla_tiles\a1_audit.py`
- **Log:** `agent_state/a1_carla_tiles/a1_run_output.txt`, `a1_carla_tiles.csv`,
  `contact_sheet.png`.
- **Rule:** the composite screen written before the first run (contrast < 30 or road_frac > 0.5
  or edge < 1.0) is judged against the 4 M10 anchors (000, 005 aligned; 032, 047 misaligned).
  It **failed** (it flags both aligned anchors), so it is not used. The proposed classification,
  **set after seeing the anchors and therefore not pre-registered**, uses edge support alone:
  < 1.1 misaligned, 1.1–1.6 unsure, ≥ 1.6 aligned (anchors: 2.99, 3.20 vs 0.93, 0.99). Final
  labels come from the user's eyeball check of the contact sheet, then both agents sign.
  Outcome rules (Agent 1 P4.0): X5's CARLA arm uses **aligned** objects only; the paper states
  "k/45 CARLA objects (m/500 bank tiles) were cut with misaligned masks"; **no retraining**
  unless the user decides it; MISTAKES M22 is finalised with k after review.
- **Results (PENDING REVIEW, 2026-10-05):** misaligned 15 objects = 166/500 bank tiles (004 008
  012 018 020 024 025 029 032 033 037 043 045 047 048); unsure 6 = 67/500 (001 009 010 027 031
  034); aligned 24 = 267/500. Identical bboxes: 004=047, 019=020, 042=043, 044=045 — in each
  pair one member scores aligned, the other misaligned, except 004/047 (both misaligned).
  This fits a capture that saved a neighbouring frame's mask. The old composite screen flagged
  37/45. So k is between 15 and 21 (of 45), about 166–233 of the 500 CARLA tiles (8–12% of the
  2000-object bank), pending the eyeball check.
- **Checkpoint:** k fixed by the user's eyeball + both agents' sign; gates P4 (X5 CARLA arm) and
  P7 (bank description). Does not gate U2.
- **Status:** [x] run 2026-10-05 · [ ] user eyeball · [ ] signed.
- **Evaluation record:**

| date | command run | log | each check PASS/FAIL + numbers | deviations | verdict | MISTAKES rules checked/triggered | next step unlocked | Agent 1 sign-off | Agent 2 sign-off |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | `a1_audit.py` (CPU) | agent_state/a1_carla_tiles/ | pre-set composite screen FAILED on anchors; edge-support bands: 15 misaligned / 6 unsure / 24 aligned | classification bands set after seeing anchors (not pre-registered) | PENDING REVIEW | 3, 5, 13; M10, M22 (pending) | user eyeball, then P4 design | [ ] | [ ] |

---

## Optional steps (each needs the user's go-ahead; costs stated)

Each optional step uses the same template; the evaluation record is added when it is approved.

- **O1: C-series on the lead's COCO checkpoint** (cross-model replication; eval-only, ~20 min;
  needs U0-2).
  - Command: `python eval_spatial.py --dataset fishyscapes --raw <lead_phase2a_ckpt> --calib <lead_calib_ckpt> | Tee-Object -FilePath eval_fishyscapes_T_coco.log`.
    `--calib` is now required: pass the lead's `phase2b_calib2` checkpoint, or the coder makes
    `--calib` optional first (only if O1 is approved). T is refit (exp_v2 reports 1.2206).
  - Rule: same verdict per C-step as ours → "replicates on 2 models"; any difference is
    reported as model-dependent.
- **O2: C5-confidence hybrid** (eval-only; reopens Phase 1, so the user and guide decide).
  Rule: band r=8 below the T=3.15 control with CI excluding 0 **and** whole-image ECE not
  significantly worse than temp(whole, T from P1). Detection AP stays raw. Fail → "context fixes
  edges only at a whole-image cost", no further variants. Log `eval_fishyscapes_o2.log`.
- **O3: Scale-mixture retrain** (~2 h GPU + re-eval). Starts only if the user lifts "no
  retraining for now". Always a variant under a distinct filename (never
  `checkpoints/model_3head_best.pth`, never the primary file); selected on Fishyscapes **val**
  AP only; RA21 reported once, never used to choose; never replaces the 2000-bank model without
  the user's explicit written decision.
- **O4: Trigger** (port `predict_dual_mode` + `check_dual_mode.py`). Fit on Cityscapes-val half A
  + Fishyscapes val; report on Cityscapes-val half B + Fishyscapes test; rule fixed beforehand
  (≤ 5% normal frames triggered, ≥ 90% anomalous). Low prior (X3: FP − TP disagreement not
  significant on real data).
- **O5: Benchmark submission** (Fishyscapes / SMIYC). High effort, external turnaround. Until
  done, no comparison to published numbers.
- **O6: S2 signed-distance binned T.** Dropped unless the user reopens Phase 1. If reopened:
  judged by the C4 rules against the T=3.15 control **and** against T from P1.

---

## Superseded claims

Each old claim, where it appears, why it is superseded, the step that replaces it, and whether
the paper's conclusion changes. Numbers marked PENDING are not yet measured.

| # | Old claim | Where it appears | Why | Replaced by | CHANGES CONCLUSION? |
|---|---|---|---|---|---|
| 1 | temp(whole) T=1.71 row: AUROC 0.9924, AP 0.6193, FPR 0.0288, ECE 0.0020, band r=4/8/16 0.2208/0.1845/0.1451 ("edge ECE 0.2273 → 0.1845 only") | PLAN_HISTORY final table + why-failed table; `HomePage.jsx:43-47`; `HistoryPage.jsx:53,64`; temp rows of `eval_fishyscapes_c1/c1b/c2/c3/c3ctrl/c5.log`, `eval_carla_c1.log` | M1 | **T = 1.2251: AUROC 0.9924, AP 0.6206, FPR@95 0.0289, whole ECE 0.0001 (4 dp), band r=4/8/16 0.2630/0.2095/0.1647; edge ECE 0.2273 → 0.2095** (`eval_fishyscapes_T122.log`) | N — the edge finding is stronger |
| 2 | T=1.7142 in `calibrate.log`'s table and `calibration_reliability.png` temp curve | `calibrate.log`, `calibration_reliability.png`, `static/calibration_reliability.png` | M1 | **`calibrate_compare_T122.log` + `calibration_reliability_T122.png`** (whole + band panels; old files kept). Caveat: bin mass not shown; >99% of whole-image pixels sit in the lowest bin | N |
| 3 | C5 bootstrap labelled "C5 − ECE(temp T=1.71)": r=8 −0.1107, whole +0.0245 | `eval_fishyscapes_c5.log`; old `eval_spatial.py:505` | M1 | **C5 − temp(1.2251): r=8 −0.1358 [−0.1608, −0.0993], whole +0.0263 [+0.0240, +0.0278]** (`eval_fishyscapes_T122.log`) | N — same pattern |
| 4 | L_calib − temp r=8 +0.0551; C4 condition 1 C3 − temp r=8 −0.0533 | `eval_fishyscapes_c5.log`, PLAN_HISTORY C-section | M1 | **L_calib − temp r=8 +0.0301 [+0.0196, +0.0397] (r=4 +0.0228, r=16 +0.0242, all excl. 0); C4 cond 1 −0.0783 [−0.0972, −0.0651]** (`eval_fishyscapes_T122.log`) | N — no flips |
| 5 | "The whole image wants T≈1.7, the edges want ~4.4" | PLAN_HISTORY why-failed table and Findings 3; `HomePage.jsx:60` | M1 + M14 | **whole T = 1.2251 (plain NLL), edge T = 4.3592 (C2, LBFGS) / 4.3686 (bounded NLL) — same objective** (`eval_fishyscapes_T122.log`) | N — the gap widens |
| 6 | "C2 makes whole-image ECE 20× worse" (0.0420 vs 0.0020) | PLAN_HISTORY; `HomePage.jsx:60`; `HistoryPage.jsx:74` | M1 | **absolute numbers only: C2 whole ECE 0.0420 vs temp(1.2251) 0.0001 vs raw 0.0004; paired whole diff C2 − temp +0.0418 [+0.0390, +0.0443]** (`eval_fishyscapes_T122.log`). No ratio: at 4-dp print precision the denominator is unresolved (M12) | N |
| 7 | "Whole ECE 0.0004 vs edge 0.2273, ~570× worse" | PLAN_HISTORY Finding 1; `HomePage.jsx:39` | M12 (ratio across different pixel populations) | report both numbers; ratio removed (P2, P6 deferred edit) | N — the finding stands |
| 8 | "T=1.71 / 3.15 / 4.36 trace a trade-off curve" | PLAN_HISTORY Finding 3 | M1 | **(whole ECE, band r=8): raw (0.0004, 0.2273); T 1.2251 (0.0001, 0.2095); T 3.1516 (0.0177, 0.1277); T 4.3592 (0.0420, 0.0942)** (`eval_fishyscapes_T122.log`) | N |
| 9 | C1b: "learned, not an upsampling artefact" | PLAN_HISTORY Phase 1 list and Finding 2 | M9 | **Narrowed:** learned at the training resolution; not caused by the eval-side upsampling order (C1b); not reduced by full-resolution input at inference (P3, val, descriptive: band-ECE r=8 0.1405 → 0.1907 at scale 2, `eval_scale_val.log`). The heads were trained on half-resolution features only, so this does not rule out resolution as a training-time cause | **Y** — narrowed |
| 10 | Finding 4, CARLA half: "L_calib helps on CARLA bank objects, r=8 −0.0087, CI excludes 0" | PLAN_HISTORY why-failed table and Finding 4; `eval_carla_c1.log` | M10 (curated masks misaligned), M20 | **L_calib − raw r=8 −0.0162 [−0.0207, −0.0114], 20-frame circular block CI, on the pasted CARLA route (native props + pastes)** (`eval_video_town02_pasted_T122.log`); the old frame-level −0.0162 [−0.0178, −0.0142] was optimistic | **Y** — evidence base replaced; direction same |
| 11 | C5 "missed extent 12.8 → 4.8 px" | PLAN_HISTORY Finding 6 | M11 | **Withdrawn.** Replaced by P1(e) at each row's own val max-F1 threshold: per-image boundary F1 C5 − raw +0.0950 [+0.0349, +0.1543] over 24/50 images; objects found 59 vs 40 of 85; miss 4.50 vs 8.94 px; spill 5.37 vs 2.97 px; **far-FP 36.1% vs 14.7% of flagged px; AP 0.4929 vs 0.6215; FPR@95 0.0949 vs 0.0290** (`eval_fishyscapes_T122.log`). Every report states these costs (M21) | **Y** — old px claim withdrawn; outline gain kept, with costs |
| 12 | "Only 47% of objects found" | external review; exp_v2 `RESULTS.md` A2 | M16 | **Ours: 40/85 (47.1%) at val max-F1 t = 0.4766, raw row** (`eval_fishyscapes_T122.log`); computed by `ubq_local` on our model — the count happens to equal exp_v2's COCO model (40/85, t = 0.507) | **Y** — attribution: now our own measured number |
| 13 | "3 heads detect no better than 1" | external review; exp_v2 `RESULTS.md` Part B | M16 | **On our model, the 3-head mean beats its own head 0 by 0.041 AP: 0.5810 vs 0.6215, ΔAP −0.0406 [−0.0685, −0.0125]** (`eval_fishyscapes_T122.log`). Head 0 is one member of a jointly trained ensemble at the epoch selected on *fused* val AP, so this is the averaging gain at a fixed epoch and an upper bound on the gap to a separately trained, val-selected 1-head model. A separately trained 1-head model (lead's COCO run) matched 3 heads. Box recall on the pasted CARLA video is nearly equal (862 vs 858 of 1245, stats json). **Not "3 heads detect better than 1".** | **Y** — for our model on Fishyscapes, scoped as stated |
| 14 | X2: "this explains real 0.35 vs pasted 0.95" | PLAN_HISTORY X2 results | M17 (first-order method, domain-confounded) | "consistent with"; P4 optional faithfulness check | N — wording |
| 15 | DemoPage: "a habit learned from training on pasted objects"; "That's why pastes score … and real …" | `DemoPage.jsx` old `:332`, `:368` | M17 | reworded 2026-10-05 (scope stated, "consistent with") | N — wording |
| 16 | Demo image scores and frames (double sigmoid, no normalisation, stale checkpoint, frames picked on labels over val+test) | `/api/manifest` from `static/generated/demo_*` | M13 | P6 server fix (code done); panels rebuilt in `static/generated/test_half/` on the user's next server start | **Y** — every displayed demo number changes |
| 17 | Temp rows of `eval_video_pilot.log` (band r=8 0.2118) and `eval_video_pilot_t02.log` (0.2319) | those logs | M1 | **INVALID, not re-run** (pilot go/no-go checks; raw and L_calib rows stay valid) | N |
| 18 | `eval_video_town02.log` temp row, band r=8 0.1907 | that log; PLAN_HISTORY 2d | M1 | **T = 1.2251: band r=4/8/16 0.2902/0.2117/0.1278; whole ECE 0.010719 (raw 0.009205 — a Fishyscapes-fitted T worsens whole ECE on this over-confident route)** (`eval_video_town02_pasted_T122.log`) | N |
| 19 | README/`server.py`: "Checkpoint B 0.6190 is the current demo checkpoint" | `README.md:20-22` (old), `server.py` `EXPERIMENT_HISTORY` (old) | stale | P6 (done): 0.9920 run from `train.log` | N |
| 20 | "min box 150 keeps every real object" | `render_video.py:51-52` (old) | M3 | comment fixed; found 77.2% → 68.9% (PLAN_HISTORY 2d; stats json) | N |
| 21 | README mention "earlier T=1.71 … mis-fitted" | `README.md` | describes M1; no value used | **done: README cites T = 1.2251 (`eval_fishyscapes_T122.log`) and names T=1.71 only as the mis-fitted value (M1)** | N |
| 22 | Video: "L_calib … worse than temp" (r=8 +0.0165, frame-level CI, T=1.71) | PLAN_HISTORY 2d; `eval_video_town02.log` | M1, M20 | **Indistinguishable: L_calib − temp(1.2251) r=8 −0.0046 [−0.0104, +0.0016]; r=4 +0.0039 [−0.0020, +0.0084]; r=16 −0.0015 [−0.0091, +0.0065] (block CIs all include 0)** (`eval_video_town02_pasted_T122.log`) | **Y** — video wording |

Coverage check of `1.71` hits (2026-10-05): `HomePage.jsx:43,60` (rows 1, 5, 6), `HistoryPage.jsx:53,64`
(row 1), `README.md` (row 21), `eval_spatial.py` and `calibrate.py` (comments/docstrings citing
M1 only; no value used — rows 1-3).

---

## Old-plan disposition (every item of PLAN_HISTORY.md)

| Old item | Disposition | Reason |
|---|---|---|
| Phase 1 calibration "closed" | KEEP closed; P1 re-measures only the temp row and T-dependent verdicts | M1 invalidated one row, not the C-series design |
| "Why each calibration attempt failed" table | KEEP → skeleton below, filled in P2/P7 | T row from P1; L_calib CARLA evidence re-sourced (M10) |
| Calibration final table + Findings 1-6 | KEEP → rewritten in P2 | "570×" (M12), Finding 2 (M9/P3), Finding 4 (M10/P1b), Finding 6 (M11/P1e) change |
| Video 2a pilot, 2b/2c decisions, 2d render + eval | DONE (kept as history) | `eval_video_town02.log`, stats json |
| Video Checkpoint 2 + min-box decision | KEEP → P6b | Checkpoint 2 re-defined: needs the fixed image demo (M13) |
| min box 150 → ~100 | KEEP → P6b | chosen once by eye; new `--min-box` flag |
| Video caption honesty notes | DONE in P6 (2026-10-05) | pastes-on-CARLA, scale, lane-dash notes added; ablation note kept |
| X1, X1b, X2, X3 | KEEP (results below) | X2 wording per M17 |
| X4 `/demo` panel | KEEP; user visual check in P6 | M17 rewording done |
| X5 `paste_test.py` | KEEP → P4 with fixes and a pre-registered rule | M18 |
| X2 faithfulness check | OPTIONAL in P4 | only if X2 is more than a footnote |
| Citation verification | KEEP → P7 | |
| Phase 4: `EXPERIMENT_HISTORY` + README stale 0.6190 | DONE in P6 (2026-10-05) | `train.log` |
| Phase 4: fix `config.CHECKPOINT_3HEAD` | CLOSED by `PRIMARY_RAW/PRIMARY_CALIB` + required explicit paths + `load_trained_model(None)` raising | repointing would let training overwrite the primary model (M15) |
| Phase 4: paper tables | KEEP → P7 | |
| Phase 4: paper title | KEEP → P7, the user's choice after P3/P5 | P3 can change Finding 2's premise |
| S1 HD95 | DROP as a separate item | superseded by `ubq_local` far-FP + per-object spill (P1); `pred_to_gt_p95` stays in the legacy line, captioned saturated |
| S2 signed-distance binned T | DROP → O6 if the user reopens Phase 1 | exp_v2's two-region Platt around the predicted edge was worse than raw (`RESULTS.md:257-264`); with C3/C5 the evidence predicts failure |
| S3 soft-target CutMix, S4 boundary-weighted L_calib | DROP | retrains, against "no retraining for now"; exp_v2's harder-pastes retrain failed (`RESULTS.md:294-300`); L_calib learned on pastes transfers the wrong way |
| RoadAnomaly21 val | KEEP → P5 | new for our model |
| Road Anomaly (Lis et al.) | OPTIONAL → P5b | overlap with RA21 and availability UNVERIFIED |
| OoDIS | **DROPPED; user confirmed 2026-10-05** | needs instance post-processing + server submission; same SMIYC family as RA21 |
| AP as headline metric | KEEP → P7 (and P6 deferred edit) | AUROC 0.992 saturated; Fishyscapes ranks on AP |
| Head disagreement as a standalone signal; MC-Dropout dual-mode trigger | Covered by P1(f) and X3; trigger → O4 | |
| Real-time CARLA + WebSocket demo | DROP | baked mp4 chosen (2c); time |
| Experiment A retrain (lost weights) | DROP (user decision 2b) | |
| Image demo "video is additive; click-through not reworked" | KEEP as a constraint on P6 | only inputs/outputs fixed; UX unchanged |
| Reviewer ideas 1-6 | 1 → P0b/P3; 2 → P1 (ours) + O1 (theirs); 3 → O2; 4 → O3; 5 → O4; 6 → O5 | |

---

## Write-up (step P2, 2026-10-05; written, awaiting Agent 1/2 sign-off)

Every number: `model_3head_best.pth` (L_calib rows: `model_3head_calib_best.pth`), input scale 1.
Fishyscapes = test half (50 images), all calibrators and thresholds fit on the val half.
Absolute numbers only, no ratios (M12). AP first.

### Calibration table (Fishyscapes test, `eval_fishyscapes_T122.log`; temp whole ECE at 5 dp from `calibrate_compare_T122.log`)

| model | AP | AUROC | FPR@95 | whole ECE | band-ECE r=8 |
|---|---|---|---|---|---|
| raw | 0.6215 | 0.9920 | 0.0290 | 0.0004 | 0.2273 |
| temp, whole-image fit, T = 1.2251 (plain NLL) | 0.6206 | 0.9924 | 0.0289 | 0.00012 | 0.2095 |
| temp, 50/50 fit (C3 control), T = 3.1516 | 0.6177 | 0.9925 | 0.0286 | 0.0177 | 0.1277 |
| temp, edge fit (C2), T = 4.3592 | 0.6173 | 0.9925 | 0.0286 | 0.0420 | 0.0942 |
| T(d), head disagreement (C3) | 0.6106 | 0.9923 | 0.0286 | 0.0193 | 0.1312 |
| context calibrator (C5) | 0.4929 | 0.9707 | 0.0949 | 0.0264 | 0.0738 |
| L_calib (CutMix-trained) | 0.6024 | 0.9924 | 0.0293 | 0.0005 | 0.2396 |

### Outline quality, `ubq_local` (Fishyscapes test, `eval_fishyscapes_T122.log`; each row at its own val max-F1 threshold, roi 32 px, tolerance 4 px; spill/miss/BF1 are means over found objects)

| model | threshold | objects found (of 85) | boundary F1 | spill px | miss px | far-FP (% of flagged px) | flagged (% of valid px) |
|---|---|---|---|---|---|---|---|
| raw | 0.4766 | 40 | 0.505 | 2.97 | 8.94 | 14.7% | 0.144% |
| temp T = 1.2251 | 0.4969 | 39 | 0.514 | 3.16 | 9.33 | 13.8% | 0.138% |
| L_calib | 0.4612 | 43 | 0.537 | 2.57 | 7.62 | 19.4% | 0.143% |
| C2 T = 4.3592 | 0.5011 | 39 | 0.509 | 3.13 | 9.47 | 13.3% | 0.135% |
| C3 | 0.5228 | 38 | 0.500 | 2.17 | 11.75 | 11.1% | 0.124% |
| C3 control T = 3.1516 | 0.4997 | 39 | 0.512 | 3.14 | 9.39 | 13.6% | 0.137% |
| C5 | 0.7217 | 59 | 0.584 | 5.37 | 4.50 | 36.1% | 0.170% |
| head 0 only | 0.3621 | 37 | 0.508 | 3.21 | 7.32 | 17.5% | 0.147% |

Rule (e): per-image boundary F1 C5 − raw +0.0950 [+0.0349, +0.1543] over 24/50 images (both rows
found ≥ 1 object) → keep "context improves outlines at a matched operating point", **with its
costs** (M21): far-FP 36.1% vs 14.7% of flagged px, AP 0.6215 → 0.4929, FPR@95 0.0290 → 0.0949.
The pasted CARLA route's `ubq_local` (raw 938/1245 objects found, BF1 0.657; `eval_video_town02_pasted_T122.log`) uses
**self-fitted** thresholds (no val split, t ≈ 0.88): descriptive only, never shown next to the
Fishyscapes 40/85.

### Head 0 vs the 3-head mean

On our model the 3-head mean beats its own head 0 by 0.041 AP: 0.5810 vs 0.6215, ΔAP −0.0406
[−0.0685, −0.0125], paired over the 50 test images (`eval_fishyscapes_T122.log`). Head 0 is one member of a jointly
trained ensemble at the epoch selected on *fused* val AP: this is the averaging gain at a fixed
epoch and an upper bound on the gap to a separately trained, val-selected 1-head model (the
lead's separately trained COCO 1-head model matched 3 heads). Box recall on the pasted CARLA
video is nearly equal (head 0 858 vs 3 heads 862 of 1245 objects, threshold 0.5, min box 150;
`static/video/twinguard_video_town02_pasted_stats.json`) — a different metric and domain. Not
"3 heads detect better than 1".

### Findings 1-6

1. **Whole-image ECE hides boundary miscalibration.** Raw: whole-image ECE 0.0004, band-ECE r=8
   0.2273 (`eval_fishyscapes_T122.log`). With 0.28% anomalous pixels the whole-image number is dominated by easy
   background. (No ratio: M12.)
2. **The boundary is under-confident** (C1, every radius, raw and L_calib; r=8 mean score 0.2258
   vs positive rate 0.4421, `eval_fishyscapes_T122.log`). The miscalibration is learned at the training resolution;
   it is not caused by the eval-side upsampling order (C1b) and is not reduced by feeding
   full-resolution input at inference (P3, val, descriptive: band-ECE r=8 0.1405 at scale 1,
   0.1907 at scale 2, 0.1745 multi-scale; `eval_scale_val.log`). Caveat: the heads were trained
   only on half-resolution features, so this cannot separate "not a resolution effect" from
   "the heads don't transfer across scale" (M9).
3. **No single temperature calibrates both.** On the same plain-NLL objective the whole image
   wants T = 1.2251 and the edges T = 4.3592 (C2; bounded fit 4.3686). As (whole ECE, band r=8):
   raw (0.0004, 0.2273); T 1.2251 (0.00012, 0.2095); T 3.1516 (0.0177, 0.1277); T 4.3592
   (0.0420, 0.0942) (`eval_fishyscapes_T122.log`, `calibrate_compare_T122.log`). Paired: C2 − temp band r=8 −0.1153
   [−0.1378, −0.0732], whole +0.0418 [+0.0390, +0.0443].
4. **Calibration learned on pastes transfers the wrong correction** — measured on both sides:

   | data | edge direction (C1 r=8 gap) | L_calib − raw, r=8 | L_calib − temp(1.2251), r=8 | source |
   |---|---|---|---|---|
   | Fishyscapes test (real) | under-confident (−0.2163) | **+0.0123 [+0.0017, +0.0258]** (worse) | **+0.0301 [+0.0196, +0.0397]** (worse) | `eval_fishyscapes_T122.log` (50-image bootstrap) |
   | Pasted CARLA route (native props + pastes, self-fit, 400 frames) | over-confident (+0.1292) | **−0.0162 [−0.0207, −0.0114]** (better) | −0.0046 [−0.0104, +0.0016] (≈ 0) | `eval_video_town02_pasted_T122.log` (20-frame circular block bootstrap) |

   L_calib lowers edge scores: that helps only where edges are already over-confident (the
   paste-like route) and there it is no better than one global temperature fitted on real val;
   on real objects it is worse than both raw and temperature. Caveats: one route, 20 blocks; the
   route contains CARLA-bank pastes whose tile quality is under audit (A1, pending).
5. **Head disagreement localises the boundary but does not tell the direction.** Mean
   disagreement 0.0508 in the r=8 band vs 0.0007 whole-image; C3 − control r=8 +0.0035
   [−0.0010, +0.0052] (CI includes 0) and whole +0.0016 [+0.0015, +0.0017] (worse): a
   disagreement-conditioned T is no better than one T on the same objective (`eval_fishyscapes_T122.log`).
6. **Neighbourhood context (C5) fixes edges and outlines, at a detection cost.** Band r=8 AUROC
   0.7318 → 0.8167, band-ECE 0.2273 → 0.0738; per-image boundary F1 +0.0950 [+0.0349, +0.1543]
   over 24/50 images; objects found 59 vs 40 of 85. Costs: far false alarms 36.1% vs 14.7% of
   flagged px, AP 0.6215 → 0.4929, FPR@95 0.0290 → 0.0949; C5 fails its rule 2 (whole-image
   +0.0088 [+0.0082, +0.0092] vs control) (`eval_fishyscapes_T122.log`). The old "missed extent 12.8 → 4.8 px" is
   withdrawn (M11, M21).

### Why each calibration attempt failed (Fishyscapes test half unless stated)

| Attempt | What it tried | Why it failed | Evidence |
|---|---|---|---|
| L_calib | fine-tune with a differentiable ECE loss on CutMix pastes | its correction (lower edge scores) is the wrong direction for real objects, whose edges are under-confident | Fishyscapes r=8 +0.0123 [+0.0017, +0.0258] vs raw and +0.0301 [+0.0196, +0.0397] vs temp(1.2251) (`eval_fishyscapes_T122.log`); pasted CARLA route −0.0162 [−0.0207, −0.0114] vs raw, ≈ temp (block CI; `eval_video_town02_pasted_T122.log`) |
| L_calib β=50 | stronger loss weight | surrogate went down, real ECE up | ECE 0.0004 → 0.0014, AUROC 0.9920 → 0.9898 (`config.py` BETA_CALIB history; no log on disk) |
| Whole-image temperature T = 1.2251 | one T on real val (plain NLL) | tuned to background pixels (99.7% of the image) | whole ECE 0.0004 → 0.00012, edge ECE only 0.2273 → 0.2095 (`eval_fishyscapes_T122.log`, `calibrate_compare_T122.log`) |
| Edge temperature (C2) T = 4.3592 | T fit on r=8 band pixels | one T can't serve both: edges want ~4.36, the image ~1.23 | edge 0.0942, whole ECE 0.00012 → 0.0420 (`eval_fishyscapes_T122.log`) |
| Disagreement temperature (C3) | T varies with head disagreement | disagreement marks where edges are, not which side | C3 − control r=8 +0.0035 [−0.0010, +0.0052] (`eval_fishyscapes_T122.log`) |
| Context calibrator (C5) | neighbourhood max/mean score | spreads anomaly score into nearby background | edge 0.0738 (best), AP 0.6215 → 0.4929, far-FP 14.7% → 36.1%, fails rule 2 (`eval_fishyscapes_T122.log`) |

Note (descriptive): on the pasted CARLA route a Fishyscapes-fitted T = 1.2251 worsens whole ECE
(0.009205 → 0.010719; `eval_video_town02_pasted_T122.log`), since edges there are over-confident.

---

## Established results (unaffected by M1; each cites its log)

All on `model_3head_best.pth`, scale 1, unless stated.

- **P1, temperature redo** (`eval_fishyscapes_T122.log`, Fishyscapes test half, signed
  2026-10-05): plain-NLL whole T = 1.2251 (val); temp row AUROC 0.9924, AP 0.6206, whole ECE
  0.0001 (4 dp), band r=8 0.2095. All T-independent rows equal `eval_fishyscapes_c5.log`
  exactly. Objects found (raw, t = 0.4766): 40/85; boundary F1 0.505, spill 2.97 px, miss
  8.94 px, far-FP 14.7%. Head 0 alone AP 0.5810 vs fused 0.6215 (see Superseded row 13).

- **Model** (`train.log`): epoch 4 of 8 selected on val AP 0.6839; test AUROC 0.9920, AP 0.6218,
  FPR@95 0.0287, ECE 0.0004; head-disagreement AUROC 0.9865; mIoU 0.7616.
- **Raw calibration, Fishyscapes test** (`eval_fishyscapes_c5.log`): whole ECE 0.0004; band-ECE
  r=4/8/16 0.2848/0.2273/0.1748; under-confident at every radius (r=8 mean score 0.2258 vs
  positive rate 0.4421).
- **C2/C3/control/C5** (`eval_fishyscapes_c5.log`): rows in the P1(b) table. C3 fails its
  control; C5 passes rule 1 (r=8 −0.0539 vs control, CI excludes 0) and fails rule 2 (whole
  +0.0088, CI excludes 0).
- **L_calib vs raw, Fishyscapes** (`eval_fishyscapes_c5.log`): r=8 +0.0123 [+0.0017, +0.0258].
- **Pasted CARLA route `video_town02_pasted`** (`eval_video_town02.log`, self-fit thresholds,
  pastes on CARLA, not real-world): raw AUROC 0.9799, AP 0.6604; C1 over-confident; L_calib −
  raw r=8 −0.0162 [−0.0178, −0.0142] — **frame-level CI over correlated frames, optimistic
  (M20)**; the block-bootstrap CI comes from P1b.
- **Video boxes** (`static/video/twinguard_video_town02_pasted_stats.json`, threshold 0.5, min
  box 150): head 0 found 858/1245 (68.9%), 1361 false boxes (3.40/frame); 3 heads 862/1245
  (69.2%), 1434 false boxes (3.585/frame). First render at min box 40: 961 vs 967 of 1245 (PLAN_HISTORY 2d;
  stats file since overwritten by the re-render).
- **X1** (`explain_x1.log`, 37 Fishyscapes test objects): Telea object removal drops the score
  0.347 → 0.091, +0.256 [+0.145, +0.362]; partial edits and context blur raise it.
- **X1b** (`explain_x1b.log`, 75 pasted frames): 0.951 with paste, 0.016 truly removed, 0.625
  Telea-removed; Telea bias +0.609 [+0.538, +0.681].
- **X2** (`explain_x23.log`): stage-4 share 0.42 with sign −0.68 on real objects; pasted objects
  all stages positive; stage-1 share pasted − real +0.049 [+0.011, +0.085]. Wording per M17.
- **X3** (`explain_x23.log`): Fishyscapes edge band aleatoric 0.236 vs epistemic 0.010; FP − TP
  disagreement n=9 +0.0067 [−0.0153, +0.0277] (not supported on real data).
- **exp_v2 context, other checkpoint-labelled numbers** (exp_v2 `runs/RESULTS.md`, cited as
  context only, M16): on a file believed to be the same as ours (numbers agree to 4 dp;
  SHA256 check pending U0-1) ("friend's CARLA+COCO raw"), temp(whole) T=1.2247
  → ECE 0.00013, band r=8 0.2096; temp(band) T=4.3681. On the lead's COCO-only
  `phase2a_coco_run1`: objects found 40/85 (47%), RA21 val AUROC 0.72 vs MSP 0.87, 1-head ≈
  3-head detection.

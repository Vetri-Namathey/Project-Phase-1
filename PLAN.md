# TwinGuard — next-steps plan

Working checklist. Tick items as they're actually done — "done" means the artifact in
the right-hand column exists on disk, not just that code was written.

## RESUME HERE (paused 2026-09-21, pod stopped)

Just cleared the project's top blocking item: a real, verified, post-BCE-fix Experiment B
number. See "Anomaly bank rebalancing — implemented 2026-09-21" below for the full result
(`AUROC=0.9920`, clears both the 0.75 and 0.83 gates). Checkpoint + `train.log` + `mlflow.db`
pulled off the pod before stopping it; pod itself and its venv/data are gone/disposable.

Pick one of these three to start tomorrow (none blocks the others, no order requirement):
1. **Wire the new checkpoint into the demo** — `server.py`/`README.md` still show the stale
   Checkpoint B (0.6190). Needs a filename-tagging decision first (see the note under the
   rebalancing checkpoint below) so it doesn't get silently overwritten by a future run.
2. **Start `L_calib` calibration work** — now unblocked, gate cleared. See "Calibration
   (L_calib) gate" section below for what it has to be (joint fine-tune + differentiable
   ECE surrogate, not temperature scaling) and the proof-of-change/frontend requirements.
3. **AP=0.6218 sanity check** — flagged as an open question, not urgent (AUROC was measured
   on real Fishyscapes, not synthetic CutMix, so probably fine — just not explicitly checked).

Also still open, not blocking, no rush: the FPR@95=1.0 vs documented-0.6802 discrepancy on
Experiment A (see "Blocking — do before trusting any Experiment A vs B comparison" below).

## Branch context — merged 2026-09-19, superseding the note below

The audited pipeline (BCE-with-logits fix, drivable-surface placement, photometric
harmonization, val/test split, AP/mIoU, `preflight.py`, `check_runs.py`) has been pulled
onto `main` from `origin/exp_coco`, file-by-file, keeping `main`'s demo (`frontend/`,
`server.py`) intact rather than losing it to a branch merge. `main` is now the single
branch to work from — see `CLAUDE.md`'s "Git branches" section for exactly what moved,
what was kept, and the one real breakage (raw-logit OOD output vs. the demo's [0,1]
rendering assumption) that was found and fixed in `server.py` during the merge.

*(Original 2026-09-19 note, kept for history: "The audited pipeline lives on
`origin/exp_coco`, not `main`... these two branches have diverged hard and will need a
deliberate merge." — that merge has now happened.)*

Confirmed already fixed on `exp_coco` (verified directly, not just claimed):
- `data/anomaly_sources.py` — CARLA and COCO already live behind one clean interface,
  gated by `config.ANOMALY_SOURCE`, which on `exp_coco` accepted `"carla" | "coco"` and
  defaulted to `"coco"` only because no CARLA machine was available at the time. On
  `main` it now also accepts `"both"`, which is the current default (see below); the
  `config.py` comment was updated to match, so don't look for the old "temporary
  stand-in" wording.
- `CUTMIX_SCALE_MIN/MAX = 0.012 / 0.20` — the oversized-paste bug is already fixed here,
  matching real Fishyscapes anomaly-size statistics ("to within a pixel or two" per the
  code comment).
- Still **not** implemented anywhere on either branch, confirmed via `grep -rn "UBQ\|Hausdorff"`:
  UBQ is objective-only, not built.

## Blocking — do before trusting any Experiment A vs B comparison

- [x] Re-run the baseline (`experiment_a.py`, now on `main`) for the test-half number —
      done 2026-09-21 on RunPod (A40), against the rebalanced 2000-object anomaly bank
      (irrelevant here — `experiment_a.py` doesn't touch CutMix or the anomaly bank at
      all, it's untrained off-the-shelf SegFormer + MSP, no OOD heads involved).
      → artifact: MLflow run `experiment_a_baseline`, logged to
        `/workspace/twinguard_v2/mlflow.db` on the pod (pull it off before the pod goes
        away — Step 9).
      **Test half**: AUROC=0.8372  AP=0.0114  FPR@95=1.0000  ECE=0.0148
      **Val half**:  AUROC=0.8170  AP=0.0176  FPR@95=1.0000  ECE=0.0161
      - AUROC and ECE line up closely with the previously documented full-100-image
        number (`README.md`: AUROC 0.8304, ECE 0.0154) — same deterministic baseline,
        consistent result, clears both the 0.75 and 0.83 gates.
      - **FPR@95=1.0000 does not match** the documented 0.6802, on *both* halves, not
        just one — not yet fully explained. Working theory, not confirmed: the
        documented 0.6802 was measured over the full 100 images combined, whereas this
        run splits into 50/50 val/test; FPR@95 is a tail-quantile metric and known to be
        unstable under this project's severe (~0.28%) class imbalance with half the
        sample size. AP=0.0114 has no prior recorded figure to compare against.
      - **Not resolved — flag if this matters for the viva**: don't present FPR@95=1.0
        as "the baseline is broken" (it isn't a trained model, it can't "collapse" the
        way the OOD heads can) or quietly drop it — if asked, the honest answer is
        "AUROC/ECE replicate the documented baseline; FPR@95 diverges on the half-splits
        and the cause isn't confirmed yet."
- [x] Run `preflight.py` clean immediately before the next paid RunPod run — done
      2026-09-19, right after merging the audited pipeline + curating CARLA + adding
      COCO. Result: **READY, with 1 warning** (no local CUDA device — expected, real
      training runs on RunPod; not a pipeline defect). All 13 checks passed: dataset
      paths, Cityscapes/Fishyscapes loaders, anomaly bank (3045 objects, `both` mode),
      CutMix output stats matching real Fishyscapes distribution, model wiring (logits
      emitted, encoder frozen correctly), one real training step, metrics sanity check.
      → artifact: preflight stdout captured this session; re-run and save a copy
        alongside the next real RunPod run's MLflow entry (this run wasn't logged to
        MLflow, it's a local sanity check only).

## CARLA object bank — curation (from tonight's discussion)

- [x] Audit all 50 local CARLA objects (`data/images/`, `data/masks/`) for oversized/
      implausible props — done 2026-09-19 by background agent, full 50-row table with
      frame-area-fraction + visual judgment.
      **Result: 5 of 50 objects are bad, 45 are fine.**
      - `#39` (fountain/monument + human statue, 40.8% of frame) — wrong category AND oversized
      - `#40` (full bus-stop shelter structure, 30.0% of frame) — wrong category AND oversized
      - `#14`, `#15` (trampoline, ~8% of frame each) — wrong category AND oversized
      - `#41` (smaller bus-shelter instance, 6.3%) — wrong category (street furniture),
        size alone isn't extreme
      - Everything ≤5.4% frame area (45 objects) reads as plausible road debris — pallets,
        road flares, cones, bollards, wheelie bins, crates, warning signs — 20 of these
        visually confirmed directly, the rest inferred from consistent size/aspect-ratio
        with the confirmed set.
      - Note: `#40`/`#41` and `#44`/`#45` and `#19`/`#20` etc. look like the same underlying
        3D prop rendered at different distances — the 50 "objects" aren't all unique;
        excluding `#39-41` removes both bad prop families in one pass.
      - Separate issue, not size/category: `#15` (and to a lesser extent `#12`) renders at
        very low contrast against the road despite a large mask — a paste that's barely
        visible is useless for training regardless of plausibility; worth a second look.
- [x] Curate the CARLA bank: **exclude indices 39, 40, 41, 14, 15** from the object bank
      — done 2026-09-19. Implemented by moving the 5 files (not a code filter list, since
      `data/anomaly_sources.py`'s CARLA builder just globs the directory) from
      `data/images/`+`data/masks/` into `data/_excluded_objects/` — reversible, not
      deleted. 45 objects remain and were verified by `preflight.py`.
- [x] Add an `ANOMALY_SOURCE = "both"` mode to `data/anomaly_sources.py` — done
      2026-09-19: `_build_both_bank()` pools curated-CARLA (45) + exclusion-filtered-COCO
      (3000, from a fresh `download_coco_anomalies.py` run) into one bank, sampled
      uniformly per paste, 3045 objects total. Verified via `preflight.py`'s anomaly-bank
      and cutmix-output checks (both passed).
      → **done 2026-09-21** — see the rebalanced-bank training run below
        (`experiment_b_3head`, 2000-object `both` bank, not the original 3045). The
        `carla`-only/`coco`-only comparison runs are still not done — see the ablation
        checklist below.

## Anomaly bank rebalancing — implemented 2026-09-21

Was a discussion-only checkpoint (2026-09-20 entry below), now built and verified.
Original problem: `"both"` bank pooled 45 CARLA + 3000 COCO sampled *uniformly per
paste* — CARLA was effectively ~1.5% of what actually got pasted during training, which
undercut the whole point of pooling both sources (the CG-render vs. photo
statistical-signature argument in `_build_both_bank`'s docstring).

**What was actually built** (not what the 2026-09-20 checklist originally assumed):
growing CARLA to 500 *distinct* objects would need the CARLA simulator running locally
to render new props — a separate, bigger task, not something codeable against RunPod.
Instead, `_build_both_bank()` (`data/anomaly_sources.py`) now **repeats** the existing
curated 45 CARLA files up to `config.CARLA_BANK_TARGET` (default 500) and takes a
**seeded random subsample** of COCO down to `config.COCO_BANK_TARGET` (default 1500) —
total pool 2000, CARLA now ≈25% of what gets pasted instead of ~1.5%. Honest tradeoff,
stated explicitly in the code comment: repeating the same 45 files does not add new
CARLA visual diversity, it only increases how often those same 45 objects get pasted
(CutMix's own random scale/position per paste means it isn't pixel-identical every
time, but it is the same 45 underlying objects). Growing to genuinely distinct CARLA
objects remains a separate, not-yet-started task requiring a live CARLA server.

- [x] Added `CARLA_BANK_TARGET = 500` / `COCO_BANK_TARGET = 1500` to `config.py`.
- [x] `_build_both_bank()` repeats CARLA pairs to the target (ceil-div tile, truncated)
      and subsamples COCO pairs to the target via `random.Random(config.GLOBAL_SEED)`
      — reproducible across machines/pod restarts, not re-randomized per run.
- [x] Verified locally: `build_anomaly_bank()` reports exactly 2000 objects;
      `preflight.py` reruns clean, `READY` with only the expected no-CUDA warning, all
      13 checks pass, CutMix output stats still track the real Fishyscapes distribution
      (pos rate 0.2925% vs real 0.280%, per-object size percentiles still overlapping
      the real band).
      → artifact: preflight stdout captured 2026-09-21, this session.
- [ ] Not done: the COCO literal-duplicate-crop check from the original discussion (dedupe
      vs. normal per-category frequency variance was never actually resolved — see
      "Pending" note below). The 1500-subsample is a plain random sample of the existing
      3000, duplicates and all, if any exist.
- [ ] Not done: re-running the CARLA curation audit "at the new scale" doesn't apply the
      same way anymore, since no new CARLA renders were produced — the 45 curated objects
      are unchanged from the existing 2026-09-19 audit. Only relevant again if/when
      genuinely new CARLA objects get rendered.
- [ ] Decide (still open): does 2000 (500/1500) become the new permanent default for
      `"both"`, or is it one arm of the CARLA-vs-COCO-vs-both ablation below? Not decided
      yet — this was built to unblock the current RunPod run, not as a final answer.
- [x] **Blocking item cleared 2026-09-21** — real RunPod A40 training run,
      `experiment_b_3head`, rebalanced 2000-object `both` bank (500 CARLA-repeated /
      1500 COCO-subsampled), 8 epochs, selected checkpoint = epoch 4 (best val AP).
      **Test half: AUROC=0.9920  AP=0.6218  FPR@95=0.0287  ECE=0.0004** (head-disagreement
      AUROC=0.9865). Clears **both** the 0.75 pre-calibration gate and the 0.8304/0.83
      post-calibration target — the first run to clear either, and the first valid
      post-BCE-fix Experiment B number to exist at all (Checkpoints A/B both predate that
      fix and are non-comparable, per `CLAUDE.md`).
      - Verified, not just trusted: `check_collapse.py` on the saved checkpoint shows all
        three heads healthy (separations +0.38/+0.50/+0.48, std 0.048-0.057 — all well
        past the 0.01 collapse-warning threshold and the low-std-is-expected-here note).
        `check_runs.py` confirms the same numbers side by side with Experiment A.
        Epoch-8 tail of `train.log` also shows healthy separation (+0.52/+0.48/+0.54) and
        mIoU held at 0.7690 — not sacrificed for OOD performance.
      - Artifacts pulled off the pod before disconnecting: `checkpoints/model_3head_best.pth`,
        `train.log`, `mlflow.db` (local copies, this session, 2026-09-21).
      - **Not yet reconciled with `server.py`**: this checkpoint is not yet wired into the
        live demo (`config.CHECKPOINT_3HEAD` / `EXPERIMENT_HISTORY`) or `README.md`'s
        Experiment B table — those still show Checkpoint B's stale, pre-fix 0.6190. Needs
        a decision on filename tagging (this run's checkpoint composition — rebalanced
        `both`, BCE-fixed — isn't yet distinguishable by filename from a future different
        run, per the tagging scheme `CLAUDE.md` describes) before treating it as "the"
        checkpoint.
      → artifact: `mlflow.db` (local copy), `train.log`, `check_collapse.py`/`check_runs.py`
        stdout, all captured 2026-09-21.

## Experiments the audit fixes unlocked

- [ ] CARLA vs COCO vs both — three-way ablation, identical pipeline, only
      `config.ANOMALY_SOURCE` flipped
      → artifact: three MLflow runs, side-by-side AUROC/AP/mIoU table
- [ ] Verify the paste-shortcut fix actually closed the gap (harmonization + feathering,
      already on `exp_coco`)
      → artifact: a short probe script result — can a simple classifier still tell
        "real Cityscapes patch" from "pasted patch" using OOD-head features? report
        its accuracy; near-chance = shortcut closed
- [ ] Confirm the "peaks epoch 1, degrades after" pattern (Slide 08's collapse signature)
      is actually gone under the fixed pipeline
      → artifact: `check_runs.py --trend` output for the new clean run
- [ ] Temperature scaling / Phase 3 calibration — now implementable since heads emit logits
      → artifact: `L_calib` implementation + a run showing AUROC before/after calibration
        (expect a small AUROC cost per `CALIBRATION_TRADEOFF_NOTE` — that's fine, not a bug)

## Calibration (L_calib) gate — reconciliation + sequencing (2026-09-20, discussion, not yet acted on)

Read `TwinGuard_Full_Plan_Updated (1).html` in full during this discussion. It disagrees
with `config.py` on the gate, and the current real numbers clear neither reading —
recorded here so this isn't re-litigated from scratch next session.

- **Gate conflict, unresolved:** the plan doc's Section 02 states flatly that *nothing* in
  Section 03 (which Novelty 1 explicitly places `L_calib`/calibration inside of) gets built
  until Experiment B beats **0.8304** (the full baseline AUROC). `config.py`'s
  `PRECALIBRATION_AUROC_GATE = 0.75` is looser. Not reconciled — treat **0.75 as the floor
  to even start Phase 2b**, but don't claim the plan doc's stricter 0.83 reading has been
  overridden without checking with whoever owns that doc.
- **Resolved 2026-09-21 — no longer moot:** the fresh `experiment_b_3head` run (rebalanced
  2000-object `both` bank, post-BCE-fix pipeline) scored **AUROC=0.9920** on the test half,
  clearing 0.75 *and* 0.8304/0.83. This is the first valid post-fix number to exist — see
  the rebalancing checkpoint above for the full result and verification (`check_collapse.py`,
  `check_runs.py`). The old Checkpoint A/B numbers (0.5727 / 0.6190) are now fully
  superseded for gate-checking purposes, though still worth keeping in `README.md` as the
  documented history of the collapse-then-fix story.
- [x] **Blocking item cleared 2026-09-21** — see above. Gate cleared by a wide margin
      (0.9920 vs both 0.75 and 0.83), not just barely.
- [ ] **New follow-up, not yet done:** the plan doc's Section 02 wants Experiment B to beat
      0.8304 specifically to unlock Section 03 work (L_calib included) — 0.9920 clears that
      too, so both readings of the gate (the looser `config.py` 0.75 and the stricter plan-doc
      0.8304) are now satisfied simultaneously. The reconciliation question (which one is
      "the" gate going forward) is now moot in practice, though still not formally answered
      for future gates that might not clear both at once.
- [ ] **Open question worth flagging, not yet investigated:** this AUROC (0.9920) is
      substantially higher than Experiment A's baseline (0.8372) — good, but also high
      enough to sanity-check before building calibration work on top of it. `check_collapse.py`
      already rules out the specific collapse failure mode (heads outputting near-constant
      scores), but a high-AUROC/low-AP combination (AP=0.6218, not near 1.0) is still
      worth a second look — e.g. confirming test-half CutMix-pasted synthetic anomalies
      aren't systematically easier than real Fishyscapes anomalies would be, given AUROC
      here is measured on the *real* Fishyscapes test half, not on CutMix — so this
      concern may not apply, but hasn't been explicitly checked.
- **What `L_calib` has to be, once the gate clears** (per Novelty 1 in the plan doc): *not*
  plain post-hoc temperature scaling. The whole justification for building it is beating
  temperature scaling specifically on **spatial** calibration (boundary-region ECE, UBQ) —
  a single scalar rescaling can only rescale confidence uniformly, it mathematically cannot
  reshape calibration spatially. So this has to be a **joint fine-tune with a differentiable
  calibration-aware loss term** (an ECE surrogate — real histogram-based ECE isn't
  differentiable), evaluated three ways against a temperature-scaling baseline: whole-image
  ECE, boundary-only ECE (Novelty 6, also not built), and UBQ (also not built — see below).
  Documented fallback if temp scaling ties on whole-image ECE: pivot the claim entirely to
  boundary ECE + UBQ, since a scalar fix structurally can't compete there.

### L_calib — implemented 2026-09-25, not yet run on real data

Code written and locally sanity-checked; **no real training run has happened yet** — this
needs the pod (GPU + real Cityscapes/Fishyscapes), not just correct code.

- [x] `losses.py`'s `SoftECELoss` — differentiable ECE surrogate (soft-binned, triangular
      kernel membership, squared per-bin conf/acc gap). Unit-tested locally on synthetic
      data (no GPU/dataset needed): near-zero loss on a well-calibrated synthetic score,
      clearly higher on a deliberately overconfident one, gradients finite and flowing.
      This is the actual test that exists — it proves the loss is directionally correct
      and differentiable, NOT that it works on the real model/data.
- [x] `metrics.py`'s `ScoreHistogram.reliability_curve()` — per-bin (confidence, accuracy,
      weight), sharing `ece()`'s own binning so the diagram and the reported ECE number
      can never silently disagree.
- [x] `config.py` — `BETA_CALIB`, `CALIB_EPOCHS`, `CALIB_LEARNING_RATE`,
      `CALIB_AUROC_DROP_LIMIT`, `CHECKPOINT_3HEAD_CALIB`, `TEMPERATURE_*` added.
- [x] `calibrate.py` (new file) — three steps in one script: (1) fits a temperature-scaling
      baseline on the Fishyscapes val half; (2) joint fine-tune of the already-verified
      checkpoint (`model_3head_best.pth`, AUROC=0.9920) with `SoftECELoss` added to the
      existing seg+OOD loss, selecting on val ECE, stopping (not saving) if val AUROC drops
      more than `CALIB_AUROC_DROP_LIMIT`; (3) the comparison table (raw vs. temp-scaled vs.
      L_calib, whole-image AUROC/AP/FPR@95/ECE) plus a saved reliability-diagram PNG, both
      logged to MLflow. Added to `make_upload.py`'s file list and `requirements-runpod.txt`
      (needed `matplotlib`, which the training-only requirements file didn't have before).
- [x] **Real run done 2026-09-25** on RunPod (A40), against `model_3head_best.pth`
      (AUROC=0.9920, the verified post-rebalance checkpoint). Fine-tune ran 5 epochs,
      AUROC-drop guard never triggered (max drop 0.0033 at epoch 3, well under the 0.03
      limit), selected epoch 1 (best val ECE 0.0009). Full comparison table, test half:

      ```
      model            AUROC      AP  FPR@95     ECE
      raw             0.9920  0.6215  0.0290  0.0004
      temp-scaled     0.9924  0.6193  0.0287  0.0020
      L_calib         0.9924  0.6024  0.0293  0.0005
      ```
      fitted temperature: 1.7142.

      **Honest read, not a clean win:** temperature scaling made whole-image ECE *worse*
      (0.0004 → 0.0020), and `L_calib`'s ECE (0.0005) is barely different from raw's
      already-tiny 0.0004. This is very likely the exact limitation `PLAN.md` already
      anticipated above: at Fishyscapes' extreme 0.28% positive rate, whole-image ECE is
      dominated by the overwhelming majority of easy true-negative pixels, so an
      already-high-AUROC model (0.992) gets a trivially low ECE almost regardless of how
      well-calibrated the hard/boundary pixels actually are — `metrics.py`'s own `ece()`
      docstring warns about exactly this class of near-perfect-but-meaningless ECE. **This
      table does not support a "L_calib beats temperature scaling" claim** — it's real
      empirical confirmation that whole-image ECE can't show the difference, which is
      *why* boundary-ECE/UBQ were always the load-bearing metrics for this claim, not a
      fallback.
      - Small legitimate side-note: AUROC moved slightly under temperature scaling
        (0.9920 → 0.9924) even though a scalar rescale is monotonic per-head — this is
        because fusion is mean-of-per-head-sigmoids, and scaling each head individually
        before averaging is not exactly the same monotonic transform as scaling the
        fused score directly. Expected, not a bug.
      → artifact: `calibrate.log` (full run log), `model_3head_calib_best.pth`,
        `calibration_reliability.png` — pulled off the pod to local machine 2026-09-25
        (this is the BETA_CALIB=1.0 run's artifacts — see below, this is the one kept).

- [x] **Tuning attempt, 2026-09-25, reverted — BETA_CALIB=1.0 → 50 → back to 1.0.**
      The near-no-op result above was suspected to be caused by L_calib's gradient
      contribution being under 1% of the base loss (raw calib-loss ~0.0002-0.0012 vs.
      base ~0.05-0.11), so `BETA_CALIB` was raised to 50 and `calibrate.py` re-run.
      Result: **worse on every axis except AP**.
      ```
                      AUROC      AP  FPR@95     ECE
      raw            0.9920  0.6215  0.0290  0.0004
      L_calib (β=1)  0.9924  0.6024  0.0293  0.0005   -- near-no-op
      L_calib (β=50) 0.9898  0.6116  0.0301  0.0014   -- worse on AUROC, FPR@95, ECE
      ```
      The raw *surrogate* loss did shrink further under β=50 (0.0001, down from
      ~0.0004-0.0006) — so the training signal genuinely strengthened as intended — but
      the real, histogram-based ECE (`metrics.py`) got worse anyway. This is a real
      surrogate/true-metric mismatch, not a bug in either implementation: pushing the
      smooth soft-binned relaxation down harder does not reliably improve the real,
      hard-binned ECE under this severe (~0.28%) class imbalance, where whole-image ECE
      is dominated by trivial true-negative pixels regardless of what happens on the
      harder ones. **Decision: reverted `BETA_CALIB` to 1.0** — the near-no-op result is
      the one being reported, not chased further. Do not re-tune `BETA_CALIB` again
      without boundary-only ECE/UBQ in hand first; whole-image ECE has now failed twice
      (once by being trivially already-low, once by not responding sensibly to a
      stronger training signal) as a metric worth optimizing against directly.
      → artifact: `calibrate2.log` (the β=50 run, pod-only, not pulled to local — the
        β=1 checkpoint above is the one that matters and is already saved locally).

- [ ] Whole-image ECE is what this table can show. Boundary-only ECE (Novelty 6) and UBQ
      are still unbuilt (see the UBQ section below) — the comparison table's own printed
      output says this explicitly, so it can't be quoted as proving the spatial claim on
      its own.
- [ ] Frontend integration (the "Proof-of-change outputs" section right below this one) is
      still entirely unbuilt — nothing from this section is in `frontend/`/`server.py` yet.

### Proof-of-change outputs — and getting them into the live demo, not just MLflow

Decided in discussion: the calibration work isn't done when the fine-tune finishes —
it's done when the result is visible in `frontend/`, not just sitting in `mlflow.db`.

- [ ] **Comparison table**: raw model vs. temperature-scaled vs. `L_calib` fine-tuned,
      on AUROC/AP/FPR@95/ECE/mIoU (Fishyscapes test half) — computable today via
      `metrics.py`'s existing streaming histogram, no new metric code needed. Must show
      the AUROC cost next to the ECE gain together (per `CALIBRATION_TRADEOFF_NOTE`), not
      ECE alone with no context.
      → artifact: `check_runs.py`-style table, same shape as the existing experiment
        comparison it already produces.
- [ ] **Boundary-only ECE** (Novelty 6) and **UBQ** — both still unbuilt (see UBQ section
      below); required before the calibration claim is complete, since whole-image ECE
      alone can't prove `L_calib` beats temperature scaling — only the spatial metrics can.
- [ ] **Reliability diagram** (confidence-vs-actual-accuracy, binned): three curves — raw,
      temp-scaled, `L_calib` — raw bowed away from the diagonal, `L_calib` closest to it.
      → artifact: a saved plot, same per-pixel-scores-and-labels data the eval loop
        already produces.
- [ ] **Frontend integration — the actual deliverable, not an afterthought:**
      - New panel in `frontend/`'s demo (`server.py` already has the plumbing pattern for
        cached static panels — same approach as the existing OOD/uncertainty heatmaps):
        the v1-vs-v3 side-by-side panel already scoped in the plan doc — same frame, v1's
        loose uncertainty blob next to v3's tight boundary-hugging outline. This was
        explicitly flagged there as "the one moment the whole dashboard exists for."
      - The comparison table and reliability diagram surfaced somewhere in the demo too
        (e.g. a "Calibration" tab/section alongside the existing "Training Runs" page),
        not left as a notebook output or an MLflow-only artifact — reuse
        `EXPERIMENT_HISTORY`'s existing pattern in `server.py` rather than inventing a
        new data path.
      - **Guardrail, already stated once in the plan doc and worth repeating here:** if a
        demo date lands before `L_calib`/UBQ/boundary-ECE actually exist, show the panel
        explicitly labeled "target output — pending Model v3," not faked or approximated
        data. This is called out there as the single most reputationally expensive mistake
        available in the whole project.

## UBQ (Uncertainty Boundary Quality) — not started

- [ ] Implement UBQ: directed-Hausdorff distance (`scipy.spatial.distance.directed_hausdorff`)
      between predicted-uncertainty boundary and true anomaly extent
      → artifact: `metrics.py` function + unit-style sanity check
- [ ] Validate first against CARLA's exact ground-truth masks (per the project's own
      objective ordering — CARLA masks are pixel-exact via semantic-tag diffing)
      → artifact: UBQ number(s) on the curated CARLA object set
- [ ] Then validate on real Fishyscapes masks
      → artifact: UBQ number(s) on the Fishyscapes test half, alongside AUROC/AP/ECE
- Until this exists, do not claim UBQ results in the viva — say it's defined as an
  objective and not yet measured (see the results-notes artifact's Q&A for exact phrasing)

## LiDAR cross-check (Section 06-A, large-scale direction) — planning checkpoint (2026-09-20)

Full version, not the doc's qualitative fallback — logged as the actual target. Gated same
as everything else in Section 06: doesn't start before Experiment B clears its AUROC gate,
and per the plan doc, pick at most one of A/B/C.

**Correction to the plan doc's premise, confirmed by reading the actual code:** the doc
describes this as "the LiDAR we're already generating... currently discarded." That's not
true of this codebase — `grep -rn "lidar"` across the whole repo returns nothing, and
`generate_anomalies.py` spawns exactly two sensors (`sensor.camera.rgb`,
`sensor.camera.semantic_segmentation`), no LiDAR actor anywhere. This is CARLA's own
*simulated* LiDAR (`sensor.lidar.ray_cast`, a built-in CARLA sensor blueprint, same family
as the two cameras already spawned) — not physical hardware — but it still has to be added
new, not just "wired up."

- [ ] Add a `sensor.lidar.ray_cast` actor to `generate_anomalies.py`, same transform/timing
      as the existing `rgb_camera`, so every captured frame gets RGB + semantic mask +
      point cloud together.
      → artifact: point-cloud file saved alongside each frame's existing RGB/mask output.
- [ ] Project the point cloud into the camera's image plane using the sensor's own
      transform/intrinsics (standard CARLA extrinsics — the sensor transform is already
      known since it's set explicitly, same as the RGB/semseg cameras) to get a per-pixel
      or per-region depth map.
      → artifact: a depth map per frame, spatially aligned to the RGB/mask pair.
- [ ] Compute local depth discontinuities (gradient/Laplacian on the projected depth) as
      the geometric "something's here" signal, independent of anything the vision model sees.
      → artifact: a discontinuity map per frame.
- [ ] Cross-check against the trained model's OOD uncertainty heatmap: quantify, not just
      eyeball, how often high vision-uncertainty regions coincide with real geometric
      discontinuities vs. how often they fire on geometrically flat/continuous surfaces
      (the mismatch case is the actual diagnostic value here).
      → artifact: a real correlation/agreement number between the two signals, across the
        Fishyscapes test half — not a handful of qualitative example figures.
- Why the full version over the fallback: the doc's own fallback ("flag disagreement
  qualitatively in a handful of figures") was offered as a scope-reduction if full fusion
  proves too costly, not as the intended target — logging the full quantitative version
  here so it isn't quietly downgraded by default later.

## Real-time CARLA video proof — why CARLA specifically, not just "a video"

Confirmed reasoning from tonight's discussion, worth keeping precise:
- Real-world video can't give **ground truth** to auto-verify "correctly predicted" live
  — CARLA's semantic-tag diffing already gives pixel-exact masks for free.
- Real-world video can't give **reproducibility** — showing "old checkpoint missed this,
  new checkpoint catches it" needs the *identical* scenario replayed through two
  checkpoints. CARLA is deterministic/scriptable (same seed, same spawn, same route);
  real driving footage can never be replayed frame-for-frame identically.
- Practical note: doesn't need a *live* simulator during the actual presentation —
  CARLA supports playback of a pre-recorded sequence, safer for a live demo (no
  simulator crash risk, deterministic, pick the clearest example ahead of time).
- This is unchanged by which source(s) feed *training* data (CARLA/COCO/both) — training
  data source and demo-video source are separate pipeline stages.

### How anomalies actually get into the demo video (2026-09-20 discussion) — two distinct mechanisms, not one

- **Native CARLA spawn** — a real `static.prop.*` object placed directly into the 3D world
  along the driving route (same mechanism `generate_anomalies.py` already uses to generate
  training crops, just left in-scene instead of cropped out). Gives real 3D consistency
  (parallax, correct occlusion, a real LiDAR return if the LiDAR checkpoint above gets
  built) and keeps the "why CARLA specifically" argument fully intact: pixel-exact ground
  truth via semantic-tag diffing, deterministic replay for old-vs-new-checkpoint comparisons.
- **Post-render CutMix compositing** — paste a 2D object cutout onto pre-recorded CARLA
  frames *after* rendering, the same technique already used for training, just applied to
  video frames instead of individual Cityscapes images.
  - **This is the only way COCO objects can appear in the video at all.** COCO cutouts are
    flat real-world photos, not 3D assets — there is no mesh for CARLA to spawn. They can
    only ever be a 2D composite on top of a rendered frame, never a native in-world object.
  - Ground truth stays exact either way (the paste mask is the same one used in training),
    but there's no real 3D behavior for a pasted object — no parallax as the camera moves,
    no LiDAR return.
- **Practical plan for "a lot of anomalies" in one video:** mix both — a handful of
  natively-spawned CARLA objects along the route (for the reproducible-replay / LiDAR-
  compatible cases) plus post-render CutMix pastes of both CARLA and COCO cutouts (for
  volume/variety). Track which mechanism produced which instance in the video's metadata —
  "pixel-exact via live semantic-tag diffing" is only strictly true for the natively-spawned
  ones; pasted ones are exact too, just via the pre-cut training-pipeline mask, not a live
  3D-scene diff. Don't blur this distinction when describing ground-truth provenance later.
- [ ] Not started: no code exists yet for either native mid-route spawning during a
      recorded sequence, or post-render CutMix-onto-video compositing. Both are new work
      on top of `generate_anomalies.py`'s existing per-frame-crop capture mode.

### RoadAnomaly21 / OoDIS — status check (2026-09-20, confirmed via code search)

`grep -rn "RoadAnomaly|OoDIS"` across the whole repo returns **nothing** — no loader, no
`config.py` entry, no data folder. Both exist only in the plan doc's prose, not in this
codebase, and are unrelated to the video-demo work above — they're static-image eval
benchmarks for numeric AUROC/AP tables, not anything CARLA/video-pipeline related.
- **RoadAnomaly21** — a second real-photo, eval-only benchmark (same role as Fishyscapes:
  never trained on, just scored against). Nothing built: no download, no dataset class.
- **OoDIS** — Novelty 10, the furthest-out of the two. TwinGuard only outputs a continuous
  per-pixel heatmap (`ood_fused`); OoDIS's instance-level metrics need discrete pseudo-
  instances, so this additionally needs a connected-component post-processing step
  (threshold + blob-labeling) that doesn't exist yet, and that thresholding choice has to
  be stated explicitly wherever OoDIS numbers are reported. Also flagged: OoDIS and
  RoadAnomaly21 both descend from the same SegmentMeIfYouCan benchmark family — possible
  image overlap needs checking before reporting all three (Fishyscapes/RoadAnomaly21/OoDIS)
  as "independent" generalization checks in a results table.
- Both gated behind Section 03's novelties, which are gated behind Experiment B beating
  the baseline gate (see the Calibration checkpoint above) — not scheduled for now.

### Getting video (not just stills) into `frontend/` — three options, one chosen (2026-09-20 discussion, nothing built)

The existing demo is unchanged by everything below. `server.py` runs inference once at
startup over a handful of curated still images, caches every panel (segmentation map, OOD
heatmap, uncertainty heatmap, detection-box overlay) as static PNGs under
`static/generated/`, and `frontend/` does an instant click-through of those cached panels
with no GPU in the loop during the walkthrough. That stays exactly as it is — **video is
additive, not a replacement**, and the still-image click-through is not to be reworked,
degraded, or removed to make room for it.

**Option 1 — baked `.mp4` file. Chosen; this is the one being built.** Same
offline-inference-then-cache-as-static-asset architecture `server.py` already uses, run
over a full recorded CARLA driving sequence (dozens-to-hundreds of frames) instead of 6
curated images, with anomalies inserted by the mixed native-spawn + post-render-CutMix
mechanism described in the "How anomalies actually get into the demo video" sub-section
above (not the RoadAnomaly21/OoDIS one immediately preceding this, which is unrelated
static-image eval work). Per frame, render the same panels
the static demo already renders; then encode the frame sequence to an actual video file
with `cv2.VideoWriter` or `imageio-ffmpeg` (either is fine — both were named in discussion,
pick one and note which). Write the `.mp4` into `static/generated/` so it is served as a
static asset by the same path the PNGs already use, and embed it in `frontend/` with a
plain HTML5 `<video>` tag. No streaming infrastructure, no WebSocket, no live GPU
dependency during the demo. Produce this for **both** Experiment A (baseline) and
Experiment B (ours) — a video of only one of them does not satisfy this item, since the
entire point is the old-vs-new comparison.

**Open decision, not made here:** whether baseline and TwinGuard ship as two separate
toggleable video files or as one side-by-side/split-screen composited video. Both were
raised, neither was chosen. Decide it explicitly and log the choice here before writing the
encoding step — don't let the implementation settle it by accident.

**Option 2 — live WebSocket streaming. Not now; this IS the already-deferred item.**
Recorded frames pushed one-by-one through the FastAPI backend and streamed to `frontend/`
over a WebSocket, panels updating live as though it were a camera feed. This is the
project's eventual end-state demo architecture, and it is already listed under "Already
documented, deliberately deferred — not now" below (Section 08 / Slide 8). Naming it here
only places it in the video-output context; it does not reopen it. It does not start before
Phase 2b/calibration lands — see the "Calibration (L_calib) gate" checkpoint above for the
gate that controls that.

**Option 3 — pre-computed per-frame JSON + client-side playback. Not chosen; logged only
so it isn't re-derived from scratch later.** Same offline batch inference as option 1, but
instead of baking pixels into a compressed video, write one manifest per frame (the exact
pattern `server.py`'s single current `manifest.json` already follows) and have `frontend/`
step through them at real frame-rate in a timed client-side loop — no server round-trips,
no WebSocket, fully static and cacheable, yet it reads as "live" to a viewer. It would also
be genuine groundwork for option 2, since the per-frame panel-rendering path in React would
already exist. Rejected for now on cost/benefit: it is strictly more frontend work than
option 1 for a difference that is cosmetic (feels live vs. is a video file). Nothing about
it is planned, so it carries no action items; revisit only as a deliberate new decision.

**Action items below belong to option 1 only** — options 2 and 3 deliberately have none.

- [ ] Record a CARLA driving sequence with anomalies present in-scene, extending
      `generate_anomalies.py`'s existing per-frame-crop capture mode to keep natively
      spawned `static.prop.*` objects in the scene along the route instead of cropping
      them out, and to write out the full ordered frame sequence.
      → artifact: an on-disk ordered folder of RGB frames + per-frame semantic/anomaly
        masks for one complete route, with the per-instance mechanism metadata
        (native-spawn vs. CutMix-paste) required by the "How anomalies actually get into
        the demo video" sub-section above.
- [ ] Post-render CutMix compositing over that recorded sequence, for the pasted-anomaly
      instances (CARLA and COCO cutouts), reusing `data/cutmix.py`'s paste path so the
      video's anomalies match training-time paste statistics.
      → artifact: the composited frame sequence on disk, plus the paste masks that serve
        as its ground truth.
- [ ] Batch-inference pass over the full sequence producing, per frame, the same panel set
      the static demo already produces (segmentation, OOD heatmap, uncertainty/disagreement
      heatmap, detection-box overlay) — run twice, once per checkpoint (Experiment A and
      Experiment B).
      **Hard dependency, easy to miss:** Checkpoint A's weights are genuinely lost (see
      `CLAUDE.md` — only its logged numbers survive, hardcoded in `server.py`'s
      `EXPERIMENT_HISTORY`), so there is no loadable baseline to run this pass with. The
      Experiment A video cannot be produced until the baseline re-run in the "Blocking"
      section at the top of this file has happened and left a loadable checkpoint on disk.
      This does not weaken the both-videos requirement — it schedules it.
      → artifact: two complete per-frame panel sets on disk, one per checkpoint, frame
        counts matching the recorded sequence.
- [ ] Decide separate-toggle vs. composited split-screen for baseline-vs-TwinGuard, and
      record the decision plus its reasoning in this section.
      → artifact: the decision written into this file before the encoder is wired up.
- [ ] Encode to `.mp4` with `cv2.VideoWriter` or `imageio-ffmpeg` and write the output into
      `static/generated/` alongside the existing cached PNGs.
      → artifact: playable `.mp4` file(s) under `static/generated/`, covering both
        Experiment A and Experiment B per the decision above, with the library actually
        used noted here (the choice between the two is still open).
- [ ] Embed the video in `frontend/` with an HTML5 `<video>` tag, served over the existing
      `/static` proxy, sitting alongside the untouched still-image click-through.
      → artifact: the video playing in the running demo at `localhost:5173` with the
        existing image panels still working unchanged.

## Lower priority / nice-to-have

- [ ] Report AP as the headline metric ("AP is what Fishyscapes actually ranks on" per
      `exp_coco`'s README), AUROC as supporting evidence — update slides/frontend copy
- [ ] Check whether head-disagreement AUROC (now logged on `exp_coco`) is a useful signal
      on its own, not just a display panel
- [ ] Scope the MC-Dropout dual-mode inference trigger (still just "planned")
      — ties to the disagreement-signal check above

## Already documented, deliberately deferred — not now

- Real-time CARLA + WebSocket streaming demo itself (Section 08 / Slide 8) — explicitly
  sequenced to come **after** Phase 2b/calibration, not before. Don't start building this
  early, even though the *reasoning* for why it needs CARLA (above) is settled now.
- ~~Merging `main` and `exp_coco`~~ — done 2026-09-19, see the "Branch context" note at
  the top of this file and `CLAUDE.md`.

---
*Update this file as items complete — check the box only once the artifact column is
verified to exist, not when the code is written.*

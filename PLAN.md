# TwinGuard — next-steps plan

Working checklist. Tick items as they're actually done — "done" means the artifact in
the right-hand column exists on disk, not just that code was written.

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

- [ ] Re-run the baseline (`experiment_a.py`, now on `main`) for the test-half number
      → artifact: MLflow run + printed AUROC/ECE/FPR@95 logged against the **test**
        half. Experiment B's numbers are compared against this.
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
      → still needed: a **training run** with `anomaly_source=both` logged to MLflow,
        AUROC/AP compared against `carla`-only and `coco`-only runs — the bank exists and
        is validated, but hasn't been trained on yet.

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

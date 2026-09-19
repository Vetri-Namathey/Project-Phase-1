# TwinGuard — next-steps plan

Working checklist. Tick items as they're actually done — "done" means the artifact in
the right-hand column exists on disk, not just that code was written.

## Branch context (2026-09-19) — read this first

The audited pipeline (BCE-with-logits fix, drivable-surface placement, photometric
harmonization, val/test split, AP/mIoU, `preflight.py`, `check_runs.py`) lives on
**`origin/exp_coco`**, not `main`. `main` (this checkout) still has the older
pipeline plus the React/FastAPI demo (`frontend/`, `server.py`); `exp_coco` has the
audited training pipeline but **does not have the demo at all** (deleted relative to
main — 59 files differ, `git diff main origin/exp_coco --stat`). These two branches
have diverged hard and will need a deliberate merge, not a fast-forward, once training
work on `exp_coco` is done. Don't assume `main`'s file contents when reasoning about
training-pipeline behavior — check which branch first.

Confirmed already fixed on `exp_coco` (verified directly, not just claimed):
- `data/anomaly_sources.py` — CARLA and COCO already live behind one clean interface,
  gated by `config.ANOMALY_SOURCE = "carla" | "coco"`. COCO is the current default only
  because "the CARLA machine is unavailable" (comment in `config.py`), not a permanent
  choice.
- `CUTMIX_SCALE_MIN/MAX = 0.012 / 0.20` — the oversized-paste bug is already fixed here,
  matching real Fishyscapes anomaly-size statistics ("to within a pixel or two" per the
  code comment).
- Still **not** implemented anywhere on either branch, confirmed via `grep -rn "UBQ\|Hausdorff"`:
  UBQ is objective-only, not built.

## Blocking — do before trusting any Experiment A vs B comparison

- [ ] Re-run the baseline (`experiment_a.py` on `exp_coco`) for the test-half number
      → artifact: MLflow run + printed AUROC/ECE/FPR@95 logged against the **test**
        half. Experiment B's numbers are compared against this.
- [ ] Run `preflight.py` clean, no warnings, immediately before the next paid RunPod run
      → artifact: full preflight stdout log saved alongside that run's MLflow entry

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
- [ ] Curate the CARLA bank: **exclude indices 39, 40, 41, 14, 15** from the object bank
      (the agent's direct recommendation — 5 excluded, 45 kept)
      → artifact: an updated object manifest/filter list in `data/anomaly_sources.py`'s
        CARLA bank construction, with the 5 exclusions and why (mirrors COCO's
        `COCO_EXCLUDED_CATEGORIES` pattern already on `exp_coco`)
- [ ] Add an `ANOMALY_SOURCE = "both"` mode to `data/anomaly_sources.py` — sample from
      curated-CARLA and exclusion-filtered-COCO per paste (not one-or-the-other), aiming
      for ~80 combined varieties. Reduces the risk of the model locking onto either
      source's own low-level statistical signature (CG-render tells vs. COCO-photo tells)
      as a shortcut, on top of just adding volume.
      → artifact: a training run with `anomaly_source=both` logged to MLflow, AUROC/AP
        compared against `carla`-only and `coco`-only runs

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
- Merging `main` and `exp_coco` — needs a deliberate plan (demo vs. training-pipeline
  code have diverged too far for a fast-forward), not something to attempt casually.

---
*Update this file as items complete — check the box only once the artifact column is
verified to exist, not when the code is written.*

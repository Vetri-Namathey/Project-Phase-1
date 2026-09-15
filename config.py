"""Central configuration for TwinGuard Section 12 (Experiment A / B).

Single source of truth for paths, seeds, and hyperparameters shared across
data/, model/, losses.py, metrics.py, and train.py. Nothing from Section 13
(L_calib, MC-Dropout, temporal metrics) lives here yet.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Cityscapes -- confirmed present, two separate roots (images and labels were
# downloaded to different locations). Verified 3475/3475 train+val images have
# a matching gtFine label.
CITYSCAPES_IMAGES_ROOT = r"C:\Users\venka\OneDrive\Documents\AVV\7th sem\Project_Phase_1\leftImg8bit_trainvaltest\leftImg8bit"
CITYSCAPES_LABELS_ROOT = r"C:\Users\venka\Downloads\gtFine_trainvaltest\gtFine"

# Fishyscapes Lost & Found -- confirmed present, two separate roots since the
# OOD labels (Zenodo) and the underlying RGB images (Lost&Found/HF mirror)
# come from different sources and don't share a folder tree. Matching a label
# to its image: strip the "{index}_" prefix and "_labels.png" suffix from the
# label filename, then look for "<that>_leftImg8bit.png" under
# FISHYSCAPES_IMAGES_ROOT/{train,test}/<city>/ -- verified 100/100 match.
FISHYSCAPES_LABELS_DIR = r"C:\Users\venka\Downloads\fishyscapes_lostandfound"
FISHYSCAPES_IMAGES_ROOT = r"C:\Users\venka\Downloads\leftImg8bit\leftImg8bit"

# CARLA-generated OOD objects (already present, from generate_anomalies.py)
CARLA_IMAGES_DIR = "data/images"
CARLA_MASKS_DIR = "data/masks"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_1HEAD = os.path.join(CHECKPOINT_DIR, "model_1head_best.pth")
CHECKPOINT_3HEAD = os.path.join(CHECKPOINT_DIR, "model_3head_best.pth")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Section 10: mit-b2 for local/dev + pipeline validation (RTX 3070 / CPU),
# mit-b5 for the RTX 5080 final training runs. Flip USE_DEV_ENCODER to switch.
ENCODER_NAME_FULL = "nvidia/mit-b5"
ENCODER_NAME_DEV = "nvidia/mit-b2"
USE_DEV_ENCODER = True
ENCODER_NAME = ENCODER_NAME_DEV if USE_DEV_ENCODER else ENCODER_NAME_FULL

NUM_SEG_CLASSES = 19
OOD_HEAD_DROPOUT_P = 0.3

# Section 02: encoder input resolution ("Camera Frame 512x1024 RGB")
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

# Section 12: Experiment A = 1 head, Experiment B = 3 independently seeded heads
OOD_HEAD_SEEDS_1HEAD = [42]
OOD_HEAD_SEEDS_3HEAD = [42, 123, 7]

# ---------------------------------------------------------------------------
# Training (Section 12 — Experiment A and B share identical hyperparameters,
# only the head count differs)
# ---------------------------------------------------------------------------
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"

# Three consecutive fixes aimed at data imbalance (loss reweighting,
# CUTMIX_PROB 0.5->0.8, CUTMIX_SCALE_MIN/MAX raised) all produced the exact
# same collapse signature (mean ~0.0005-0.001, std ~0.008-0.018, unmoved).
# That rules out data density as the cause -- consistent instead with the
# OOD heads saturating early in training (sigmoid'(x)->0 once pre-activations
# go deeply negative, killing the gradient regardless of later batch content)
# because LEARNING_RATE is too large for them specifically. Give them their
# own, 10x lower rate as a separate optimizer param group (see train.py) --
# but ONLY when USE_OOD_HEAD_LR_SPLIT is True.
OOD_HEAD_LEARNING_RATE = 1e-5

# Checkpoint A/B/C plan (RunPod, mit-b5): Checkpoint A isolates the backbone
# variable alone -- LR must stay exactly as already tuned (flat, single
# group), so this defaults False. Only flip True for Checkpoint B, if
# Checkpoint A alone doesn't clear the gate. Toggling this instead of
# hand-editing train.py's optimizer between runs keeps the two checkpoints
# from silently blurring into an unclean, unreportable ablation.
USE_OOD_HEAD_LR_SPLIT = False

# --- Section 12 -> Phase 2b staging: the 0.75 / 0.83 distinction ---------
# 0.75 is NOT the final target. It is the pre-calibration gate confirming
# Experiment B is solid enough to begin Phase 2b (L_calib training). 0.83
# (Experiment A's measured baseline) remains the actual bar Model v3 needs
# to approach or match AFTER L_calib is fully applied. Phase 2b's own design
# already expects a small AUROC cost in exchange for calibration honesty
# (restart if it drops >3% -- see Section 03/04). Cite both numbers together,
# always -- never let 0.75 stand alone as if it were the finish line.
# If asked in a viva why the bar moved: it didn't. Calibration is a stated,
# accounted-for tradeoff, not a lowered target.
PRECALIBRATION_AUROC_GATE = 0.75
POSTCALIBRATION_AUROC_TARGET = 0.83  # = Experiment A's measured baseline
CALIBRATION_TRADEOFF_NOTE = (
    "0.75 is the pre-calibration gate to begin Phase 2b (L_calib); 0.83 "
    "(Experiment A baseline) remains the actual post-calibration target "
    "Model v3 must approach or match. The bar did not move -- calibration "
    "is a stated, accounted-for tradeoff, not a lowered goal."
)

# Section 05: CutMix augmentation probability. Spec states 0.5; raised to 0.8
# to fix Experiment B v2's training instability -- at batch_size=4 and p=0.5,
# ~6.25% of batches carried zero anomaly pixels at all, teaching the OOD
# heads two alternating, conflicting lessons instead of one consistent one
# (see Experiment_B_v2_Update.html). At p=0.8 that drops to ~0.16%, while
# stopping short of 1.0 so the model still sees genuinely clean negatives.
CUTMIX_PROB = 0.8

# Pasted-object scale range (fraction of the image's shorter side, applied to
# the object's larger dimension). Doc Section 02 08-i collapse check found
# both loss reweighting and CUTMIX_PROB=0.8 insufficient -- all 3 heads still
# collapsed to near-zero output (mean ~0.001, std ~0.01-0.02). Roughly
# doubling the linear scale here roughly quadruples anomaly-pixel area per
# pasted object, further reducing the imbalance the loss has to fight.
CUTMIX_SCALE_MIN = 0.30
CUTMIX_SCALE_MAX = 0.55

# Section 03: L_total = L_seg + alpha * L_OOD for Section 12.
# beta * L_calib is Phase 2b (Section 13+) — not used by anything here.
ALPHA_OOD = 1.0

# ---------------------------------------------------------------------------
# Section 12, Experiment A (revised): off-the-shelf baseline, no training.
# Distinct from ENCODER_NAME above -- this is the full finetuned checkpoint,
# used only here, exactly as published.
# ---------------------------------------------------------------------------
BASELINE_MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# "msp" = 1 - max softmax probability, the standard OOD-scoring baseline for
# a model with no dedicated OOD head. Swap to "entropy" here if needed later.
OOD_SCORE_METHOD = "msp"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "twinguard_section12_1head_vs_3head"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Controls data shuffling / general torch seeding. Independent of the
# per-OOD-head seeds above, which exist specifically to keep the heads
# independently initialised from each other.
GLOBAL_SEED = 0

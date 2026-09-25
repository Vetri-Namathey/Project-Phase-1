"""Central configuration for TwinGuard (Experiment A / B).

Single source of truth for paths, seeds, and hyperparameters shared across
data/, model/, losses.py, metrics.py, and train.py. Nothing from Phase 2b
(L_calib, MC-Dropout, temporal metrics) lives here yet.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Where the four datasets live. Override without editing this file:
#
#     Linux/pod :  export TWINGUARD_DATA_ROOT=/workspace/datasets
#     Windows   :  set TWINGUARD_DATA_ROOT=D:\...\gtFine_trainvaltest
#
# Each of the four paths below can also be overridden individually by its own
# env var (same name), for a machine where the layout differs. Paths are
# assembled from separate components rather than written as literal strings,
# so the same config works on Windows and on the Linux pod -- a hardcoded
# "a\b" separator silently becomes part of the filename on Linux.
# NOTE: this default is one developer's local Windows layout, not a path any
# other machine is expected to have. It is a fallback, not a requirement --
# set TWINGUARD_DATA_ROOT (or the four per-path vars) and this value is never
# read. On the pod, setup_env.sh does that for you.
_DEFAULT_DATA_ROOT = r"D:\Academics (D)\SEM-7\PROJECTS\FinalYearProject\gtFine_trainvaltest"
_DATA_ROOT = os.environ.get("TWINGUARD_DATA_ROOT", _DEFAULT_DATA_ROOT)


def _data_path(env_var, *parts):
    override = os.environ.get(env_var)
    return override if override else os.path.join(_DATA_ROOT, *parts)


# Cityscapes -- images and labels were downloaded as two separate archives and
# unpack to sibling trees, hence two roots. Verified 2975/2975 train and
# 500/500 val images have a matching gtFine label.
CITYSCAPES_IMAGES_ROOT = _data_path(
    "CITYSCAPES_IMAGES_ROOT", "leftImg8bit_trainvaltest", "leftImg8bit")
CITYSCAPES_LABELS_ROOT = _data_path(
    "CITYSCAPES_LABELS_ROOT", "gtFine_trainvaltest", "gtFine")

# Fishyscapes Lost & Found -- two separate roots, since the OOD labels
# (Zenodo) and the underlying RGB images (Lost&Found) come from different
# sources and do not share a folder tree. Matching a label to its image:
# strip the "{index}_" prefix and "_labels.png" suffix from the label
# filename, then look for "<that>_leftImg8bit.png" under
# FISHYSCAPES_IMAGES_ROOT/{train,test}/<city>/ -- verified 100/100 match.
FISHYSCAPES_LABELS_DIR = _data_path(
    "FISHYSCAPES_LABELS_DIR", "fishyscapes_lostandfound")
FISHYSCAPES_IMAGES_ROOT = _data_path(
    "FISHYSCAPES_IMAGES_ROOT", "leftImg8bit", "leftImg8bit")

# CARLA-generated OOD objects, from generate_anomalies.py (needs a running
# CARLA server). This remains the canonical anomaly source for the project.
CARLA_IMAGES_DIR = "data/images"
CARLA_MASKS_DIR = "data/masks"

# COCO cutouts, from download_coco_anomalies.py. Originally added as a
# stand-in for the period where no CARLA machine was available; now pooled
# WITH the CARLA bank rather than replacing it -- see ANOMALY_SOURCE below.
COCO_OBJECTS_DIR = "data/coco_objects"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_1HEAD = os.path.join(CHECKPOINT_DIR, "model_1head_best.pth")
CHECKPOINT_3HEAD = os.path.join(CHECKPOINT_DIR, "model_3head_best.pth")

# ---------------------------------------------------------------------------
# Anomaly source (training outlier exposure)
# ---------------------------------------------------------------------------
# "carla" is the canonical source: props rendered in CARLA with pixel-exact
# masks from semantic-tag diffing. "coco" is real-photo object cutouts
# (Common Objects in Context) with every Cityscapes-overlapping category
# removed, which is the standard outlier exposure used by PEBAL /
# DenseHybrid / Mask2Anomaly; it was added while no CARLA server was
# available. "both" pools the curated CARLA bank (45 objects) and the
# filtered COCO bank (3000 cutouts) and samples uniformly per paste, so the
# heads cannot lock onto either source's own low-level signature
# (CG-render tells vs. photo tells) as a shortcut. "both" is the current
# default; it needs BOTH banks present on disk.
#
# Both sources hand CutMix the exact same thing -- an (RGB crop, binary mask)
# pair -- so nothing downstream of data/anomaly_sources.py changes when this
# flips. The value is logged to MLflow on every run, so no result is ever
# ambiguous about which data produced it.
ANOMALY_SOURCE = "both"  # "carla" | "coco" | "both"

# Rebalancing the "both" pool (2026-09-21, PLAN.md checkpoint). Only 45
# distinct CARLA objects exist on disk -- there is no CARLA server on this
# pod to render more -- so reaching CARLA_BANK_TARGET repeats those same 45
# files (CutMix still applies its own random scale/position per paste, so
# it isn't literally the same pixels every time, but it is the same 45
# underlying objects, not new visual diversity). COCO_BANK_TARGET is a
# random, seeded subsample of the full 3000-object COCO bank, so the pool
# actually used for training is CARLA_BANK_TARGET + COCO_BANK_TARGET objects
# ( default 500 + 1500 = 2000 ), not the full 45 + 3000 = 3045.
CARLA_BANK_TARGET = 500
COCO_BANK_TARGET = 1500

# COCO categories that overlap Cityscapes' 19 known classes. These MUST be
# excluded: pasting a COCO car and labelling it "anomaly" would directly
# teach the model that cars are anomalous, and Fishyscapes AUROC would
# collapse. Reviewed by hand against the Cityscapes trainId list.
COCO_EXCLUDED_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# The frozen encoder is taken from a Cityscapes-FINETUNED SegFormer, not from
# the ImageNet-only "nvidia/mit-b*" weights.
#
# Why: nvidia/mit-b5 is a SegformerForImageClassification with 1000 ImageNet
# labels -- it has never seen a street scene, which is exactly the criticism
# that applied to mit-b2. Swapping b2 -> b5 would have tested backbone SIZE,
# not road-adaptation. The road-adapted encoder lives inside the checkpoint
# Experiment A already uses; the architecture is identical (hidden_sizes
# [64,128,320,512], depths [3,6,40,3]), so SegformerModel.from_pretrained()
# on that repo loads the encoder directly.
#
# This also makes Experiment A vs B a fair comparison: both now sit on the
# SAME features, so the question becomes "do 3 trained heads beat max-softmax
# on identical features?" rather than "is a bigger backbone better?".
ENCODER_NAME_FULL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
ENCODER_NAME_DEV = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
USE_DEV_ENCODER = False
ENCODER_NAME = ENCODER_NAME_DEV if USE_DEV_ENCODER else ENCODER_NAME_FULL

NUM_SEG_CLASSES = 19
OOD_HEAD_DROPOUT_P = 0.3

# Input resolution fed to the encoder.
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

# The frozen SegFormer encoder was pretrained on ImageNet-normalised input,
# and its own preprocessor_config.json confirms do_normalize=True with these
# exact statistics. Feeding it raw [0,1] tensors puts it out of distribution,
# and because it is FROZEN it cannot adapt -- the heads then receive features
# with little usable signal. Measured on the b0 Cityscapes checkpoint over
# 10 Fishyscapes images, MSP AUROC only: 0.8812 normalised vs 0.7329 raw.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Experiment A = 1 head, Experiment B = 3 independently seeded heads
OOD_HEAD_SEEDS_1HEAD = [42]
OOD_HEAD_SEEDS_3HEAD = [42, 123, 7]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# 15 was arbitrary. The first full run converged early: mIoU and AP were
# essentially at their final values by epoch 5-6, and epochs 7-15 added
# ~0.02 AP while costing two thirds of the GPU bill. 8 keeps the useful part
# with margin. Raise it only if the metric trend is still clearly climbing
# at the end of a run.
EPOCHS = 8
BATCH_SIZE = 4

# Mixed precision for the forward pass. The frozen b5 encoder dominates
# runtime -- 2976 images per epoch through it -- and bfloat16 roughly halves
# that on an A40 (Ampere has native bf16). bf16 rather than fp16 on purpose:
# it has the same exponent range as fp32, so no gradient scaler and no
# overflow tuning. The OOD heads still accumulate in fp32.
USE_AMP = True

# DataLoader worker processes. 0 means the main process does all decoding and
# CutMix compositing, which leaves the GPU waiting on the CPU -- each sample
# decodes two PNGs, resizes, pastes objects and harmonises them.
#
# Default 0 because Windows spawns (rather than forks) workers, which makes
# multiprocessing DataLoaders fragile locally. On a Linux pod set this to
# something like 8 via the env var -- the survey showed 96 cores idle:
#     export TWINGUARD_NUM_WORKERS=8
# Do not take every core on a shared machine; leave headroom for whoever else
# is using it.
NUM_WORKERS = int(os.environ.get("TWINGUARD_NUM_WORKERS", "0"))
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"

# The OOD heads now train with BCE-with-logits and an explicit pos_weight
# (see losses.py), which removes the sigmoid-saturation path that produced
# the earlier collapse. Kept as a separate knob in case the heads still need
# a gentler rate than the seg head.
OOD_HEAD_LEARNING_RATE = 1e-5
USE_OOD_HEAD_LR_SPLIT = False

# --- Staging gates: the 0.75 / 0.83 distinction --------------------------
# 0.75 is NOT the final target. It is the pre-calibration gate confirming
# Experiment B is solid enough to begin Phase 2b (L_calib training). 0.83
# (Experiment A's measured baseline) remains the actual bar Model v3 needs
# to approach or match AFTER L_calib is fully applied. Phase 2b's own design
# already expects a small AUROC cost in exchange for calibration honesty
# (restart if it drops >3%). Cite both numbers together, always -- never let
# 0.75 stand alone as if it were the finish line.
PRECALIBRATION_AUROC_GATE = 0.75
POSTCALIBRATION_AUROC_TARGET = 0.83  # = Experiment A's measured baseline
CALIBRATION_TRADEOFF_NOTE = (
    "0.75 is the pre-calibration gate to begin Phase 2b (L_calib); 0.83 "
    "(Experiment A baseline) remains the actual post-calibration target "
    "Model v3 must approach or match. The bar did not move -- calibration "
    "is a stated, accounted-for tradeoff, not a lowered goal."
)

# --- Phase 2b: L_calib (calibrate.py) -------------------------------------
# Joint fine-tune of an already-trained checkpoint with a differentiable
# ECE-surrogate term added to the existing loss (see losses.py's
# SoftECELoss). Not post-hoc temperature scaling -- see PLAN.md's
# "Calibration (L_calib) gate" section for why a scalar rescale cannot be
# the answer here (it cannot reshape calibration spatially, which is the
# whole point of beating temperature scaling on boundary-region ECE/UBQ).

# BETA_CALIB was 1.0 for the first real run (2026-09-25) and produced a
# negligible effect: logged calib-loss values were ~0.0002-0.0012 against a
# base (seg+OOD) loss of ~0.05-0.11, meaning L_calib contributed under 1% of
# the gradient -- the fine-tune was effectively just re-running the base
# objective. 50 puts L_calib's weighted contribution in the same order of
# magnitude as the base loss (ratio of base/calib across that run's observed
# range was ~42-550x), enough to actually move the model instead of riding
# along unused. Re-tune from here if it over/under-shoots on the next run.
BETA_CALIB = 50           # weight on L_calib in L_seg + alpha*L_OOD + beta*L_calib
CALIB_EPOCHS = 5          # short fine-tune from an already-converged checkpoint
CALIB_LEARNING_RATE = 1e-5  # same order as OOD_HEAD_LEARNING_RATE -- fine-tuning,
                            # not training from scratch
CALIB_AUROC_DROP_LIMIT = 0.03  # restart if val AUROC falls more than this
                                # relative to the starting checkpoint (see
                                # CALIBRATION_TRADEOFF_NOTE -- a *small*,
                                # accounted-for cost, not an open-ended one)
CHECKPOINT_3HEAD_CALIB = os.path.join(CHECKPOINT_DIR, "model_3head_calib_best.pth")
TEMPERATURE_LEARNING_RATE = 0.01
TEMPERATURE_EPOCHS = 3   # passes over the (small, 50-image) val half to fit
                          # the single scalar T -- cheap, converges fast

# ---------------------------------------------------------------------------
# CutMix anomaly pasting
# ---------------------------------------------------------------------------
CUTMIX_PROB = 0.8

# Pasted-object scale (fraction of the image's shorter side, applied to the
# object's LARGER dimension), sampled LOG-uniformly so small objects dominate.
#
# Measured from all 188 annotated anomaly objects in Fishyscapes L&F, as a
# fraction of the 1024px short side:
#     min 0.007 | p25 0.022 | median 0.043 | p75 0.072 | max 0.406
#
# The previous range (0.30-0.55) did not overlap that distribution AT ALL --
# every training object was larger than ~96% of real test objects. That is
# why three successive "give the heads more anomaly pixels" fixes all
# plateaued: each one moved training further from the test distribution.
# Raising the scale from 0.15-0.35 to 0.30-0.55 is also the change that took
# AUROC 0.6282 -> 0.6193.
# Log-uniform on [lo,hi] has geometric median sqrt(lo*hi). Solved so that
# median lands on the measured real median of 0.043: sqrt(0.012*0.20)=0.049.
# The lower bound also matches the real minimum (0.007 of the short side) to
# within a pixel or two at this input resolution.
CUTMIX_SCALE_MIN = 0.012
CUTMIX_SCALE_MAX = 0.20

# Because objects are now realistically small, paste several per image to
# keep enough positive signal per batch without distorting object scale.
CUTMIX_MIN_OBJECTS = 1
CUTMIX_MAX_OBJECTS = 3

# Real road anomalies sit on the drivable surface. Under uniform placement
# only 33% of pasted objects landed on road -- the rest floated in sky (or
# inside buildings, or on top of cars), teaching the heads a cue that cannot
# transfer. Restrict paste centres to these trainIds: 0=road, 1=sidewalk.
CUTMIX_VALID_SURFACE_TRAINIDS = [0, 1]

# Light photometric harmonisation of the pasted crop toward the local
# brightness/contrast of the region it lands in, plus a soft mask edge.
# Without this, a hard-edged crop with foreign colour statistics is an easy
# shortcut: the heads learn "paste artefact", not "unfamiliar object".
CUTMIX_HARMONIZE = True
CUTMIX_EDGE_FEATHER_PX = 2

# After a pasted object is scaled down, drop mask fragments smaller than this
# many pixels and keep only what remains connected.
#
# COCO annotations are often multi-polygon (a chair seen through its own legs,
# an object split by an occluder), so downscaling can leave a trail of 1-2
# pixel specks labelled "anomaly". Those are unlearnable -- no model can
# detect a one-pixel object -- and they inflate the positive set with pure
# label noise. Measured before this filter: per-object p25 was 0.002 of the
# image short side, i.e. a single pixel.
#
# generate_anomalies.py already applies the same idea on the CARLA side
# (MIN_COMPONENT_PIXELS, keep the largest blob), so this keeps the two
# sources consistent rather than adding a COCO-only special case.
CUTMIX_MIN_OBJECT_PIXELS = 32

# Pixels covered by a pasted object no longer carry a valid Cityscapes class
# (that bench is not "road"), so the segmentation label there is set to
# ignore_index rather than left stale. Prevents the seg head being trained
# on knowingly wrong targets.
CUTMIX_SEG_IGNORE_INDEX = 255

# L_total = L_seg + alpha * L_OOD. beta * L_calib is Phase 2b -- not here.
ALPHA_OOD = 1.0

# Explicit positive-class weight for the OOD BCE term, replacing the old
# per-batch inverse-frequency weight. Fixed rather than batch-dependent so
# the gradient scale does not swing with whatever happened to be pasted into
# the current batch. Capped well below the raw imbalance ratio on purpose:
# with BCE-with-logits the gradient no longer vanishes, so an extreme weight
# is not needed and only destabilises training.
OOD_POS_WEIGHT = 20.0

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# The 100 Fishyscapes images are split deterministically. Checkpoint
# selection reads ONLY the val half; the test half is what gets reported.
# Selecting the best epoch on the same images you report is model selection
# on the test set -- it biases every number even though Fishyscapes never
# enters training, and it is the first thing a reviewer will check given how
# prominently the "never train on the benchmark" rule is stated.
FISHYSCAPES_VAL_FRACTION = 0.5
FISHYSCAPES_SPLIT_SEED = 0

# Which val-half metric picks the best epoch.
#
# "ap", not "auroc". Once the model works, AUROC saturates: across a full
# 15-epoch run it moved only 0.9818-0.9936 (range 0.012) while AP moved
# 0.6802-0.7958 (range 0.116) -- ten times the dynamic range. Selecting on a
# saturated metric is close to selecting on noise, and in the first full run
# it picked epoch 1, which had the WORST AP of all 15 epochs and the worst
# FPR@95 and mIoU too.
#
# AP is also what the Fishyscapes benchmark ranks on, so this is the metric
# the result will be judged by regardless.
SELECTION_METRIC = "ap"  # "ap" | "auroc"

# mIoU is measured on Cityscapes val to catch the OOD heads degrading normal
# segmentation. Capped for speed -- it runs every epoch.
MIOU_EVAL_IMAGES = 100

# Evaluate the TEST half only when the val half produces a new best, plus
# once at the end. Scoring test every epoch costs ~25% of epoch time and
# buys nothing: the only test numbers that get reported are the selected
# checkpoint's. It is also better discipline -- watching a test score climb
# every epoch invites choosing against it.
EVAL_TEST_ON_IMPROVEMENT_ONLY = True

# ---------------------------------------------------------------------------
# Experiment A: off-the-shelf baseline, no training. Distinct from
# ENCODER_NAME above -- this is the full finetuned model including its own
# decode head, used exactly as published.
# ---------------------------------------------------------------------------
BASELINE_MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# "msp" = 1 - max softmax probability, the standard OOD-scoring baseline for
# a model with no dedicated OOD head.
OOD_SCORE_METHOD = "msp"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
# SQLite rather than the "file:./mlruns" directory store. MLflow 3.x refuses
# the filesystem backend outright ("in maintenance mode") unless
# MLFLOW_ALLOW_FILE_STORE=true, so a run that works locally would die on a pod
# with a newer MLflow. SQLite is supported by both old and new versions, and
# the whole history ends up in one file that is trivial to copy off the pod.
#
# Viewing the UI with this backend:
#     mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "twinguard_section12_1head_vs_3head"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Controls data shuffling / general torch seeding. Independent of the
# per-OOD-head seeds above, which exist specifically to keep the heads
# independently initialised from each other -- those now use local
# torch.Generator objects so they no longer mutate the global RNG.
GLOBAL_SEED = 0

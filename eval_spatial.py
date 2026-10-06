"""Boundary-only ECE (Novelty 6) + UBQ, eval-only.

PLAN.md's "Boundary-only ECE (Novelty 6) + UBQ" section, design point 2:
this script is intentionally separate from train.py's evaluate_ood (drives
checkpoint selection, must not change) and calibrate.py's evaluate_fused
(used by fit/finetune/comparison_table). It never trains and never writes
a checkpoint -- forward passes only.

Checkpoints: --raw and --calib are required. config.CHECKPOINT_3HEAD is a
stale training-output path on this machine, and a default once scored the
wrong model (MISTAKES.md M15).

Temperature: on Fishyscapes the whole-image T is fitted here, on the VAL
half, by plain NLL (calibrate.fit_temperatures_from_cache), and printed as
"fitted whole T = x.xxxx". The old default T=1.7142 came from a fit on the
wrong objective (MISTAKES.md M1). CARLA has no val split, so its temp row
exists only when --temp-whole passes the Fishyscapes T; it is never
defaulted.

Thresholds: Fishyscapes fits every row's thresholds (TPR-95 for the legacy
UBQ line, max-F1 for ubq_local) on the VAL half and applies them unchanged
to test. CARLA self-fits on the same frames -- acceptable only for pipeline
checks and paste-on-CARLA results, which are labelled as such.

PLAN.md P1 additions: a head-0-only row with a paired AP bootstrap against
the fused score (rule f), ubq_local per row at its val max-F1 threshold
(rules e, g) and a paired per-image boundary-F1 bootstrap C5 - raw (rule e).
--input-scale (P3) feeds the encoder 1024x512 (1, training resolution),
2048x1024 (2) or the mean of both passes' logits (ms).

Usage:
    python eval_spatial.py --dataset fishyscapes --raw model_3head_best.pth --calib model_3head_calib_best.pth
    python eval_spatial.py --dataset carla --carla-root video_town02_pasted --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole <T>
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from calibrate import fit_temperatures_from_cache, temperature_cache_entry
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from metrics import ScoreHistogram, boundary_band, paired_bootstrap_ap, ubq, ubq_local
from train import amp_context
from utils import get_device, load_trained_model

RADII = (4, 8, 16)
# 1500 = 15 x 100, so ece()'s 15 bins line up exactly with per-image bins.
PER_IMAGE_BINS = 1500
N_BOOTSTRAP = 1000
# ubq_local settings, as in exp_v2: an object is scored only on predictions
# within 32px of it (~ the median real object's longest side at Fishyscapes'
# native resolution), so far false alarms are reported apart, not mixed into
# the outline score. Boundary F1 counts outline pixels within 4px as matched.
UBQ_ROI_PX = 32
UBQ_TOL_PX = 4
# Full-resolution windows (P3 fallback if a whole 2048x1024 pass does not fit
# in 8 GB): two 1024-tall windows, each half the width plus half this
# overlap, logits averaged where they overlap.
WINDOW_OVERLAP_PX = 128


def carla_pairs(root=None):
    """Default: the 45 curated training-bank frames. With root: a recorded
    route from record_route.py (root/rgb/*.png + root/mask/*.npy)."""
    image_dir = os.path.join(root, "rgb") if root else config.CARLA_IMAGES_DIR
    mask_dir = os.path.join(root, "mask") if root else config.CARLA_MASKS_DIR
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
    assert len(images) == len(masks), (
        f"CARLA images/masks must pair 1:1 -- found {len(images)} images, "
        f"{len(masks)} masks.")
    return list(zip(images, masks))


def load_label(mask_path, dataset):
    """-> (anomaly_mask bool (H,W), valid bool (H,W))."""
    if dataset == "carla":
        mask = np.load(mask_path).astype(bool)
        return mask, np.ones_like(mask)  # CLAUDE.md: no 255/void pixels in CARLA masks
    label_map = np.array(Image.open(mask_path))
    return (label_map == 1), (label_map != 255)


@torch.no_grad()
def forward_logits(model, image_path, device, scale=1, window=False):
    """-> (heads, H, W) float32 CPU logits at the model's output resolution,
    which is the input resolution (the heads upsample internally).

    scale 1 is the training resolution and goes through the exact call every
    earlier log used, so P1 can reproduce them. scale 2 feeds 2048x1024;
    window=True runs it as two overlapping 1024-tall windows instead of one
    pass, for when the whole image does not fit in 8 GB (P0b decides).
    """
    if scale == 1:
        assert not window, "windows exist only for the scale-2 memory fallback"
        image = load_image_tensor(image_path, device)
    else:
        image = load_image_tensor(image_path, device,
                                  size=(config.INPUT_WIDTH * scale, config.INPUT_HEIGHT * scale))
    if not window:
        with amp_context(device):
            out = model(image)
        return out["ood_logits"].float().squeeze(0).cpu()

    width = image.shape[-1]
    half, pad = width // 2, WINDOW_OVERLAP_PX // 2
    total, count = None, torch.zeros(width, device=device)
    for x0, x1 in ((0, half + pad), (half - pad, width)):
        with amp_context(device):
            part = model(image[..., x0:x1])["ood_logits"].float().squeeze(0)
        if total is None:
            total = torch.zeros(part.shape[0], part.shape[1], width, device=device)
        total[..., x0:x1] += part
        count[x0:x1] += 1
    return (total / count).cpu()


@torch.no_grad()
def cache_logits(model, pairs, device, dataset, input_scale="1", window=False):
    """One forward pass per image (two for "ms"). Logits are kept at the
    model's output resolution -- 3 heads x 512x1024 at scale 1, ~6MB per
    image -- and upsampled on demand. At scale 2 / ms they are kept at
    1024x2048 (~25MB per image per model), so those modes need several GB of
    free RAM for the val + test caches."""
    cached = []
    model.eval()
    for image_path, mask_path in pairs:
        anomaly_mask, valid = load_label(mask_path, dataset)
        if input_scale == "ms":
            # Mean of the per-head LOGITS of both passes, at label resolution
            # (PLAN.md P3 "ms"), so fusion and calibration see one logit map.
            both = [F.interpolate(forward_logits(model, image_path, device, s, window and s == 2)
                                  .unsqueeze(0), size=anomaly_mask.shape, mode="bilinear",
                                  align_corners=False).squeeze(0) for s in (1, 2)]
            logits = (both[0] + both[1]) / 2
        else:
            scale = int(input_scale)
            logits = forward_logits(model, image_path, device, scale, window and scale == 2)
        cached.append((logits, anomaly_mask, valid))
    return cached


def head0(cached):
    """Head 0's logits alone. Fused through the same mean-of-sigmoids path,
    a 1-element mean is head 0's own sigmoid -- the free 1-head proxy
    (PLAN.md P1 rule f): the heads share no parameters and the encoder is
    frozen, so head 0 is what a 1-head model on these features would see."""
    return [(logits[:1], mask, valid) for logits, mask, valid in cached]


# "logit" matches calibrate.py's evaluate_fused and train.py. "prob" is PLAN.md
# C1b: does upsampling logits (dominated by very negative background logits)
# cause part of the boundary under-confidence? Set from --upsample in main().
UPSAMPLE_MODE = "logit"


class DisagreementTemperature:
    """PLAN.md C3: per-pixel T(d) = exp(a + b*d), d = std of the 3 heads'
    raw sigmoid scores (the model's own ood_disagreement). Needs no ground
    truth at test time, unlike band membership."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, d):
        return torch.exp(self.a + self.b * d)

    def __repr__(self):
        return f"T(d)=exp({self.a:.3f} {self.b:+.3f}*d)"


class ContextCalibrator:
    """PLAN.md C5 ("spatial Platt scaling"): p = sigmoid(w . [1, z, max(z), mean(z), d])
    over a (2r+1)^2 neighbourhood. z = logit of the raw fused score, d = head
    disagreement. Uses only the score map, so no ground truth at test time."""

    def __init__(self, weights, radius=8):
        self.weights = weights  # tensor (5,)
        self.radius = radius

    def features(self, up):
        """up: (1, heads, H, W) upsampled raw logits -> (4, H, W)."""
        probs = torch.sigmoid(up)
        fused = probs.mean(dim=1, keepdim=True)
        d = probs.std(dim=1, unbiased=False, keepdim=True)
        z = torch.logit(fused.clamp(1e-6, 1 - 1e-6))
        k, r = 2 * self.radius + 1, self.radius
        z_max = F.max_pool2d(z, k, stride=1, padding=r)
        z_mean = F.avg_pool2d(z, k, stride=1, padding=r, count_include_pad=False)
        return torch.cat([z, z_max, z_mean, d], dim=1).squeeze(0)

    def logit(self, feats):
        """feats: (4, ...) -> calibrated logit (...)."""
        w = self.weights
        return w[0] + (w[1:].view(-1, *([1] * (feats.dim() - 1))) * feats).sum(dim=0)

    def scores(self, up):
        return torch.sigmoid(self.logit(self.features(up))).numpy()

    def __repr__(self):
        w = [f"{x:+.3f}" for x in self.weights.tolist()]
        return f"C5 w=[bias {w[0]}, z {w[1]}, max {w[2]}, mean {w[3]}, d {w[4]}]"


def fused_scores(logits, shape, temperature=1.0):
    """Default order, same as calibrate.py's evaluate_fused: upsample logits,
    then per-head sigmoid(logit/T), then mean across heads. `temperature`
    may be a scalar, a DisagreementTemperature (C3) or a ContextCalibrator (C5)."""
    if UPSAMPLE_MODE == "prob":
        if not isinstance(temperature, (int, float)):
            raise SystemExit("C3/C5 are only defined for --upsample logit")
        probs = torch.sigmoid(logits / temperature).unsqueeze(0)
        up = F.interpolate(probs, size=shape, mode="bilinear", align_corners=False)
        return up.mean(dim=1).squeeze(0).numpy()
    up = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)
    if isinstance(temperature, ContextCalibrator):
        return temperature.scores(up)
    if callable(temperature):
        d = torch.sigmoid(up).std(dim=1, unbiased=False, keepdim=True)
        temperature = temperature(d)
    return torch.sigmoid(up / temperature).mean(dim=1).squeeze(0).numpy()


def fit_disagreement_temperature(cached, bands, radius=8, per_image_sample=20000,
                                 use_disagreement=True):
    """PLAN.md C3, fit on the val half only. Objective, fixed before seeing any
    C3 number: 0.5 * per-head BCE over a uniform sample of all valid pixels
    (the whole-image term, which C2 sacrificed) + 0.5 * per-head BCE over the
    r-band pixels (the boundary term, which whole-image T sacrificed).
    Equal weighting is a stated choice, not tuned."""
    rng = np.random.default_rng(config.GLOBAL_SEED)
    parts = {"u": ([], [], []), "b": ([], [], [])}
    for (logits, mask, valid), img_bands in zip(cached, bands):
        up = F.interpolate(logits.unsqueeze(0), size=mask.shape, mode="bilinear",
                           align_corners=False).squeeze(0)
        up_flat = up.reshape(up.shape[0], -1)
        d_flat = torch.sigmoid(up).std(dim=0, unbiased=False).reshape(-1)
        y_flat = torch.from_numpy(mask.reshape(-1).astype(np.float32))
        flat_valid = np.flatnonzero(valid)
        pick = torch.from_numpy(rng.choice(flat_valid, size=min(per_image_sample, flat_valid.size),
                                           replace=False))
        band = torch.from_numpy(np.flatnonzero(img_bands[radius]))
        for key, idx in (("u", pick), ("b", band)):
            parts[key][0].append(up_flat[:, idx])
            parts[key][1].append(y_flat[idx])
            parts[key][2].append(d_flat[idx])
    data = {k: tuple(torch.cat(v, dim=-1) for v in vals) for k, vals in parts.items()}
    if use_disagreement:
        print(f"C3 diagnostic: mean head disagreement in r={radius} band = "
              f"{data['b'][2].mean():.4f} vs whole-image sample = {data['u'][2].mean():.4f}")

    # use_disagreement=False is C3's control: a single scalar T on the identical
    # objective and data. b stays 0 and is never optimized.
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=use_disagreement)
    opt = torch.optim.LBFGS([a, b] if use_disagreement else [a], lr=0.1, max_iter=300)

    def bce(x, y, d):
        return F.binary_cross_entropy_with_logits(x / torch.exp(a + b * d), y.expand_as(x))

    def closure():
        opt.zero_grad()
        loss = 0.5 * bce(*data["u"]) + 0.5 * bce(*data["b"])
        loss.backward()
        return loss

    opt.step(closure)
    return DisagreementTemperature(float(a.item()), float(b.item()))


def fit_context_calibrator(cached, bands, radius=8, per_image_sample=20000):
    """PLAN.md C5, fit on the val half only. Identical objective and pixel
    sampling (same seed) as fit_disagreement_temperature, so C5 vs the C3
    control compares calibrator families, not data or objectives."""
    rng = np.random.default_rng(config.GLOBAL_SEED)
    calib = ContextCalibrator(torch.zeros(5), radius=radius)
    parts = {"u": ([], []), "b": ([], [])}
    for (logits, mask, valid), img_bands in zip(cached, bands):
        up = F.interpolate(logits.unsqueeze(0), size=mask.shape, mode="bilinear",
                           align_corners=False)
        feats = calib.features(up).reshape(4, -1)
        y_flat = torch.from_numpy(mask.reshape(-1).astype(np.float32))
        flat_valid = np.flatnonzero(valid)
        pick = torch.from_numpy(rng.choice(flat_valid, size=min(per_image_sample, flat_valid.size),
                                           replace=False))
        band = torch.from_numpy(np.flatnonzero(img_bands[radius]))
        for key, idx in (("u", pick), ("b", band)):
            parts[key][0].append(feats[:, idx])
            parts[key][1].append(y_flat[idx])
    data = {k: (torch.cat(f, dim=1), torch.cat(y)) for k, (f, y) in parts.items()}

    # Start at the identity calibrator (p = sigmoid(z)) so LBFGS begins from "raw".
    w = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
    calib.weights = w
    opt = torch.optim.LBFGS([w], lr=0.1, max_iter=300)

    def closure():
        opt.zero_grad()
        loss = sum(0.5 * F.binary_cross_entropy_with_logits(calib.logit(f), y)
                   for f, y in data.values())
        loss.backward()
        return loss

    opt.step(closure)
    calib.weights = w.detach()
    return calib


def fit_band_temperature(cached, bands, radius=8):
    """PLAN.md C2: plain (unweighted) per-head BCE, restricted to r-band
    pixels of the given (val) images, fit in log-space with LBFGS so T stays
    positive. Same objective as calibrate.fit_temperatures' band fit (plain
    NLL), different optimiser. It is NOT the objective of the old
    calibrate.fit_temperature, which carried pos_weight=20 -- this docstring
    used to claim it was, which hid that the "edges want 4.4, image wants
    1.7" contrast compared two different objectives (MISTAKES.md M14)."""
    xs, ys = [], []
    for (logits, mask, _), img_bands in zip(cached, bands):
        up = F.interpolate(logits.unsqueeze(0), size=mask.shape, mode="bilinear",
                           align_corners=False).squeeze(0)
        sel = torch.from_numpy(img_bands[radius])
        xs.append(up[:, sel])
        ys.append(torch.from_numpy(mask[img_bands[radius]].astype(np.float32)))
    x = torch.cat(xs, dim=1)
    y = torch.cat(ys).expand_as(x)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def compute_bands(cached, radii=RADII):
    """Bands depend only on ground truth, so compute once and share across models."""
    return [{r: boundary_band(mask, radius_px=r, valid=valid) for r in radii}
            for _, mask, valid in cached]


def whole_histogram(cached, temperature):
    hist = ScoreHistogram()
    for logits, mask, valid in cached:
        scores = fused_scores(logits, mask.shape, temperature)
        hist.update(scores[valid], mask[valid].astype(np.int64))
    return hist


def histogram_from_arrays(pos, neg, cps, cpn):
    hist = ScoreHistogram(n_bins=PER_IMAGE_BINS)
    hist.pos, hist.neg, hist.conf_sum_pos, hist.conf_sum_neg = pos, neg, cps, cpn
    return hist


def evaluate_model(name, cached, bands, temperature, threshold_source=None, radii=RADII,
                   keep_ap=False):
    """Prints one row block. Returns per-image data for the paired
    bootstraps: band/whole ECE histograms, "bf1" (per-image mean boundary F1
    over found objects, NaN if none found) and, with keep_ap, "ap" =
    per-image (pos, neg) counts at full histogram resolution for
    paired_bootstrap_ap (~3 MB per image, so only rows that need it)."""
    whole = ScoreHistogram()
    band_hists = {r: ScoreHistogram() for r in radii}
    band_counts = {r: [0, 0] for r in radii}
    # PLAN.md C1: signed direction of band miscalibration. Inside the band, the
    # GT-anomaly pixels are exactly the inner half and the rest the outer half
    # (signed distance is never 0 on the pixel grid), so no extra band compute.
    score_sums = {r: {"both": 0.0, "inner": 0.0, "outer": 0.0} for r in radii}
    per_image = {r: [] for r in radii}
    per_image["whole"] = []  # whole-image ECE bootstrap (C4 rule needs it)
    per_image["ap"] = []
    score_maps = []

    for (logits, mask, valid), img_bands in zip(cached, bands):
        scores = fused_scores(logits, mask.shape, temperature)
        score_maps.append(scores)
        whole.update(scores[valid], mask[valid].astype(np.int64))
        if keep_ap:
            full = ScoreHistogram()
            full.update(scores[valid], mask[valid].astype(np.int64))
            per_image["ap"].append((full.pos, full.neg))
        small_whole = ScoreHistogram(n_bins=PER_IMAGE_BINS)
        small_whole.update(scores[valid], mask[valid].astype(np.int64))
        per_image["whole"].append(small_whole)
        for r in radii:
            band = img_bands[r]
            s, y = scores[band], mask[band].astype(np.int64)
            band_hists[r].update(s, y)
            small = ScoreHistogram(n_bins=PER_IMAGE_BINS)
            small.update(s, y)
            per_image[r].append(small)
            band_counts[r][0] += int(band.sum())
            band_counts[r][1] += int(y.sum())
            inner = y == 1
            score_sums[r]["both"] += float(s.sum())
            score_sums[r]["inner"] += float(s[inner].sum())
            score_sums[r]["outer"] += float(s[~inner].sum())

    threshold_hist = threshold_source if threshold_source is not None else whole
    threshold = threshold_hist.threshold_at_tpr(0.95)

    fields = {"pred_to_gt_px": [], "gt_to_pred_px": [], "pred_to_gt_p95": []}
    pred_fracs, skipped = [], 0
    for scores, (_, mask, valid) in zip(score_maps, cached):
        pred_mask = (scores >= threshold) & valid
        pred_fracs.append(pred_mask.sum() / max(valid.sum(), 1))
        result = ubq(pred_mask, mask, valid=valid)
        if any(np.isnan(v) for v in result.values()):
            skipped += 1
            continue
        for k in fields:
            fields[k].append(result[k])
    ubq_means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in fields.items()}

    # ubq_local at this row's own max-F1 threshold (fit on the val half for
    # Fishyscapes), so every row is compared at its own matched operating
    # point -- not at TPR-95 thresholds flagging different frame fractions
    # (MISTAKES.md M11).
    f1_threshold = threshold_hist.threshold_at_max_f1()
    objects, far_px, pred_px, flagged = [], 0, 0, []
    per_image["bf1"] = []
    for scores, (_, mask, valid) in zip(score_maps, cached):
        pred = (scores >= f1_threshold) & valid
        flagged.append(pred.sum() / max(valid.sum(), 1))
        objs, f, p = ubq_local(pred, mask, valid=valid, roi_px=UBQ_ROI_PX, tolerance_px=UBQ_TOL_PX)
        objects += objs
        far_px += f
        pred_px += p
        found_here = [o["boundary_f1"] for o in objs if o["detected"]]
        per_image["bf1"].append(float(np.mean(found_here)) if found_here else float("nan"))
    found = [o for o in objects if o["detected"]]
    local = {k: (float(np.mean([o[k] for o in found])) if found else float("nan"))
             for k in ("spill_px", "miss_px", "boundary_f1")}

    t_desc = f"{temperature:.4f}" if isinstance(temperature, (int, float)) else repr(temperature)
    print(f"\n=== {name} (temperature={t_desc}) ===")
    m = whole.summary()
    # Whole ECE at 6 dp: after temperature scaling it is ~0.0001, where 4 dp
    # cannot support any comparison (eval_fishyscapes_T122.log printed 0.0001).
    print(f"  whole-image  AUROC={m['auroc']:.4f} AP={m['ap']:.4f} "
          f"FPR@95={m['fpr95']:.4f} ECE={m['ece']:.6f}")
    for r in radii:
        bm = band_hists[r].summary()
        total, pos = band_counts[r]
        print(f"  band r={r:<2} AUROC={bm['auroc']:.4f} AP={bm['ap']:.4f} "
              f"FPR@95={bm['fpr95']:.4f} ECE={bm['ece']:.4f}  "
              f"(n_px={total}, pos_rate={pos / total if total else float('nan'):.4f})")
    for r in radii:
        total, pos = band_counts[r]
        neg = total - pos
        mean_both = score_sums[r]["both"] / total if total else float("nan")
        pos_rate = pos / total if total else float("nan")
        mean_inner = score_sums[r]["inner"] / pos if pos else float("nan")
        mean_outer = score_sums[r]["outer"] / neg if neg else float("nan")
        direction = "UNDER-confident" if mean_both < pos_rate else "OVER-confident"
        print(f"  C1 r={r:<2} mean_score={mean_both:.4f} vs pos_rate={pos_rate:.4f} "
              f"(gap {mean_both - pos_rate:+.4f}, {direction})  "
              f"inner(true anomaly) mean={mean_inner:.4f}  outer(true normal) mean={mean_outer:.4f}")
    src = "val half" if threshold_source is not None else "same set (self-fit)"
    print(f"  UBQ legacy (frame-wide max, saturated -- read ubq_local below) "
          f"@ threshold_at_tpr(0.95)={threshold:.5f} [fit on {src}]  "
          f"pred_to_gt_px={ubq_means['pred_to_gt_px']:.2f}  "
          f"gt_to_pred_px={ubq_means['gt_to_pred_px']:.2f}  "
          f"pred_to_gt_p95={ubq_means['pred_to_gt_p95']:.2f}  "
          f"mean predicted-positive fraction={np.mean(pred_fracs):.4f}  "
          f"(skipped {skipped}/{len(cached)} images with an empty mask)")
    if np.mean(pred_fracs) > 0.2:
        print("  WARNING: threshold marks >20% of the frame as anomalous -- UBQ is "
              "saturated at this threshold and does not measure boundary quality.")
    n_obj = len(objects)
    print(f"  ubq_local @ max-F1 threshold={f1_threshold:.5f} [fit on {src}] "
          f"(roi={UBQ_ROI_PX}px, boundary-F1 tol={UBQ_TOL_PX}px): "
          f"objects found {len(found)}/{n_obj} ({len(found) / n_obj if n_obj else float('nan'):.1%})  "
          f"boundary_F1={local['boundary_f1']:.3f}  spill_px={local['spill_px']:.2f}  "
          f"miss_px={local['miss_px']:.2f}  (means over found objects)  "
          f"far-FP={far_px / pred_px if pred_px else float('nan'):.1%} of flagged px  "
          f"flagged={np.mean(flagged):.3%} of valid px")
    return per_image


def paired_bootstrap(per_image_a, per_image_b, label, radii=RADII, seed=config.GLOBAL_SEED):
    """95% CI of band-ECE(a) - band-ECE(b), resampling the SAME image indices
    for both models each round (paired)."""
    rng = np.random.default_rng(seed)
    print(f"\npaired bootstrap, {N_BOOTSTRAP} resamples: band-ECE({label})")
    for r in radii:
        stacks = []
        for hists in (per_image_a[r], per_image_b[r]):
            stacks.append(tuple(np.stack([getattr(h, f) for h in hists])
                                for f in ("pos", "neg", "conf_sum_pos", "conf_sum_neg")))
        n = len(per_image_a[r])
        full = [histogram_from_arrays(*(a.sum(0) for a in s)).ece() for s in stacks]
        diffs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            eces = [histogram_from_arrays(*(a[idx].sum(0) for a in s)).ece() for s in stacks]
            diffs[i] = eces[0] - eces[1]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        where = "whole-image" if r == "whole" else f"r={r:<2}"
        print(f"  {where} diff={full[0] - full[1]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def paired_block_bootstrap(per_image_a, per_image_b, label, block, radii=RADII,
                           seed=config.GLOBAL_SEED):
    """CARLA routes only: paired CIRCULAR block bootstrap of band-ECE(a) -
    band-ECE(b). Consecutive route frames are ~1 m apart and show the same
    objects for many frames, so resampling single frames treats correlated
    frames as independent and gives CIs that are too narrow (MISTAKES.md
    M20). Each resample draws ceil(n/block) start frames uniformly and takes
    `block` consecutive frames from each, wrapping past the last frame back
    to the first (circular, so every frame is equally likely to be drawn),
    truncated to n frames. Frames are in route order (carla_pairs sorts by
    filename). Same pairing as paired_bootstrap: one index set for both."""
    rng = np.random.default_rng(seed)
    print(f"\npaired CIRCULAR BLOCK bootstrap ({block}-frame blocks), {N_BOOTSTRAP} resamples: "
          f"band-ECE({label})")
    for r in radii:
        stacks = []
        for hists in (per_image_a[r], per_image_b[r]):
            stacks.append(tuple(np.stack([getattr(h, f) for h in hists])
                                for f in ("pos", "neg", "conf_sum_pos", "conf_sum_neg")))
        n = len(per_image_a[r])
        n_blocks = -(-n // block)
        full = [histogram_from_arrays(*(a.sum(0) for a in s)).ece() for s in stacks]
        diffs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            starts = rng.integers(0, n, n_blocks)
            idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
            eces = [histogram_from_arrays(*(a[idx].sum(0) for a in s)).ece() for s in stacks]
            diffs[i] = eces[0] - eces[1]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        where = "whole-image" if r == "whole" else f"r={r:<2}"
        print(f"  {where} diff={full[0] - full[1]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def print_ap_bootstrap(per_image_a, per_image_b, label):
    """Paired image bootstrap of whole-image AP(a) - AP(b)."""
    res = paired_bootstrap_ap(per_image_a["ap"], per_image_b["ap"],
                              n_resamples=N_BOOTSTRAP, seed=config.GLOBAL_SEED)
    verdict = "CI excludes 0" if (res["lo"] > 0 or res["hi"] < 0) else "CI includes 0"
    print(f"\npaired bootstrap, {N_BOOTSTRAP} resamples: AP({label})")
    print(f"  AP {res['ap_a']:.4f} vs {res['ap_b']:.4f}  diff={res['diff']:+.4f}  "
          f"95% CI [{res['lo']:+.4f}, {res['hi']:+.4f}]  ({verdict})")


def paired_bootstrap_bf1(per_image_a, per_image_b, label, seed=config.GLOBAL_SEED):
    """PLAN.md P1 rule (e): per-image mean boundary F1, a - b, paired over
    the images where BOTH rows found at least one object at their own val
    max-F1 threshold (an image with no found object has no outline to
    score)."""
    a, b = np.asarray(per_image_a["bf1"]), np.asarray(per_image_b["bf1"])
    keep = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[keep], b[keep]
    print(f"\npaired bootstrap, {N_BOOTSTRAP} resamples: per-image boundary F1({label}) "
          f"over {len(a)}/{len(keep)} images where both rows found >= 1 object")
    if len(a) == 0:
        print("  no such image -- nothing to compare")
        return
    rng = np.random.default_rng(seed)
    diffs = np.array([(a[idx] - b[idx]).mean()
                      for idx in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOTSTRAP))])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
    print(f"  mean {a.mean():.4f} vs {b.mean():.4f}  diff={(a - b).mean():+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["carla", "fishyscapes"], required=True)
    parser.add_argument("--raw", required=True,
                        help=f"raw checkpoint path, e.g. {config.PRIMARY_RAW} (explicit, MISTAKES.md M15)")
    parser.add_argument("--calib", required=True,
                        help=f"L_calib checkpoint path, e.g. {config.PRIMARY_CALIB}")
    parser.add_argument("--temp-whole", type=float, default=None,
                        help="carla only: whole-image T fitted on Fishyscapes val by an earlier "
                             "fishyscapes run. Without it the carla temp row is omitted.")
    parser.add_argument("--carla-root", default=None,
                        help="score a record_route.py output folder instead of the 45 training-bank frames")
    parser.add_argument("--upsample", choices=["logit", "prob"], default="logit",
                        help="PLAN.md C1b: interpolate logits (default, matches training eval) "
                             "or probabilities")
    parser.add_argument("--input-scale", choices=["1", "2", "ms"], default="1",
                        help="PLAN.md P3: encoder input 1024x512 (1, training resolution), "
                             "2048x1024 (2), or the mean of both passes' logits (ms)")
    parser.add_argument("--block-bootstrap", type=int, default=None, metavar="N",
                        help="carla only: paired circular block bootstrap over N consecutive "
                             "frames (PLAN.md P1b pre-registers N=20)")
    parser.add_argument("--window", action="store_true",
                        help="run scale 2 as two overlapping windows (only if P0b chose windows)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset == "fishyscapes" and args.temp_whole is not None:
        parser.error("fishyscapes fits its own whole-image T on the val half; "
                     "--temp-whole is for --dataset carla only")
    if args.block_bootstrap is not None and (args.dataset != "carla" or args.block_bootstrap < 1):
        parser.error("--block-bootstrap N (N >= 1) is for --dataset carla only")
    if args.window and args.input_scale == "1":
        parser.error("--window applies to --input-scale 2 or ms only")
    global UPSAMPLE_MODE
    UPSAMPLE_MODE = args.upsample
    print(f"upsample mode: {UPSAMPLE_MODE}")
    print(f"input scale: {args.input_scale}{' (windowed)' if args.window else ''}  "
          f"raw={args.raw}  calib={args.calib}")

    device = get_device()
    raw_model = load_trained_model(args.raw, device)
    calib_model = load_trained_model(args.calib, device)
    cache = lambda model, pairs: cache_logits(model, pairs, device, args.dataset,
                                              args.input_scale, args.window)

    thresholds = {"raw": None, "temp": None, "calib": None}
    temperature = args.temp_whole
    if args.dataset == "carla":
        pairs = carla_pairs(args.carla_root)
        if args.carla_root:
            print(f"CARLA recorded route {args.carla_root}: {len(pairs)} frames "
                  f"(PLAN.md V0 pilot). Frames with no visible prop add background-only pixels.")
        else:
            print(f"CARLA: {len(pairs)} frames. NOTE per PLAN.md Step 3 caveats -- these "
                  f"objects are in the training CutMix bank and CARLA backgrounds are "
                  f"out-of-domain for a Cityscapes-trained model. Pipeline validation only, "
                  f"not generalization evidence.")
        if temperature is None:
            print("temp row omitted: CARLA has no val split to fit T on, and T is never "
                  "defaulted (MISTAKES.md M1). Pass --temp-whole <T from the fishyscapes run>.")
    else:
        all_pairs = list_fishyscapes_pairs()
        assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
        fishy_val, pairs = split_fishyscapes_pairs(all_pairs)
        print(f"Fishyscapes: fitting T, calibrators and thresholds on {len(fishy_val)} val "
              f"images, evaluating on {len(pairs)} test images.")
        raw_val = cache(raw_model, fishy_val)
        calib_val = cache(calib_model, fishy_val)

        # P1 rule (a): whole-image T by plain NLL on val, on the same
        # label-resolution logits every row below is scored from.
        temp_cache = [temperature_cache_entry(
            F.interpolate(logits.unsqueeze(0), size=mask.shape, mode="bilinear",
                          align_corners=False).squeeze(0), mask, valid)
            for logits, mask, valid in raw_val]
        temps = fit_temperatures_from_cache(temp_cache)
        del temp_cache
        temperature = temps["whole"]
        print(f"fitted whole T = {temperature:.4f}  (plain NLL, Fishyscapes val half)")
        print(f"fitted band T (r={config.TEMPERATURE_BAND_RADIUS_PX}, plain NLL) = "
              f"{temps['band']:.4f}  (reference only; the C2 row uses its own LBFGS fit below)")

        val_bands = compute_bands(raw_val, radii=(8,))
        band_temperature = fit_band_temperature(raw_val, val_bands)
        print(f"C2: band-fitted temperature (val half, r=8 band pixels only) = "
              f"{band_temperature:.4f}  (whole-image fit is {temperature:.4f})")
        thresholds = {
            "raw": whole_histogram(raw_val, 1.0),
            "temp": whole_histogram(raw_val, temperature),
            "temp-band": whole_histogram(raw_val, band_temperature),
            "calib": whole_histogram(calib_val, 1.0),
            "head0": whole_histogram(head0(raw_val), 1.0),
        }
        if UPSAMPLE_MODE == "logit":
            disagree_temperature = fit_disagreement_temperature(raw_val, val_bands)
            print(f"C3: disagreement-conditioned temperature (val half) = {disagree_temperature!r}  "
                  f"-> T at d=0: {float(disagree_temperature(torch.tensor(0.0))):.4f}, "
                  f"at d=0.1: {float(disagree_temperature(torch.tensor(0.1))):.4f}, "
                  f"at d=0.3: {float(disagree_temperature(torch.tensor(0.3))):.4f}")
            thresholds["temp-disagree"] = whole_histogram(raw_val, disagree_temperature)
            control_temperature = float(fit_disagreement_temperature(
                raw_val, val_bands, use_disagreement=False)(torch.tensor(0.0)))
            print(f"C3 control: single T on the same 50/50 objective = {control_temperature:.4f}")
            thresholds["temp-mixed"] = whole_histogram(raw_val, control_temperature)
            context_calibrator = fit_context_calibrator(raw_val, val_bands)
            print(f"C5: {context_calibrator!r}")
            thresholds["context"] = whole_histogram(raw_val, context_calibrator)
        del raw_val, calib_val, val_bands

    raw_cached = cache(raw_model, pairs)
    calib_cached = cache(calib_model, pairs)
    bands = compute_bands(raw_cached)

    # Per-image AP counts only where an AP bootstrap reads them (head 0 vs
    # fused, Fishyscapes) -- at ~3 MB per image they would cost >1 GB on a
    # 400-frame CARLA route.
    per_raw = evaluate_model("raw", raw_cached, bands, 1.0, thresholds["raw"],
                             keep_ap=args.dataset == "fishyscapes")
    per_temp = None
    if temperature is not None:
        source = "fitted on val above" if args.dataset == "fishyscapes" else "--temp-whole"
        per_temp = evaluate_model(f"temp-scaled (whole-image T={temperature:.4f}, {source})",
                                  raw_cached, bands, temperature, thresholds["temp"])
    per_calib = evaluate_model("L_calib", calib_cached, bands, 1.0, thresholds["calib"])

    if per_temp is not None:
        paired_bootstrap(per_calib, per_temp, f"L_calib) - band-ECE(temp T={temperature:.4f}")
    paired_bootstrap(per_calib, per_raw, "L_calib) - band-ECE(raw")
    if args.block_bootstrap is not None:
        # Carla only (checked above). The frame-level lines just printed are
        # kept for comparison with eval_video_town02.log; P1b's rule reads
        # these block lines.
        if per_temp is not None:
            paired_block_bootstrap(per_calib, per_temp,
                                   f"L_calib) - band-ECE(temp T={temperature:.4f}",
                                   args.block_bootstrap)
        paired_block_bootstrap(per_calib, per_raw, "L_calib) - band-ECE(raw", args.block_bootstrap)

    if args.dataset == "fishyscapes":
        per_tband = evaluate_model("temp-band (C2)", raw_cached, bands, band_temperature,
                                   thresholds["temp-band"])
        paired_bootstrap(per_tband, per_temp, f"temp-band) - band-ECE(temp T={temperature:.4f}")
        paired_bootstrap(per_tband, per_raw, "temp-band) - band-ECE(raw")
        paired_bootstrap(per_tband, per_temp, f"temp-band) - ECE(temp T={temperature:.4f}",
                         radii=("whole",))

        if UPSAMPLE_MODE == "logit":
            per_c3 = evaluate_model("temp-disagreement (C3)", raw_cached, bands,
                                    disagree_temperature, thresholds["temp-disagree"])
            # C4 decision rule (PLAN.md, fixed before any C3 number): C3 must beat
            # whole-image temp at the boundary AND beat temp-band on whole-image ECE.
            paired_bootstrap(per_c3, per_temp, f"C3) - ECE(temp T={temperature:.4f}  "
                             f"[C4 condition 1: r=8 must be < 0]", radii=(4, 8, 16, "whole"))
            paired_bootstrap(per_c3, per_tband, "C3) - ECE(temp-band  [C4 condition 2: whole-image must be < 0]",
                             radii=(4, 8, 16, "whole"))

            # C3 control (PLAN.md, rule fixed before this ran): disagreement adds
            # value only if C3 beats the single-T control on r=8 or whole-image
            # (CI excluding 0) without being significantly worse on the other.
            per_ctrl = evaluate_model("temp-mixed (C3 control, single T)", raw_cached, bands,
                                      control_temperature, thresholds["temp-mixed"])
            paired_bootstrap(per_c3, per_ctrl, "C3) - ECE(control  [does disagreement add value?]",
                             radii=(4, 8, 16, "whole"))

            # C5 (PLAN.md, rule fixed before this ran): (1) r=8 < control with CI
            # excluding 0; (2) whole-image not significantly worse than control;
            # (3) whole-image AUROC within CALIB_AUROC_DROP_LIMIT of raw (see the
            # AUROC line in the C5 block vs raw's).
            per_ctx = evaluate_model("context (C5)", raw_cached, bands, context_calibrator,
                                     thresholds["context"])
            paired_bootstrap(per_ctx, per_ctrl, "C5) - ECE(control  [rule 1: r=8 < 0; rule 2: whole-image not > 0]",
                             radii=(4, 8, 16, "whole"))
            paired_bootstrap(per_ctx, per_temp, f"C5) - ECE(temp T={temperature:.4f}  [context only]",
                             radii=(8, "whole"))
            print(f"C5 rule 3: AUROC drop limit vs raw = {config.CALIB_AUROC_DROP_LIMIT}")
            # P1 rule (e), MISTAKES.md M11: outlines compared at each row's own
            # val max-F1 operating point, paired over images.
            paired_bootstrap_bf1(per_ctx, per_raw, "C5) - boundary F1(raw  [P1 rule e]")

        # P1 rule (f): head 0 alone vs the 3-head fused score, same images.
        # Descriptive; the checkpoint epoch was selected on fused val AP.
        per_h0 = evaluate_model("head 0 only (head 0's sigmoid, no mean)", head0(raw_cached),
                                bands, 1.0, thresholds["head0"], keep_ap=True)
        print_ap_bootstrap(per_h0, per_raw, "head 0) - AP(fused 3 heads  [P1 rule f]")


if __name__ == "__main__":
    main()

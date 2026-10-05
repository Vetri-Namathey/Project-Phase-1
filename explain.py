"""Explainability for TwinGuard: WHY does it flag an object?  (evaluation only)

Implements the plan a teammate wrote on `main` (PLAN.md "Explainability",
steps X1-X4), built on this branch's model and tested here. Nothing trains.
Everything runs at the model's input resolution (1024x512), in fp32.

  X1  Counterfactual removal. Edit the image, re-run the model, measure what
      the score does. Per real object, one edit at a time:
        object     inpaint the object (+3px)               "is it the object?"
        ring       inpaint a band of +-4px around the edge, core kept
        interior   inpaint the core, ring kept
        context    inpaint an annulus 4-20px outside, object kept
        ring_blur  blur the ring (Gaussian), nothing removed
      Two fills (cv2 Telea inpainting; a same-row road patch copied from the
      same frame with a soft 2px skirt), because a repainted patch can itself
      look odd to the model. CONTROLS: the same edit applied to an
      object-shaped patch of clean road; a rise there is the fill's own
      artefact, not evidence. "Excess over artefact" subtracts it per object.
      Repeated on CutMix training composites ("paste") so real and pasted
      objects can be compared edit by edit.
  X2  Which encoder stage drives the score (gradient x activation on each of
      the 4 stage outputs, checked against a faithful stage ablation), real
      vs paste.
  X3  Epistemic / aleatoric split of the 3 heads (entropy of the mean =
      mean entropy + mutual information), by region, and whether either
      predicts where the model is wrong at object edges.
  X4  --demo writes the figures the /demo page shows (static/generated/explain/).

Neighbourhood = the object plus a ring of R_BAND px (4px here = the r=8 edge
band at the 2x-larger label resolution). All statistics are over objects with
a 95% CI from a bootstrap that resamples whole IMAGES (objects within one
image are not independent).

    python explain.py --checkpoint runs/phase2a_coco_run1/model_3head_best.pth
    python explain.py --checkpoint ... --demo        # figures only (needs a prior run)
"""

import argparse
import csv
import json
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import denormalize_imagenet, normalize_imagenet
from metrics import ScoreHistogram
from utils import get_device, load_trained_model

W, H = config.INPUT_WIDTH, config.INPUT_HEIGHT
R_BAND = 4          # px at model resolution; = r=8 at the 2x-larger label resolution
R_CONTEXT = 20      # outer radius of the context annulus
D_OBJECT = 3        # object removal also takes this many px beyond the mask
MIN_OBJ_PX = 12     # smaller components do not survive the downscale reliably
MIN_CORE_PX = 6     # "interior" needs at least this many core pixels
SIZE_MATCH = (100, 500)   # object area (px at model resolution) for the size-matched real-vs-paste comparisons
BLUR_SIGMA = 3.0
STAGE_NAMES = ["stage 1 (1/4, fine)", "stage 2 (1/8)", "stage 3 (1/16)", "stage 4 (1/32, coarse)"]
OUT_DIR = os.path.join("runs", "explain")
DEMO_DIR = os.path.join("static", "generated", "explain")

# (edit kind, region it edits, fill methods)
EDITS = [
    ("object", "object", ("telea", "patch")),
    ("ring", "ring", ("telea", "patch")),
    ("interior", "interior", ("telea", "patch")),
    ("context", "context", ("telea", "patch")),
    ("ring_blur", "ring", ("blur",)),
]
CONTROL_KINDS = {"object", "ring", "interior", "ring_blur"}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def objects_of(anomaly):
    """Connected components of the anomaly mask large enough to analyse."""
    labelled, n = ndimage.label(anomaly)
    if n == 0:
        return []
    sizes = ndimage.sum(anomaly, labelled, range(1, n + 1))
    return [labelled == k for k, s in zip(range(1, n + 1), sizes) if s >= MIN_OBJ_PX]


def regions_of(comp, others):
    """Edit regions and the scoring neighbourhood for one object.

    Never edits another object: `others` (dilated by 1px) is carved out."""
    d_out = ndimage.distance_transform_edt(~comp)
    d_in = ndimage.distance_transform_edt(comp)
    ring_in = comp & (d_in <= R_BAND)
    ring_out = (~comp) & (d_out <= R_BAND)
    protect = (ndimage.binary_dilation(others, iterations=1) if others.any()
               else np.zeros_like(comp))
    regions = {
        "object": d_out <= D_OBJECT,
        "ring": ring_in | ring_out,
        "interior": comp & ~ring_in,
        "context": (d_out > R_BAND) & (d_out <= R_CONTEXT),
    }
    regions = {k: v & ~protect for k, v in regions.items()}
    return regions, comp | ring_out


def bbox_mask(mask, pad):
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1] = True
    return out


def shift_mask(mask, dy, dx):
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[ys + dy, xs + dx] = True
    return out


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------

def fill_telea(rgb, region):
    if not region.any():
        return rgb
    return cv2.inpaint(np.ascontiguousarray(rgb), region.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def fill_blur(rgb, region):
    out = rgb.copy()
    out[region] = cv2.GaussianBlur(np.ascontiguousarray(rgb), (0, 0), BLUR_SIGMA)[region]
    return out


def find_shift(region, road, avoid):
    """(dy, dx) moving `region`'s bounding box onto road it does not overlap,
    preferring the same image row (same distance from the camera, so similar
    scale and lighting). None if the road has no free box that big."""
    ys, xs = np.nonzero(region)
    y0, x0 = ys.min(), xs.min()
    h, w = ys.max() - y0 + 1, xs.max() - x0 + 1
    if h > H or w > W:
        return None
    ok = road & ~avoid
    S = np.zeros((H + 1, W + 1), np.int32)
    S[1:, 1:] = ok.astype(np.int32).cumsum(0).cumsum(1)
    sums = S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]    # free pixels in each h x w box
    iy, ix = np.nonzero(sums == h * w)
    if len(iy) == 0:
        return None
    k = np.argmin(np.abs(iy - y0) * 1000 + np.abs(ix - x0))
    return int(iy[k] - y0), int(ix[k] - x0)


def fill_patch_hard(rgb, region, shift):
    """Plain copy of the shifted road pixels into `region` (leaves a visible seam)."""
    dy, dx = shift
    ys, xs = np.nonzero(region)
    out = rgb.copy()
    out[ys, xs] = rgb[ys + dy, xs + dx]
    return out


FEATHER_SIGMA = 2.0
FEATHER_PAD = 6


def fill_patch(rgb, region, shift):
    """Road patch copied from `shift` away, with a soft 2px skirt around the region.

    Inside `region` the pixels are exactly the shifted road pixels; outside it,
    a Gaussian skirt blends the patch in over ~3px so there is no hard seam
    (the training pastes were feathered the same way).

    Chosen by measurement, on 58 clean-road controls and 30 pasted objects:
        fill        false detections on clean road   pasted objects removed
        hard copy              55%                           47%
        Poisson blend          10%                            0%   <- leaves a flat grey
                                                                      ghost of the object
        Telea                  22%                           73%
        feathered (this)       22%                           80%
    The Poisson blend looked cleanest only because it removed nothing."""
    dy, dx = shift
    ys, xs = np.nonzero(region)
    y0, y1 = max(ys.min() - FEATHER_PAD, 0), min(ys.max() + FEATHER_PAD + 1, H)
    x0, x1 = max(xs.min() - FEATHER_PAD, 0), min(xs.max() + FEATHER_PAD + 1, W)
    alpha = np.maximum(region.astype(np.float32),
                       cv2.GaussianBlur(region.astype(np.float32), (0, 0), FEATHER_SIGMA))
    keep = np.zeros_like(alpha)
    keep[y0:y1, x0:x1] = 1.0                      # no influence beyond the padded box
    alpha = (alpha * keep)[..., None]
    sy = np.clip(np.arange(H) + dy, 0, H - 1)
    sx = np.clip(np.arange(W) + dx, 0, W - 1)
    shifted = rgb[sy[:, None], sx[None, :]]
    return np.rint(alpha * shifted + (1.0 - alpha) * rgb).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------

def to_input(rgb, device):
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return normalize_imagenet(t).to(device)


@torch.no_grad()
def run_model(model, rgb, device, want_seg=False):
    """-> (hidden_states, per-head probabilities (3,H,W) numpy, seg class map or None)."""
    x = to_input(rgb, device)
    hs = model.encode(x)
    size = x.shape[-2:]
    p = torch.sigmoid(torch.stack([head(hs, size) for head in model.ood_heads], dim=1))[0]
    seg = model.seg_head(hs, size).argmax(dim=1)[0].cpu().numpy() if want_seg else None
    return hs, p.cpu().numpy(), seg


def head_probs(model, hs, size):
    return torch.sigmoid(torch.stack([head(hs, size) for head in model.ood_heads], dim=1))[0]


def ood_target(model, leaves, size, region_t, head=None):
    """Sum over `region_t` of the fused (or one head's) probability: the
    scalar whose gradient is attributed to the encoder stages."""
    p = head_probs(model, leaves, size)
    p = p.mean(0) if head is None else p[head]
    return (p * region_t).sum()


def stage_attribution(model, leaves, size, region_t):
    """Share of |grad x activation| carried by each encoder stage.
    Returns (fused shares (4,), per-head shares (3,4), per-stage signed maps)."""
    def shares_for(head):
        target = ood_target(model, leaves, size, region_t, head)
        grads = torch.autograd.grad(target, leaves)
        attr = [g * a for g, a in zip(grads, leaves)]
        mag = torch.stack([a.abs().sum() for a in attr])
        return (mag / mag.sum().clamp(min=1e-30)).detach().cpu().numpy(), attr

    fused, attr = shares_for(None)
    per_head = np.stack([shares_for(h)[0] for h in range(len(model.ood_heads))])
    maps = [a.sum(dim=1)[0].detach().cpu().numpy() for a in attr]
    return fused, per_head, maps


@torch.no_grad()
def stage_ablation(model, hs, size, region_t):
    """Relative drop of the object's mean fused score when ONE stage's
    features at the object's location are replaced by the average of the
    features just around it (a 2-cell ring, in that stage's own grid).

    This is the faithful counterpart to the gradient shares: it removes the
    object's evidence from one stage only and re-runs the heads. (Replacing a
    whole stage by its global average was tried first and rejected: it feeds
    the heads out-of-distribution features everywhere, so scores rose 50-fold
    for reasons unrelated to the object.)"""
    denom = region_t.sum().clamp(min=1)
    base = (head_probs(model, hs, size).mean(0) * region_t).sum() / denom
    mask4 = region_t[None, None]
    rel, absolute = [], []
    for i, h in enumerate(hs):
        cell = F.interpolate(mask4, size=h.shape[-2:], mode="area")[0, 0] > 0
        ring = (F.max_pool2d(cell[None, None].float(), 5, 1, 2)[0, 0] > 0) & ~cell
        if not ring.any():
            rel.append(float("nan"))
            absolute.append(float("nan"))
            continue
        mod_h = h.clone()
        mod_h[:, :, cell] = h[:, :, ring].mean(dim=2, keepdim=True).expand(-1, -1, int(cell.sum()))
        mods = list(hs)
        mods[i] = mod_h
        s = (head_probs(model, mods, size).mean(0) * region_t).sum() / denom
        rel.append(float((base - s) / base.clamp(min=1e-9)))
        absolute.append(float(base - s))
    return np.array(rel), np.array(absolute)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def boot_idx(n_groups, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_groups, size=(n_boot, n_groups))


def cluster_mean(values, groups, n_boot=2000, seed=0):
    """(mean, lo, hi): mean of `values`, 95% CI from resampling whole groups."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=values), np.bincount(inv)
    idx = boot_idx(len(uniq), n_boot, seed)
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def cluster_diff(va, ga, vb, gb, n_boot=2000, seed=0):
    """(mean(a) - mean(b), lo, hi) for two independent samples of images."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    if len(va) == 0 or len(vb) == 0:
        return float("nan"), float("nan"), float("nan")
    ua, ia = np.unique(ga, return_inverse=True)
    ub, ib = np.unique(gb, return_inverse=True)
    sa, ca = np.bincount(ia, weights=va), np.bincount(ia)
    sb, cb = np.bincount(ib, weights=vb), np.bincount(ib)
    xa, xb = boot_idx(len(ua), n_boot, seed), boot_idx(len(ub), n_boot, seed + 1)
    boots = sa[xa].sum(1) / ca[xa].sum(1) - sb[xb].sum(1) / cb[xb].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(va.mean() - vb.mean()), float(lo), float(hi)


def fmt(ci, digits=3, pct=False):
    m, lo, hi = ci
    if not np.isfinite(m):
        return "n/a"          # e.g. the surroundings edit has no clean-road control
    if pct:
        return f"{100 * m:5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]"
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def json_safe(o):
    """Recursively make a result tree valid JSON: NaN/inf -> null, numpy -> Python.
    (Python's json module writes a bare NaN, which browsers and FastAPI reject.)"""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def bern_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def uncertainty_split(p):
    """p: (heads, ...) probabilities -> dict of maps, in bits.

    total = H(mean p);  aleatoric = mean_h H(p_h);  epistemic = total - aleatoric
    (mutual information, >= 0 by Jensen);  spread = std across heads."""
    total = bern_entropy(p.mean(0))
    aleatoric = bern_entropy(p).mean(0)
    return {"total": total, "aleatoric": aleatoric,
            "epistemic": np.maximum(total - aleatoric, 0.0), "spread": p.std(0)}


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def load_real(image_path, label_path):
    rgb = np.array(Image.open(image_path).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)
    label = np.array(Image.open(label_path).resize((W, H), Image.NEAREST))
    return rgb, label == 1, label != 255


def paste_samples(n):
    """CutMix composites of Cityscapes val frames + COCO objects: what the
    heads trained on (same objects bank, unseen backgrounds)."""
    from data.cityscapes_dataset import CityscapesDataset
    from data.cutmix import CutMixAugmentedDataset
    random.seed(config.GLOBAL_SEED)
    np.random.seed(config.GLOBAL_SEED)
    ds = CutMixAugmentedDataset(CityscapesDataset(split="val", normalize=False),
                                p=1.0, degradation_prob=0.0)
    out = []
    for i in np.random.default_rng(config.GLOBAL_SEED).choice(len(ds), n, replace=False):
        image_t, _, ood, _ = ds[int(i)]
        rgb = (denormalize_imagenet(image_t).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        out.append((f"paste{int(i):04d}", rgb, ood.numpy() > 0.5, np.ones((H, W), bool)))
    return out


def fit_threshold(model, val_pairs, device):
    """Max-F1 operating threshold on the Fishyscapes val half, at THIS script's
    resolution and precision (so it is self-consistent)."""
    hist = ScoreHistogram()
    for image_path, label_path in val_pairs:
        rgb, anomaly, valid = load_real(image_path, label_path)
        _, p, _ = run_model(model, rgb, device)
        hist.update(p.mean(0)[valid], anomaly[valid].astype(np.int64))
    return hist.threshold_at_max_f1()


# ---------------------------------------------------------------------------
# One image: X1 edits, X2 attribution
# ---------------------------------------------------------------------------

def stats_in(fused, region):
    v = fused[region]
    return float(v.max()), float(v.mean())


def analyse_image(model, device, source, name, rgb, anomaly, valid, tau, want_maps=False):
    """-> (records, stage_records, baseline) for every object in one image."""
    objs = objects_of(anomaly)
    if not objs:
        return [], [], None
    hs, p0, seg = run_model(model, rgb, device, want_seg=True)
    fused0 = p0.mean(0)
    size = (H, W)
    road = ndimage.binary_erosion(seg == 0, iterations=2)
    anom_dil = ndimage.binary_dilation(anomaly, iterations=16)

    # X2 setup: leaf copies of the four stage outputs, heads in eval mode.
    leaves = [h.detach().clone().requires_grad_(True) for h in hs]
    records, stage_records = [], []
    for oi, comp in enumerate(objs):
        others = anomaly & ~comp
        regions, nbh = regions_of(comp, others)
        peak0, mean0 = stats_in(fused0, nbh)
        head_mean0 = p0[:, nbh].mean(1)
        base = {"source": source, "image": name, "obj": oi, "obj_px": int(comp.sum()),
                "peak0": peak0, "mean0": mean0, "det0": peak0 >= tau}

        # ---- X2 ----
        region_t = torch.from_numpy(comp.astype(np.float32)).to(device)
        with torch.enable_grad():
            shares, per_head, maps = stage_attribution(model, leaves, size, region_t)
        ablation, ablation_abs = stage_ablation(model, hs, size, region_t)
        stage_records.append({**base, "share": shares, "per_head": per_head, "ablation": ablation,
                              "ablation_abs": ablation_abs, "maps": maps if want_maps else None})

        # ---- X1 ----
        for kind, region_key, methods in EDITS:
            F_ = regions[region_key]
            if F_.sum() == 0 or (kind == "interior" and F_.sum() < MIN_CORE_PX):
                continue
            avoid = anom_dil | bbox_mask(F_, 8)
            shift = find_shift(F_, road, avoid) if ("patch" in methods or kind in CONTROL_KINDS) else None
            ctrl_F = shift_mask(F_, *shift) if (shift and kind in CONTROL_KINDS) else None
            ctrl_E = ndimage.binary_dilation(ctrl_F, iterations=2) if ctrl_F is not None else None
            for method in methods:
                if method == "patch" and shift is None:
                    continue
                edit = (fill_telea(rgb, F_) if method == "telea" else
                        fill_blur(rgb, F_) if method == "blur" else fill_patch(rgb, F_, shift))
                _, p1, _ = run_model(model, edit, device)
                fused1 = p1.mean(0)
                peak1, mean1 = stats_in(fused1, nbh)
                rec = {**base, "kind": kind, "method": method, "peak1": peak1, "mean1": mean1,
                       "det1": peak1 >= tau, "fill_mean1": float(fused1[F_].mean()),
                       "head_mean0": head_mean0, "head_mean1": p1[:, nbh].mean(1),
                       "ctrl_peak0": None, "ctrl_peak1": None}
                if ctrl_F is not None:
                    c_edit = None
                    if method == "patch":
                        s2 = find_shift(ctrl_F, road, avoid | bbox_mask(ctrl_F, 8))
                        c_edit = fill_patch(rgb, ctrl_F, s2) if s2 else None
                    elif method == "telea":
                        c_edit = fill_telea(rgb, ctrl_F)
                    else:
                        c_edit = fill_blur(rgb, ctrl_F)
                    if c_edit is not None:
                        _, pc, _ = run_model(model, c_edit, device)
                        rec["ctrl_peak0"] = float(fused0[ctrl_E].max())
                        rec["ctrl_peak1"] = float(pc.mean(0)[ctrl_E].max())
                records.append(rec)
    return records, stage_records, {"fused0": fused0, "p0": p0, "seg": seg}


# ---------------------------------------------------------------------------
# X3
# ---------------------------------------------------------------------------

def uncertainty_regions(anomaly, valid):
    d_out = ndimage.distance_transform_edt(~anomaly)
    d_in = ndimage.distance_transform_edt(anomaly)
    return {"object core": anomaly & (d_in > R_BAND),
            "inner edge": anomaly & (d_in <= R_BAND),
            "outer edge": ~anomaly & valid & (d_out <= R_BAND),
            "background": ~anomaly & valid & (d_out > R_BAND)}


def x3_image(p0, anomaly, valid, tau, acc):
    unc = uncertainty_split(p0)
    fused = p0.mean(0)
    regions = uncertainty_regions(anomaly, valid)
    for rname, m in regions.items():
        if m.any():
            for uname, arr in unc.items():
                acc["region"].setdefault((rname, uname), []).append((float(arr[m].sum()), int(m.sum())))
            acc["region"].setdefault((rname, "score"), []).append((float(fused[m].sum()), int(m.sum())))
    band = regions["inner edge"] | regions["outer edge"]
    pred = fused >= tau
    outcome = np.where(anomaly, np.where(pred, "TP", "FN"), np.where(pred, "FP", "TN"))
    for oname in ("TP", "FN", "FP", "TN"):
        m = band & (outcome == oname)
        if m.any():
            for uname, arr in unc.items():
                acc["outcome"].setdefault((oname, uname), []).append((float(arr[m].sum()), int(m.sum())))
    # error prediction inside the band
    err = (pred != anomaly)[band].astype(np.int64)
    conf = (1 - np.abs(fused - tau) / max(tau, 1 - tau))[band]
    for uname, scale in (("total", 1.0), ("aleatoric", 1.0), ("epistemic", 1.0), ("spread", 2.0)):
        acc["err_hist"].setdefault(uname, ScoreHistogram()).update(np.clip(unc[uname][band] * scale, 0, 1), err)
    acc["err_hist"].setdefault("near-threshold", ScoreHistogram()).update(np.clip(conf, 0, 1), err)
    # direction of the error, among errors only: missed object pixel (1) vs false alarm (0)
    wrong = err.astype(bool)
    if wrong.any():
        is_fn = (outcome[band][wrong] == "FN").astype(np.int64)
        for uname, scale in (("total", 1.0), ("aleatoric", 1.0), ("epistemic", 1.0), ("spread", 2.0)):
            acc["dir_hist"].setdefault(uname, ScoreHistogram()).update(
                np.clip(unc[uname][band][wrong] * scale, 0, 1), is_fn)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def group(records, **match):
    return [r for r in records if all(r.get(k) == v for k, v in match.items())]


def report_x1(records, summary):
    print("\n" + "=" * 100)
    print("X1  COUNTERFACTUAL REMOVAL  (objects the model detects; neighbourhood = object + 4px ring)")
    print("=" * 100)
    for source in ("real", "paste"):
        print(f"\n--- {source} objects ---")
        print(f"{'edit':<10}{'fill':<7}{'n':>4}  {'peak before->after':>20}  {'peak drop [95% CI]':>28}"
              f"  {'detection lost [95% CI]':>26}  {'excess over fill artefact':>28}")
        for kind, _, methods in EDITS:
            for method in methods:
                rs = [r for r in group(records, source=source, kind=kind, method=method) if r["det0"]]
                if not rs:
                    continue
                img = [r["image"] for r in rs]
                drop = [r["peak0"] - r["peak1"] for r in rs]
                lost = [float(not r["det1"]) for r in rs]
                pk0, pk1 = np.mean([r["peak0"] for r in rs]), np.mean([r["peak1"] for r in rs])
                ci_d, ci_l = cluster_mean(drop, img), cluster_mean(lost, img)
                # post-edit peak minus the peak the SAME edit gives on clean road: what is left
                # of the object's evidence once the fill's own damage is subtracted
                wc = [r for r in rs if r["ctrl_peak1"] is not None]
                ci_x = cluster_mean([r["peak1"] - r["ctrl_peak1"] for r in wc], [r["image"] for r in wc]) \
                    if wc else (float("nan"),) * 3
                print(f"{kind:<10}{method:<7}{len(rs):>4}  {pk0:>9.3f} -> {pk1:<8.3f}  {fmt(ci_d):>28}  "
                      f"{fmt(ci_l, pct=True):>26}  {fmt(ci_x):>28}")
                summary["x1"].setdefault(source, []).append({
                    "kind": kind, "method": method, "n": len(rs), "peak0": float(pk0), "peak1": float(pk1),
                    "drop": ci_d, "lost": ci_l, "excess": ci_x,
                    "mean_drop": cluster_mean([r["mean0"] - r["mean1"] for r in rs], img)})

    print("\n--- fill artefact controls: the SAME edit on an object-shaped patch of clean road ---")
    print(f"{'edit':<10}{'fill':<7}{'n':>4}  {'peak before->after':>20}  {'rise [95% CI]':>28}  {'made a detection':>18}")
    for kind, _, methods in EDITS:
        if kind not in CONTROL_KINDS:
            continue
        for method in methods:
            rs = [r for r in records if r.get("kind") == kind and r["method"] == method and r["ctrl_peak1"] is not None]
            if not rs:
                continue
            ci = cluster_mean([r["ctrl_peak1"] - r["ctrl_peak0"] for r in rs], [r["image"] for r in rs])
            made = np.mean([float(r["ctrl_peak1"] >= summary["tau"] > r["ctrl_peak0"]) for r in rs])
            print(f"{kind:<10}{method:<7}{len(rs):>4}  {np.mean([r['ctrl_peak0'] for r in rs]):>9.3f} -> "
                  f"{np.mean([r['ctrl_peak1'] for r in rs]):<8.3f}  {fmt(ci):>28}  {100 * made:>16.1f}%")
            summary["x1"].setdefault("control", []).append(
                {"kind": kind, "method": method, "n": len(rs), "rise": ci, "made_detection": float(made)})

    print("\n--- contrasts (peak drop, paired per object; Telea fill) ---")
    for source in ("real", "paste"):
        by = {}
        for r in group(records, source=source, method="telea"):
            if r["det0"]:
                by.setdefault((r["image"], r["obj"]), {})[r["kind"]] = r["peak0"] - r["peak1"]
        for a, b in (("object", "ring"), ("ring", "interior"), ("object", "interior")):
            pairs = [(k[0], d[a] - d[b]) for k, d in by.items() if a in d and b in d]
            if pairs:
                ci = cluster_mean([v for _, v in pairs], [g for g, _ in pairs])
                print(f"  {source:<6} drop({a}) - drop({b}) = {fmt(ci)}   n={len(pairs)}")
                summary["x1"].setdefault("contrast", []).append({"source": source, "a": a, "b": b, "n": len(pairs), "ci": ci})
    print("\n--- real vs paste (peak drop, Telea fill; difference of means over independent image sets) ---")
    for kind, _, methods in EDITS:
        method = methods[0]
        ra = [r for r in group(records, source="real", kind=kind, method=method) if r["det0"]]
        rb = [r for r in group(records, source="paste", kind=kind, method=method) if r["det0"]]
        if ra and rb:
            ci = cluster_diff([r["peak0"] - r["peak1"] for r in ra], [r["image"] for r in ra],
                              [r["peak0"] - r["peak1"] for r in rb], [r["image"] for r in rb])
            print(f"  {kind:<10}{method:<7} real - paste = {fmt(ci)}   (n {len(ra)} real, {len(rb)} paste)")
            summary["x1"].setdefault("real_vs_paste", []).append({"kind": kind, "method": method, "ci": ci})

    lo, hi = SIZE_MATCH
    print(f"\n--- real vs paste again, SIZE-MATCHED to objects of {lo}-{hi} px (pasted objects are larger, which confounds the above) ---")
    for kind, _, methods in EDITS:
        for method in methods:
            ra = [r for r in group(records, source="real", kind=kind, method=method) if r["det0"] and lo <= r["obj_px"] <= hi]
            rb = [r for r in group(records, source="paste", kind=kind, method=method) if r["det0"] and lo <= r["obj_px"] <= hi]
            if len(ra) >= 5 and len(rb) >= 5:
                ci = cluster_diff([r["peak0"] - r["peak1"] for r in ra], [r["image"] for r in ra],
                                  [r["peak0"] - r["peak1"] for r in rb], [r["image"] for r in rb])
                print(f"  {kind:<10}{method:<7} real - paste = {fmt(ci)}   (n {len(ra)} real, {len(rb)} paste)")
                summary["x1"].setdefault("real_vs_paste_size_matched", []).append(
                    {"kind": kind, "method": method, "n_real": len(ra), "n_paste": len(rb), "ci": ci})


def report_x2(stage_records, summary):
    print("\n" + "=" * 100)
    print("X2  WHICH ENCODER STAGE DRIVES THE SCORE  (share of |gradient x activation|; all detected objects)")
    print("=" * 100)
    print(f"{'':<8}{'n':>4}  " + "".join(f"{s:>26}" for s in STAGE_NAMES))
    for source in ("real", "paste"):
        rs = [r for r in stage_records if r["source"] == source and r["det0"]]
        if not rs:
            continue
        img = [r["image"] for r in rs]
        cis = [cluster_mean([r["share"][i] for r in rs], img) for i in range(4)]
        print(f"{source:<8}{len(rs):>4}  " + "".join(f"{100 * m:>9.1f}% [{100 * lo:4.1f},{100 * hi:4.1f}]  " for m, lo, hi in cis))
        abl = [cluster_mean([r["ablation"][i] for r in rs], img) for i in range(4)]
        abl_abs = [cluster_mean([r["ablation_abs"][i] for r in rs], img) for i in range(4)]
        summary["x2"][source] = {"n": len(rs), "share": cis, "ablation": abl, "ablation_abs": abl_abs,
                                 "per_head": np.mean([r["per_head"] for r in rs], axis=0).tolist()}
    print("\nfaithful check: relative drop of the object's score when ONE stage's features at the object are")
    print("replaced by the average of their surroundings (positive = that stage carries the object's evidence):")
    for source in ("real", "paste"):
        if source in summary["x2"]:
            s = summary["x2"][source]
            print(f"{source:<8}" + "".join(f"{100 * m:>9.1f}% [{100 * lo:5.1f},{100 * hi:5.1f}] " for m, lo, hi in s["ablation"])
                  + "   (relative)")
            print(f"{'':<8}" + "".join(f"{m:>+9.3f}  [{lo:+.3f},{hi:+.3f}]" for m, lo, hi in s["ablation_abs"])
                  + "   (score points)")
    ra = [r for r in stage_records if r["source"] == "real" and r["det0"]]
    rb = [r for r in stage_records if r["source"] == "paste" and r["det0"]]
    if ra and rb:
        print("\nreal - paste share difference per stage [95% CI]:")
        diffs = [cluster_diff([r["share"][i] for r in ra], [r["image"] for r in ra],
                              [r["share"][i] for r in rb], [r["image"] for r in rb]) for i in range(4)]
        for name, ci in zip(STAGE_NAMES, diffs):
            print(f"  {name:<26}{100 * ci[0]:+6.1f} pts [{100 * ci[1]:+5.1f}, {100 * ci[2]:+5.1f}]")
        summary["x2"]["real_minus_paste"] = diffs
        lo, hi = SIZE_MATCH
        sa = [r for r in ra if lo <= r["obj_px"] <= hi]
        sb = [r for r in rb if lo <= r["obj_px"] <= hi]
        if len(sa) >= 5 and len(sb) >= 5:
            print(f"\nsize-matched ({lo}-{hi} px; n {len(sa)} real, {len(sb)} paste): real - paste share, and relative ablation drop:")
            sm_share = [cluster_diff([r["share"][i] for r in sa], [r["image"] for r in sa],
                                     [r["share"][i] for r in sb], [r["image"] for r in sb]) for i in range(4)]
            sm_abl = [cluster_diff([r["ablation"][i] for r in sa], [r["image"] for r in sa],
                                   [r["ablation"][i] for r in sb], [r["image"] for r in sb]) for i in range(4)]
            for name, c1, c2 in zip(STAGE_NAMES, sm_share, sm_abl):
                print(f"  {name:<26}share {100 * c1[0]:+6.1f} pts [{100 * c1[1]:+5.1f}, {100 * c1[2]:+5.1f}]"
                      f"   ablation {100 * c2[0]:+6.1f} pts [{100 * c2[1]:+5.1f}, {100 * c2[2]:+5.1f}]")
            summary["x2"]["size_matched"] = {"n_real": len(sa), "n_paste": len(sb), "share": sm_share, "ablation": sm_abl}


def report_x3(acc, n_images, summary):
    print("\n" + "=" * 100)
    print("X3  EPISTEMIC / ALEATORIC SPLIT OF THE 3 HEADS  (real Fishyscapes test objects, bits)")
    print("=" * 100)
    names = ["object core", "inner edge", "outer edge", "background"]
    meas = ["score", "total", "aleatoric", "epistemic", "spread"]
    print(f"{'region':<14}" + "".join(f"{m:>12}" for m in meas) + f"{'epistemic share':>18}")
    summary["x3"]["regions"] = {}
    for r in names:
        vals = {m: sum(s for s, _ in acc["region"][(r, m)]) / sum(c for _, c in acc["region"][(r, m)])
                for m in meas if (r, m) in acc["region"]}
        if not vals:
            continue
        share = vals["epistemic"] / vals["total"] if vals["total"] > 1e-12 else float("nan")
        print(f"{r:<14}" + "".join(f"{vals[m]:>12.4f}" for m in meas) + f"{share:>17.1%}")
        summary["x3"]["regions"][r] = {**vals, "epistemic_share": share}
    print("\nmean uncertainty by outcome inside the +-4px edge band (threshold = val-fitted):")
    print(f"{'outcome':<8}" + "".join(f"{m:>12}" for m in meas[1:]) + f"{'pixels':>12}")
    summary["x3"]["outcomes"] = {}
    for o in ("TP", "FN", "FP", "TN"):
        if (o, "total") in acc["outcome"]:
            n = sum(c for _, c in acc["outcome"][(o, "total")])
            v = {m: sum(s for s, _ in acc["outcome"][(o, m)]) / n for m in meas[1:]}
            print(f"{o:<8}" + "".join(f"{v[m]:>12.4f}" for m in meas[1:]) + f"{n:>12,}")
            summary["x3"]["outcomes"][o] = {**v, "n": n}
    print("\nAUROC of each signal for 'is this edge pixel wrong?' (band pixels, error = predicted != true):")
    summary["x3"]["error_auroc"] = {}
    for u, h in acc["err_hist"].items():
        print(f"  {u:<16}{h.auroc():.4f}")
        summary["x3"]["error_auroc"][u] = h.auroc()
    print("AUROC for telling the DIRECTION of an error (missed object pixel vs false alarm), errors only:")
    summary["x3"]["direction_auroc"] = {}
    for u, h in acc["dir_hist"].items():
        print(f"  {u:<16}{h.auroc():.4f}   (0.5 = no information about direction; >0.5 = higher means missed)")
        summary["x3"]["direction_auroc"][u] = h.auroc()
    print("\nCaveat: the 3 heads share one frozen encoder, so their disagreement under-states true epistemic uncertainty.")


# ---------------------------------------------------------------------------
# Demo figures (X4)
# ---------------------------------------------------------------------------

def inferno(arr):
    return (matplotlib.colormaps["inferno"](np.clip(arr, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def crop_window(comp):
    ys, xs = np.nonzero(comp)
    hh = int(max(80, 4 * max(ys.max() - ys.min() + 1, (xs.max() - xs.min() + 1) / 2)))
    ww = 2 * hh
    if ww > W or hh > H:
        ww, hh = W, H
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - hh // 2, 0, H - hh))
    x0 = int(np.clip(cx - ww // 2, 0, W - ww))
    return y0, y0 + hh, x0, x0 + ww


def save_crop(arr, win, path, width=420, nearest=False):
    y0, y1, x0, x1 = win
    img = Image.fromarray(arr[y0:y1, x0:x1])
    img.resize((width, int(width * (y1 - y0) / (x1 - x0))), Image.NEAREST if nearest else Image.BICUBIC).save(path)


def build_demo(model, device, tau, summary_path):
    manifest_path = os.path.join("static", "generated", "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("run `python server.py` once first: the demo frames come from its manifest")
    with open(manifest_path) as f:
        manifest = json.load(f)
    pairs = {os.path.basename(a): (a, b) for a, b in list_fishyscapes_pairs()}
    os.makedirs(DEMO_DIR, exist_ok=True)
    frames = []
    for fr in manifest["frames"]:
        image_path, label_path = pairs[fr["title"]]
        rgb, anomaly, valid = load_real(image_path, label_path)
        objs = objects_of(anomaly)
        if not objs:
            continue
        hs, p0, seg = run_model(model, rgb, device, want_seg=True)
        fused0 = p0.mean(0)
        # the largest object the model detects; else the largest
        scored = [(float(fused0[ndimage.binary_dilation(c, iterations=R_BAND)].max()), int(c.sum()), i)
                  for i, c in enumerate(objs)]
        detected = [s for s in scored if s[0] >= tau] or scored
        comp = objs[max(detected, key=lambda s: s[1])[2]]
        others = anomaly & ~comp
        regions, nbh = regions_of(comp, others)
        road = ndimage.binary_erosion(seg == 0, iterations=2)
        anom_dil = ndimage.binary_dilation(anomaly, iterations=16)
        win = crop_window(comp)
        out_dir = os.path.join(DEMO_DIR, fr["id"])
        os.makedirs(out_dir, exist_ok=True)

        base_img = rgb.copy()
        base_img[outline(comp)] = (0, 200, 83)
        save_crop(base_img, win, os.path.join(out_dir, "original.png"))
        save_crop(inferno(fused0), win, os.path.join(out_dir, "heat_original.png"))
        peak0, mean0 = stats_in(fused0, nbh)

        edits = []
        for kind, label in (("object", "object removed"), ("ring", "edge band removed, core kept"),
                            ("interior", "core removed, edge band kept")):
            F_ = regions[kind]
            if F_.sum() == 0 or (kind == "interior" and F_.sum() < MIN_CORE_PX):
                continue
            edit = fill_telea(rgb, F_)
            _, p1, _ = run_model(model, edit, device)
            fused1 = p1.mean(0)
            peak1, mean1 = stats_in(fused1, nbh)
            shown = edit.copy()
            shown[outline(F_)] = (0, 229, 255)
            save_crop(shown, win, os.path.join(out_dir, f"{kind}.png"))
            save_crop(inferno(fused1), win, os.path.join(out_dir, f"heat_{kind}.png"))
            edits.append({"kind": kind, "label": label, "peak0": peak0, "peak1": peak1,
                          "mean0": mean0, "mean1": mean1, "detected_after": peak1 >= tau,
                          "image": f"/static/generated/explain/{fr['id']}/{kind}.png",
                          "heat": f"/static/generated/explain/{fr['id']}/heat_{kind}.png"})

        leaves = [h.detach().clone().requires_grad_(True) for h in hs]
        region_t = torch.from_numpy(comp.astype(np.float32)).to(device)
        with torch.enable_grad():
            shares, per_head, maps = stage_attribution(model, leaves, (H, W), region_t)
        ablation, _ = stage_ablation(model, hs, (H, W), region_t)
        stage_urls = []
        for i, m in enumerate(maps):
            up = F.interpolate(torch.from_numpy(np.maximum(m, 0))[None, None], size=(H, W),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            up = up / max(up.max(), 1e-12)
            save_crop(inferno(up ** 0.5), win, os.path.join(out_dir, f"stage{i + 1}.png"), width=300)
            stage_urls.append(f"/static/generated/explain/{fr['id']}/stage{i + 1}.png")

        frames.append({
            "id": fr["id"], "title": fr["title"], "selection": fr.get("selection"),
            "object_px": int(comp.sum()), "detected": peak0 >= tau, "peak0": peak0, "mean0": mean0,
            "original": f"/static/generated/explain/{fr['id']}/original.png",
            "heat_original": f"/static/generated/explain/{fr['id']}/heat_original.png",
            "edits": edits, "stage_share": shares.tolist(), "stage_ablation": ablation.tolist(),
            "stage_maps": stage_urls})
        print(f"  {fr['id']}: object {int(comp.sum())} px, peak {peak0:.3f}, "
              + ", ".join(f"{e['kind']}->{e['peak1']:.3f}" for e in edits)
              + f", stage shares {np.round(shares, 2).tolist()}")
    aggregate = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            aggregate = json.load(f)
    with open(os.path.join(DEMO_DIR, "explain.json"), "w") as f:
        json.dump(json_safe({"status": "ready", "threshold": tau, "stage_names": STAGE_NAMES, "r_band_px": R_BAND,
                             "frames": frames, "aggregate": aggregate}), f, indent=2)
    print(f"wrote {DEMO_DIR}/explain.json  ({len(frames)} frames)")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_3HEAD)
    parser.add_argument("--pastes", type=int, default=30, help="CutMix composites for the real-vs-paste comparison")
    parser.add_argument("--limit", type=int, default=None, help="only the first N real test images (smoke test)")
    parser.add_argument("--demo", action="store_true", help="write the /demo figures instead of the full run")
    args = parser.parse_args()

    device = get_device()
    model = load_trained_model(args.checkpoint, device)
    val_pairs, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "summary.json")

    t0 = time.time()
    tau = fit_threshold(model, val_pairs, device)
    print(f"operating threshold (max-F1 on the val half, model resolution, fp32) = {tau:.4f}  [{time.time() - t0:.0f}s]")
    if args.demo:
        build_demo(model, device, tau, summary_path)
        return

    sources = [("real", os.path.basename(a), *load_real(a, b)) for a, b in test_pairs[:args.limit]]
    if args.pastes:
        sources += [("paste", name, rgb, anom, valid) for name, rgb, anom, valid in paste_samples(args.pastes)]

    records, stage_records = [], []
    x3_acc = {"region": {}, "outcome": {}, "err_hist": {}, "dir_hist": {}}
    n_real = 0
    for i, (source, name, rgb, anomaly, valid) in enumerate(sources):
        recs, srecs, base = analyse_image(model, device, source, name, rgb, anomaly, valid, tau)
        records += recs
        stage_records += srecs
        if source == "real" and base is not None:
            x3_image(base["p0"], anomaly, valid, tau, x3_acc)
            n_real += 1
        if (i + 1) % 5 == 0 or i == len(sources) - 1:
            el = time.time() - t0
            print(f"  {i + 1}/{len(sources)} images, {len(srecs)} objs this image, {len(records)} edit records, "
                  f"{el:.0f}s (eta {el / (i + 1) * (len(sources) - i - 1):.0f}s)", flush=True)

    summary = {"checkpoint": args.checkpoint, "tau": tau, "r_band_px": R_BAND,
               "n_real_images": n_real, "n_real_objects": len({(r['image'], r['obj']) for r in stage_records if r['source'] == 'real'}),
               "n_paste_objects": len({(r['image'], r['obj']) for r in stage_records if r['source'] == 'paste'}),
               "x1": {}, "x2": {}, "x3": {}}
    print(f"\n{summary['n_real_objects']} real objects in {n_real} test images; "
          f"{summary['n_paste_objects']} pasted objects; threshold {tau:.4f}; "
          f"{sum(r['det0'] for r in stage_records if r['source'] == 'real')} real objects detected")
    for source in ("real", "paste"):
        sizes = [r["obj_px"] for r in stage_records if r["source"] == source]
        if sizes:
            print(f"  {source}: object area median {int(np.median(sizes))} px "
                  f"(quartiles {int(np.percentile(sizes, 25))}-{int(np.percentile(sizes, 75))}) at {W}x{H}")
            summary[f"{source}_median_px"] = float(np.median(sizes))
    report_x1(records, summary)
    report_x2(stage_records, summary)
    report_x3(x3_acc, n_real, summary)

    with open(summary_path, "w") as f:
        json.dump(json_safe(summary), f, indent=1)
    with open(os.path.join(OUT_DIR, "x1_objects.csv"), "w", newline="") as f:
        cols = ["source", "image", "obj", "obj_px", "kind", "method", "peak0", "peak1", "mean0", "mean1",
                "det0", "det1", "fill_mean1", "ctrl_peak0", "ctrl_peak1"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nsaved {summary_path} and x1_objects.csv   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

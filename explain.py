"""Explainability, eval-only (PLAN.md "Explainability (2026-10-03)").

X1 -- counterfactual removal. Remove part of the evidence, re-run the model,
measure how far the anomaly score falls. Faithful by construction: it is what
the model actually does, not a gradient approximation, and it works at the
size of Fishyscapes objects (~22 px at model resolution), which CAM cannot.

Four edits per Fishyscapes test image, each measured on a fixed region:
  object    fill the whole object (dilated 2 px)    -> score on the object
  edges     fill the ring around the edge (both     -> score on the interior
            sides, +-RING px)                          (does the interior need its edges?)
  interior  fill the object minus its inner ring    -> score on the inner ring
                                                       (do edges alone carry the score?)
  context   blur everything >CONTEXT_PX away        -> score on the object
                                                       (is the score local?)

Two fill methods, because an inpainted patch can itself look anomalous and
understate the drop (PLAN.md main risk): cv2 Telea inpainting, and a road patch
copied from elsewhere in the same frame (road = the model's own segmentation).
Reported side by side; the filled region's own score is printed as a check.

All work happens at model input resolution (1024x512): the image is resized
exactly like load_image_tensor does, the label with NEAREST. RING=4 px here is
r=8 at Fishyscapes label resolution, the radius used throughout C1-C5.

Usage (CARLA closed, GPU):
    python explain.py | Tee-Object -FilePath explain_x1.log
Writes explain_out/x1_per_image.json and a few example strips to explain_out/.
"""

import argparse
import json
import os

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import normalize_imagenet
from metrics import boundary_band
from train import amp_context
from utils import get_device, load_trained_model

RING = 4
CONTEXT_PX = 32
MIN_OBJECT_PX = 30
MIN_REGION_PX = 10
ROAD_TRAINID = 0
N_BOOTSTRAP = 1000
CONDITIONS = ("object", "edges", "interior", "context")
# Raw-model val max-F1 threshold (eval_fishyscapes_T122.log; = server.BOX_THRESHOLD).
DETECT_THRESHOLD = 0.47658


@torch.no_grad()
def run(model, image, device):
    """uint8 (H,W,3) at model resolution -> fused (H,W), heads (3,H,W), seg argmax (H,W)."""
    t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = normalize_imagenet(t).to(device)
    with amp_context(device):
        out = model(t)
    size = image.shape[:2]
    heads = torch.sigmoid(F.interpolate(out["ood_logits"].float(), size=size, mode="bilinear",
                                        align_corners=False))[0]
    seg = F.interpolate(out["seg_logits"].float(), size=size, mode="bilinear",
                        align_corners=False)[0].argmax(0)
    return heads.mean(0).cpu().numpy(), heads.cpu().numpy(), seg.cpu().numpy()


def load_pair(image_path, label_path):
    image = np.asarray(Image.open(image_path).convert("RGB").resize(
        (config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR), dtype=np.uint8).copy()
    label = np.asarray(Image.open(label_path).resize(
        (config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.NEAREST))
    return image, label == 1, label != 255


def fill_telea(image, region):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, region.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def find_road_shift(region, road, keep_out, step=8, max_dx=320, max_dy=96):
    """Smallest (dy, dx) that moves `region` entirely onto predicted road and
    clear of the object. None if the frame has no such patch."""
    ys, xs = np.nonzero(region)
    H, W = region.shape
    offsets = sorted(((dy, dx) for dy in range(-max_dy, max_dy + 1, step)
                      for dx in range(-max_dx, max_dx + 1, step) if (dy, dx) != (0, 0)),
                     key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dy, dx in offsets:
        sy, sx = ys + dy, xs + dx
        if sy.min() < 0 or sx.min() < 0 or sy.max() >= H or sx.max() >= W:
            continue
        if keep_out[sy, sx].any():
            continue
        if road[sy, sx].mean() >= 0.95:
            return dy, dx
    return None


def fill_patch(image, region, shift):
    out = image.copy()
    ys, xs = np.nonzero(region)
    out[ys, xs] = image[ys + shift[0], xs + shift[1]]
    return out


def blur_context(image, keep):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=15)
    out = image.copy()
    out[~keep] = blurred[~keep]
    return out


def bootstrap_ci(values, rng):
    values = np.asarray(values)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(N_BOOTSTRAP)]
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def heat_strip(images_scores, path):
    tiles = []
    for image, score in images_scores:
        heat = (cm.inferno(np.clip(score, 0, 1))[..., :3] * 255).astype(np.uint8)
        tiles.append(np.concatenate([image, (0.4 * image + 0.6 * heat).astype(np.uint8)], axis=0))
    Image.fromarray(np.concatenate(tiles, axis=1)).resize(
        (len(tiles) * 512, 512), Image.BILINEAR).save(path)


def x1b(model, device, clean_root, every):
    """X1b -- exact counterfactual on the CARLA video route. <root>_pasted is
    <root> plus pastes, frame for frame, so the clean frame IS the object
    removed, with no inpainting. Also inpaints the same region of the pasted
    frame, to measure how far Telea distorts the 'after' score (X1's confound:
    run 1 showed every edit gets flagged)."""
    pasted_root = clean_root.rstrip("/\\") + "_pasted"
    with open(os.path.join(pasted_root, "frame_pastes.jsonl")) as f:
        frames = [json.loads(line) for line in f]
    frames = [fr for fr in frames if fr["pastes"]][::every]
    print(f"X1b exact counterfactual: {clean_root} vs {pasted_root}, {len(frames)} frames (every {every}th with a paste)")

    rows = []
    for fr in frames:
        name = f"{fr['frame']:06d}"
        clean = np.array(Image.open(os.path.join(clean_root, "rgb", name + ".png")).convert("RGB"))
        pasted = np.array(Image.open(os.path.join(pasted_root, "rgb", name + ".png")).convert("RGB"))
        native = np.load(os.path.join(clean_root, "mask", name + ".npy")).astype(bool)
        paste = np.load(os.path.join(pasted_root, "mask", name + ".npy")).astype(bool) & ~native
        if paste.sum() < MIN_REGION_PX:
            continue
        far_bg = ndimage.distance_transform_edt(~(paste | native)) > CONTEXT_PX
        s_pasted, _, _ = run(model, pasted, device)
        s_clean, _, _ = run(model, clean, device)
        s_telea, _, _ = run(model, fill_telea(pasted, ndimage.binary_dilation(paste, iterations=2)), device)
        rows.append({"frame": fr["frame"], "paste_px": int(paste.sum()),
                     "pasted": float(s_pasted[paste].mean()), "clean": float(s_clean[paste].mean()),
                     "telea": float(s_telea[paste].mean()), "bg": float(s_clean[far_bg].mean())})

    rng = np.random.default_rng(config.GLOBAL_SEED)
    get = lambda k: np.array([r[k] for r in rows])
    true_drop = bootstrap_ci(get("pasted") - get("clean"), rng)
    telea_drop = bootstrap_ci(get("pasted") - get("telea"), rng)
    bias = bootstrap_ci(get("telea") - get("clean"), rng)
    print(f"\n{len(rows)} frames used. Mean fused score on the pasted pixels:")
    print(f"  with paste {get('pasted').mean():.3f}   true removal (clean frame) {get('clean').mean():.3f}   "
          f"Telea removal {get('telea').mean():.3f}   (far background {get('bg').mean():.4f})")
    print(f"  true drop   {true_drop[0]:+.3f} [{true_drop[1]:+.3f}, {true_drop[2]:+.3f}]")
    print(f"  Telea drop  {telea_drop[0]:+.3f} [{telea_drop[1]:+.3f}, {telea_drop[2]:+.3f}]")
    print(f"  Telea bias (Telea after - true after) {bias[0]:+.3f} [{bias[1]:+.3f}, {bias[2]:+.3f}]  "
          f"-> how much an inpainted fill itself scores as anomalous")
    return rows


def samples(clean_root, every):
    """(dataset, name, image uint8 at 1024x512, anomaly mask, valid) for the
    Fishyscapes test half (object >= MIN_OBJECT_PX) and, if given, every Nth
    pasted frame of the CARLA route (anomaly = native props + pastes)."""
    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    for image_path, label_path in test_pairs:
        image, obj, valid = load_pair(image_path, label_path)
        if obj.sum() >= MIN_OBJECT_PX:
            yield "fishyscapes", os.path.basename(image_path), image, obj, valid
    if not clean_root:
        return
    pasted_root = clean_root.rstrip("/\\") + "_pasted"
    with open(os.path.join(pasted_root, "frame_pastes.jsonl")) as f:
        frames = [json.loads(line) for line in f if json.loads(line)["pastes"]][::every]
    for fr in frames:
        name = f"{fr['frame']:06d}"
        image = np.array(Image.open(os.path.join(pasted_root, "rgb", name + ".png")).convert("RGB"))
        obj = np.load(os.path.join(pasted_root, "mask", name + ".npy")).astype(bool)
        if obj.sum() >= MIN_OBJECT_PX:
            yield "carla_pasted", name, image, obj, np.ones_like(obj)


def stage_attribution(model, image, region, device):
    """X2: grad x activation of each head's mean logit over `region`, summed
    per encoder stage. The encoder runs under no_grad in the model, so this
    makes its 4 hidden states leaf tensors and differentiates only the heads
    -- the training forward is untouched. -> (n_heads, 4) signed contributions."""
    t = normalize_imagenet(torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        hidden = model.encoder(pixel_values=t, output_hidden_states=True).hidden_states
    hidden = [h.detach().float().requires_grad_(True) for h in hidden]
    region_t = torch.from_numpy(region).to(device)
    out = []
    for head in model.ood_heads:
        target = head(hidden, image.shape[:2])[0][region_t].mean()
        grads = torch.autograd.grad(target, hidden)
        out.append([float((g * h).sum()) for g, h in zip(grads, hidden)])
    return np.array(out)


@torch.no_grad()
def stage_ablation(model, image, region, device):
    """X2 faithfulness (PLAN.md P4), ported from exp_v2 explain.stage_ablation:
    replace ONE stage's features at the object's cells by the mean of a 2-cell
    ring around them (that stage's own grid), re-run the heads, and measure the
    drop of the object's mean fused score. Local on purpose: exp_v2 found that
    a global-average ablation feeds the heads impossible features everywhere.
    -> (relative drop (4,), absolute drop (4,)); NaN where a stage has no ring."""
    t = normalize_imagenet(torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(device)
    hidden = [h.float() for h in model.encoder(pixel_values=t, output_hidden_states=True).hidden_states]
    size = image.shape[:2]
    region_t = torch.from_numpy(region).to(device).float()
    denom = region_t.sum().clamp(min=1)

    def score(hs):
        p = torch.sigmoid(torch.stack([head(hs, size) for head in model.ood_heads], dim=1))[0].mean(0)
        return (p * region_t).sum() / denom

    base = score(hidden)
    rel, absolute = [], []
    for i, h in enumerate(hidden):
        cell = F.interpolate(region_t[None, None], size=h.shape[-2:], mode="area")[0, 0] > 0
        ring = (F.max_pool2d(cell[None, None].float(), 5, 1, 2)[0, 0] > 0) & ~cell
        if not ring.any():
            rel.append(float("nan"))
            absolute.append(float("nan"))
            continue
        mod = h.clone()
        mod[:, :, cell] = h[:, :, ring].mean(dim=2, keepdim=True).expand(-1, -1, int(cell.sum()))
        s = score(hidden[:i] + [mod] + hidden[i + 1:])
        rel.append(float((base - s) / base.clamp(min=1e-9)))
        absolute.append(float(base - s))
    return np.array(rel), np.array(absolute)


def x2_faith(model, device, clean_root, every):
    """Pre-registered rule (PLAN.md P4): per image, does the stage with the
    largest grad x act share also give the largest ablation drop? Agreement
    > 0.5 with the 95% CI excluding 0.25 (chance, 4 stages) -> "faithful at the
    top-stage level"; else "not validated", X2 stays a footnote."""
    rows = []
    for dataset, name, image, obj, valid in samples(clean_root, every):
        contrib = stage_attribution(model, image, obj, device)
        share = (np.abs(contrib) / np.abs(contrib).sum(axis=1, keepdims=True)).mean(0)
        rel, absolute = stage_ablation(model, image, obj, device)
        fused, _, _ = run(model, image, device)
        ok = np.isfinite(absolute)
        top_abl = int(np.nanargmax(np.where(ok, absolute, -np.inf)))
        rows.append({"dataset": dataset, "image": name, "share": share.tolist(), "ablation_rel": rel.tolist(),
                     "ablation_abs": absolute.tolist(), "top_share": int(share.argmax()), "top_ablation": top_abl,
                     "all_stages_ablated": bool(ok.all()),
                     "peak": float(fused[obj].max()), "detected": bool(fused[obj].max() >= DETECT_THRESHOLD)})
    rng = np.random.default_rng(config.GLOBAL_SEED)
    print("\nX2 faithfulness: per-stage grad x act share vs local stage ablation (drop of the object's "
          "mean fused score when that stage's features at the object are replaced by their surroundings)")
    for dataset in ("fishyscapes", "carla_pasted"):
        n_all = sum(r["dataset"] == dataset for r in rows)
        used = [r for r in rows if r["dataset"] == dataset and r["detected"]]
        if not used:
            continue
        print(f"  {dataset}: {len(used)} of {n_all} objects detected (peak >= {DETECT_THRESHOLD}); "
              f"undetected ones are excluded (their drops are noise around a ~0 score)")
        share = np.array([r["share"] for r in used])
        rel = np.array([r["ablation_rel"] for r in used])
        print(f"  {dataset} n={len(used)} ({sum(not r['all_stages_ablated'] for r in used)} with a stage not ablatable)")
        for s in range(4):
            fin = rel[:, s][np.isfinite(rel[:, s])]
            m_s, lo_s, hi_s = bootstrap_ci(share[:, s], rng)
            m_a, lo_a, hi_a = bootstrap_ci(fin, rng) if len(fin) else (np.nan,) * 3
            print(f"    stage{s + 1}  grad share {m_s:.2f} [{lo_s:.2f},{hi_s:.2f}]   "
                  f"ablation drop {100 * m_a:+6.1f}% [{100 * lo_a:+.1f}, {100 * hi_a:+.1f}]  (n={len(fin)})")
        agree = [float(r["top_share"] == r["top_ablation"]) for r in used]
        m, lo, hi = bootstrap_ci(agree, rng)
        verdict = ("faithful at the top-stage level" if m > 0.5 and lo > 0.25
                   else "NOT validated (X2 stays a footnote)")
        print(f"    top-stage agreement {m:.2f} [{lo:.2f}, {hi:.2f}]  (chance 0.25; rule: > 0.5 and CI > 0.25)"
              f"  -> {verdict}")
        print(f"    top grad stage counts {np.bincount([r['top_share'] for r in used], minlength=4).tolist()}, "
              f"top ablation stage counts {np.bincount([r['top_ablation'] for r in used], minlength=4).tolist()}")
    return rows


def bernoulli_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def x2(model, device, clean_root, every):
    rows = []
    for dataset, name, image, obj, valid in samples(clean_root, every):
        contrib = stage_attribution(model, image, obj, device)
        share = np.abs(contrib) / np.abs(contrib).sum(axis=1, keepdims=True)
        rows.append({"dataset": dataset, "image": name, "contrib": contrib.tolist(),
                     "share": share.mean(0).tolist()})
    print("\nX2 per-stage gradient x activation on the anomaly's mean logit "
          "(share of |contribution|, averaged over heads, then images; stage 1 = finest, 1/4 res)")
    rng = np.random.default_rng(config.GLOBAL_SEED)
    by_ds = {}
    for dataset in ("fishyscapes", "carla_pasted"):
        used = [r for r in rows if r["dataset"] == dataset]
        if not used:
            continue
        share = np.array([r["share"] for r in used])
        sign = np.sign(np.array([r["contrib"] for r in used]).mean(1)).mean(0)
        by_ds[dataset] = share
        cis = [bootstrap_ci(share[:, s], rng) for s in range(4)]
        cells = "  ".join(f"stage{s + 1} {m:.2f} [{lo:.2f},{hi:.2f}]" for s, (m, lo, hi) in enumerate(cis))
        print(f"  {dataset:12s} n={len(used):3d}  {cells}")
        print(f"  {'':12s}        mean sign of contribution (+1 = raises the score): "
              + "  ".join(f"stage{s + 1} {sign[s]:+.2f}" for s in range(4)))
    if len(by_ds) == 2:
        a, b = by_ds["carla_pasted"][:, 0], by_ds["fishyscapes"][:, 0]
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOTSTRAP)]
        print(f"  stage-1 share, pasted minus real: {a.mean() - b.mean():+.3f} "
              f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]  "
              f"(> 0 = pastes lean more on fine texture/edges: the paste-shortcut prediction)")
    return rows


def x3(model, device, clean_root, every, threshold=0.5):
    """X3: per-pixel split of the 3-head uncertainty. total = H(mean p),
    aleatoric = mean H(p_k), epistemic = total - aleatoric (mutual
    information). Caveat: 3 heads on one frozen encoder underestimate
    epistemic uncertainty. Also tests the video observation: is disagreement
    higher on false positives than on true positives?"""
    rows = []
    for dataset, name, image, obj, valid in samples(clean_root, every):
        _, heads, _ = run(model, image, device)
        p = heads.mean(0)
        total = bernoulli_entropy(p)
        alea = bernoulli_entropy(heads).mean(0)
        maps = {"std": heads.std(0), "epistemic": total - alea, "aleatoric": alea, "score": p}
        band = boundary_band(obj, RING, side="both") & valid
        far_bg = (ndimage.distance_transform_edt(~obj) > CONTEXT_PX) & valid
        regions = {"edge band": band, "interior": obj & ~band, "far background": far_bg,
                   "true positive": (p > threshold) & obj & ~band,
                   "false positive": (p > threshold) & far_bg}
        row = {"dataset": dataset, "image": name}
        for rname, rmask in regions.items():
            if rmask.sum() >= MIN_REGION_PX:
                row[rname] = {k: float(v[rmask].mean()) for k, v in maps.items()}
        rows.append(row)

    rng = np.random.default_rng(config.GLOBAL_SEED)
    print(f"\nX3 uncertainty split (mean per image, then over images; TP/FP at fused > {threshold}, "
          f"FP only >{CONTEXT_PX}px from any object)")
    for dataset in ("fishyscapes", "carla_pasted"):
        used = [r for r in rows if r["dataset"] == dataset]
        if not used:
            continue
        print(f"  {dataset}:")
        for rname in ("edge band", "interior", "far background", "true positive", "false positive"):
            have = [r[rname] for r in used if rname in r]
            if have:
                print(f"    {rname:15s} n={len(have):3d}  " + "  ".join(
                    f"{k} {np.mean([h[k] for h in have]):.4f}" for k in ("score", "std", "epistemic", "aleatoric")))
        both = [r for r in used if "true positive" in r and "false positive" in r]
        if len(both) >= 5:
            for k in ("std", "epistemic"):
                d = [r["false positive"][k] - r["true positive"][k] for r in both]
                m, lo, hi = bootstrap_ci(d, rng)
                print(f"    FP minus TP {k:9s} (paired, n={len(both)}): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        else:
            print(f"    FP vs TP: only {len(both)} images have both -- not tested")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="model_3head_best.pth",
                        help="repo-root checkpoint, NOT config.CHECKPOINT_3HEAD (stale)")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--out", default="explain_out")
    parser.add_argument("--x1b", default=None, metavar="CLEAN_ROOT",
                        help="run X1b instead: e.g. video_town02 (needs video_town02_pasted next to it)")
    parser.add_argument("--every", type=int, default=4, help="X1b/X2/X3: use every Nth frame with a paste")
    parser.add_argument("--x2", action="store_true", help="run X2 (per-stage gradient attribution)")
    parser.add_argument("--x3", action="store_true", help="run X3 (aleatoric/epistemic split)")
    parser.add_argument("--x2-faith", action="store_true",
                        help="run the X2 faithfulness check (local stage ablation vs grad x act)")
    parser.add_argument("--carla", default="video_town02",
                        help="X2/X3: clean CARLA route to add next to Fishyscapes ('' = Fishyscapes only)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = get_device()
    model = load_trained_model(args.raw, device)
    if args.x2 or args.x3 or args.x2_faith:
        for flag, fn in (("x2", x2), ("x3", x3), ("x2_faith", x2_faith)):
            if getattr(args, flag):
                rows = fn(model, device, args.carla, args.every)
                with open(os.path.join(args.out, f"{flag}_per_image.json"), "w") as f:
                    json.dump(rows, f, indent=1)
        return
    if args.x1b:
        rows = x1b(model, device, args.x1b, args.every)
        with open(os.path.join(args.out, "x1b_per_frame.json"), "w") as f:
            json.dump(rows, f, indent=1)
        return
    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    print(f"X1 counterfactual removal: {len(test_pairs)} Fishyscapes test images, ring={RING}px "
          f"at 1024x512 (= r=8 at label res), context beyond {CONTEXT_PX}px")

    rows = []
    skipped = 0
    no_road = 0
    for idx, (image_path, label_path) in enumerate(test_pairs):
        image, obj, valid = load_pair(image_path, label_path)
        if obj.sum() < MIN_OBJECT_PX:
            skipped += 1
            continue
        fused, heads, seg = run(model, image, device)
        rim = boundary_band(obj, RING, side="inner") & obj
        interior = obj & ~rim
        edge_band = boundary_band(obj, RING, side="both")
        # Euclidean, not binary_dilation: iterated 3x3 dilation gives a diamond.
        near = ndimage.distance_transform_edt(~obj) <= CONTEXT_PX
        far_bg = ~near & valid
        obj_fill = ndimage.binary_dilation(obj, iterations=2)
        road = (seg == ROAD_TRAINID) & ~near

        # (condition, region to fill, region to measure)
        tests = [("object", obj_fill, obj), ("edges", edge_band, interior),
                 ("interior", interior, rim)]
        row = {"image": os.path.basename(image_path), "object_px": int(obj.sum()),
               "interior_px": int(interior.sum()),
               "bg_mean": float(fused[far_bg].mean())}
        examples = [(image, fused)]
        for name, fill, measure in tests:
            if measure.sum() < MIN_REGION_PX or fill.sum() == 0:
                continue
            before = float(fused[measure].mean())
            row[f"{name}_before"] = before
            row[f"{name}_before_heads"] = [float(h[measure].mean()) for h in heads]
            edited = {"telea": fill_telea(image, fill)}
            shift = find_road_shift(fill, road, near)
            if shift is not None:
                edited["patch"] = fill_patch(image, fill, shift)
            elif name == "object":
                no_road += 1
            for method, img2 in edited.items():
                f2, h2, _ = run(model, img2, device)
                row[f"{name}_{method}_after"] = float(f2[measure].mean())
                row[f"{name}_{method}_after_heads"] = [float(h[measure].mean()) for h in h2]
                row[f"{name}_{method}_fill_score"] = float(f2[fill].mean())
                row[f"{name}_{method}_bg_after"] = float(f2[far_bg].mean())
                if method == "telea":
                    examples.append((img2, f2))

        f2, _, _ = run(model, blur_context(image, near), device)
        row["context_before"] = float(fused[obj].mean())
        row["context_blur_after"] = float(f2[obj].mean())
        examples.append((blur_context(image, near), f2))
        rows.append(row)

        if len(rows) <= args.examples:
            heat_strip(examples, os.path.join(args.out, f"x1_{len(rows):02d}_{row['image']}"))
        print(f"[{idx + 1}/{len(test_pairs)}] {row['image']}  obj={row['object_px']}px  "
              f"score {row.get('object_before', float('nan')):.3f} -> removed "
              f"{row.get('object_telea_after', float('nan')):.3f} (telea) / "
              f"{row.get('object_patch_after', float('nan')):.3f} (patch)")

    with open(os.path.join(args.out, "x1_per_image.json"), "w") as f:
        json.dump(rows, f, indent=1)

    rng = np.random.default_rng(config.GLOBAL_SEED)
    print(f"\n{len(rows)} images used, {skipped} skipped (object < {MIN_OBJECT_PX}px), "
          f"{no_road} without a clean road patch (telea only)")
    print("mean fused score on the measured region; drop = before - after, 95% CI over images "
          f"({N_BOOTSTRAP} resamples)")
    for name in CONDITIONS:
        for method in (("blur",) if name == "context" else ("telea", "patch")):
            key = f"{name}_{method}_after"
            used = [r for r in rows if key in r]
            if not used:
                continue
            before = np.array([r[f"{name}_before"] for r in used])
            after = np.array([r[key] for r in used])
            drop, lo, hi = bootstrap_ci(before - after, rng)
            # Ratio of means, not mean of per-image ratios: the first run (2026-10-05)
            # printed -15621% because some objects score ~0 before the edit.
            line = (f"  {name:8s} {method:5s} n={len(used):2d}  before {before.mean():.3f}  "
                    f"after {after.mean():.3f}  drop {drop:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                    f"after/before {after.mean() / max(before.mean(), 1e-6):.2f}")
            if name != "context":
                fill_score = np.mean([r[f"{name}_{method}_fill_score"] for r in used])
                bg_shift = np.mean([r[f"{name}_{method}_bg_after"] - r["bg_mean"] for r in used])
                line += f"  | filled-region score {fill_score:.3f}, far-background shift {bg_shift:+.4f}"
            print(line)
    for name in ("object", "edges", "interior"):
        used = [r for r in rows if f"{name}_telea_after_heads" in r]
        if used:
            drops = np.array([[b - a for b, a in zip(r[f"{name}_before_heads"], r[f"{name}_telea_after_heads"])]
                              for r in used]).mean(0)
            print(f"  per-head drop, {name} (telea): " + "  ".join(f"head{i} {d:+.3f}" for i, d in enumerate(drops)))
    print(f"  reference: mean far-background score {np.mean([r['bg_mean'] for r in rows]):.4f}")
    print(f"\nexample strips (original | object | edges | interior | context, telea): {args.out}/")


if __name__ == "__main__":
    main()

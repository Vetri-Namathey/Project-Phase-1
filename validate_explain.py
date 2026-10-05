"""Checks that explain.py's building blocks are correct BEFORE its numbers are trusted.

Geometry and fills are tested on synthetic masks. The gradient hook is tested
against a finite-difference derivative of the real model's heads (in float64);
the stage ablation and the entropy split against cases with a known answer.

    python validate_explain.py                 # all checks, needs the GPU checkpoint
    python validate_explain.py --no-model      # synthetic checks only

Ends with ALL EXPLAIN CHECKS PASS or a list of failures.
"""

import argparse
import copy

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy

import config
import explain as ex

failures = []


def check(name, cond, detail=""):
    detail = str(detail)
    print(("OK   " if cond else "FAIL ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def square(y, x, s):
    m = np.zeros((ex.H, ex.W), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_geometry():
    print("\nregions")
    comp, other = square(200, 300, 20), square(200, 345, 15)
    regions, nbh = ex.regions_of(comp, np.zeros_like(comp))
    check("object removal contains the object", (regions["object"] & comp).sum() == comp.sum())
    check("interior == 12x12 core of a 20x20 square (r=4)", regions["interior"].sum() == 144, regions["interior"].sum())
    ring = regions["ring"]
    check("ring and interior are disjoint", not (ring & regions["interior"]).any())
    check("ring + interior cover the object", ((ring | regions["interior"]) & comp).sum() == comp.sum())
    check("context touches neither ring nor object", not (regions["context"] & (ring | comp)).any())
    check("neighbourhood = object + outer ring", nbh.sum() == comp.sum() + (ring & ~comp).sum())
    outer = (ring & ~comp).sum()
    check("outer ring area close to 4*20*4 + pi*16", 340 <= outer <= 400, outer)
    regions2, _ = ex.regions_of(comp, other)
    check("another object is never edited", not any((v & other).any() for v in regions2.values()))
    check("objects_of drops specks, keeps real objects",
          len(ex.objects_of(comp | square(10, 10, 2))) == 1)


def test_fills():
    print("\nfills")
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (ex.H, ex.W, 3), dtype=np.uint8)
    comp = square(200, 300, 20)
    regions, _ = ex.regions_of(comp, np.zeros_like(comp))
    F_ = regions["ring"]
    for name, out in (("telea", ex.fill_telea(rgb, F_)), ("blur", ex.fill_blur(rgb, F_))):
        check(f"{name}: pixels outside the region are untouched", np.array_equal(out[~F_], rgb[~F_]))
        check(f"{name}: pixels inside the region changed", (out[F_] != rgb[F_]).any())
    road = np.zeros((ex.H, ex.W), bool)
    road[300:500, 100:900] = True
    avoid = ex.bbox_mask(F_, 8)
    shift = ex.find_shift(F_, road, avoid)
    check("find_shift finds free road", shift is not None)
    dy, dx = shift
    moved = ex.shift_mask(F_, dy, dx)
    check("shifted region lies entirely on free road", (moved & ~(road & ~avoid)).sum() == 0)
    ys, xs = np.nonzero(F_)
    hard = ex.fill_patch_hard(rgb, F_, shift)
    check("hard patch: region filled from the shifted source", np.array_equal(hard[ys, xs], rgb[ys + dy, xs + dx]))
    check("hard patch: pixels outside the region are untouched", np.array_equal(hard[~F_], rgb[~F_]))

    # feathered patch (the default patch fill)
    feather = ex.fill_patch(rgb, F_, shift)
    check("feathered patch: inside the region it is exactly the shifted road pixels",
          np.array_equal(feather[ys, xs], rgb[ys + dy, xs + dx]))
    box = ex.bbox_mask(F_, ex.FEATHER_PAD)
    check("feathered patch: nothing changes outside the region's box + pad",
          np.array_equal(feather[~box], rgb[~box]))
    skirt = box & ~F_
    changed = (feather != rgb).any(-1)
    check("feathered patch: the soft skirt is only just outside the region",
          not (changed & ~F_ & (ndimage.distance_transform_edt(~F_) > 9)).any())
    check("feathered patch: the skirt blends (partial change), it does not replace",
          changed[skirt].any() and (np.abs(feather.astype(int) - rgb.astype(int))[skirt].max()
                                    < np.abs(rgb[ys + dy, xs + dx].astype(int) - rgb[ys, xs].astype(int)).max()))
    flat = np.full_like(rgb, 100)
    flat[300:500, 100:900] = 100
    check("feathered patch: a flat road patch on a flat road changes nothing",
          np.array_equal(ex.fill_patch(flat, F_, shift), flat))
    check("find_shift: no road -> None", ex.find_shift(F_, np.zeros_like(road), avoid) is None)
    near = np.zeros_like(road)
    near[150:250, 100:900] = True
    near[190:240, 280:340] = False
    s = ex.find_shift(F_, near | road, avoid | square(190, 280, 60))
    check("find_shift prefers the same row", s is not None and abs(s[0]) < 40, s)


def test_stats():
    print("\nstatistics")
    p = np.array([0.0, 1.0])[:, None]
    u = ex.uncertainty_split(p)
    check("heads that fully disagree: epistemic ~ 1 bit, aleatoric ~ 0", abs(u["epistemic"][0] - 1) < 1e-3 and u["aleatoric"][0] < 1e-3)
    same = ex.uncertainty_split(np.full((3, 5), 0.3))
    check("identical heads: epistemic == 0", np.allclose(same["epistemic"], 0, atol=1e-9))
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.02, 0.98, (3, 200))
    u = ex.uncertainty_split(probs)
    ref_total = np.array([scipy_entropy([m, 1 - m], base=2) for m in probs.mean(0)])
    ref_ale = np.array([np.mean([scipy_entropy([q, 1 - q], base=2) for q in col]) for col in probs.T])
    check("total entropy matches scipy", np.allclose(u["total"], ref_total, atol=1e-6))
    check("aleatoric matches scipy", np.allclose(u["aleatoric"], ref_ale, atol=1e-6))
    check("total = aleatoric + epistemic, epistemic >= 0",
          np.allclose(u["total"], u["aleatoric"] + u["epistemic"], atol=1e-9) and (u["epistemic"] >= 0).all())
    import json
    tree = {"a": (float("nan"), 1.0), "b": np.float32(2.5), "c": np.array([1, float("inf")]),
            "d": np.int64(3), "e": np.bool_(True)}
    safe = ex.json_safe(tree)
    text = json.dumps(safe)
    check("json_safe: NaN/inf become null and the result is strict JSON",
          json.loads(text, parse_constant=lambda c: 1 / 0) == {"a": [None, 1.0], "b": 2.5, "c": [1, None], "d": 3, "e": True},
          text)
    m, lo, hi = ex.cluster_mean([0.5] * 12, np.repeat(np.arange(4), 3))
    check("bootstrap CI of a constant is that constant", m == lo == hi == 0.5)
    vals = rng.normal(1.0, 1.0, 400)
    m, lo, hi = ex.cluster_mean(vals, np.repeat(np.arange(100), 4))
    check("bootstrap CI brackets the sample mean and is narrower than the data spread",
          lo < m < hi and (hi - lo) < 1.0, f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
    d, lo, hi = ex.cluster_diff(rng.normal(2, .5, 200), np.repeat(np.arange(50), 4),
                                rng.normal(1, .5, 200), np.repeat(np.arange(50), 4))
    check("difference CI excludes 0 for a real gap of 1", lo > 0.5 and hi < 1.5, f"{d:.2f} [{lo:.2f}, {hi:.2f}]")


def test_model(checkpoint):
    print("\nmodel hooks (needs the GPU checkpoint)")
    from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
    from utils import get_device, load_trained_model
    device = get_device()
    model = load_trained_model(checkpoint, device)
    _, test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    rgb, anomaly, _ = ex.load_real(*test[0])
    objs = ex.objects_of(anomaly)
    check("the first test image has an object", len(objs) > 0)
    comp = objs[0]
    hs, p0, _ = ex.run_model(model, rgb, device)
    region_t = torch.from_numpy(comp.astype(np.float32)).to(device)
    size = (ex.H, ex.W)

    heads64 = copy.deepcopy(model).ood_heads.double()
    shim = type("M", (), {"ood_heads": heads64})()
    leaves64 = [h.detach().double().clone().requires_grad_(True) for h in hs]

    def target(ls, head=None):
        return ex.ood_target(shim, ls, size, region_t.double(), head)

    g = torch.autograd.grad(target(leaves64), leaves64[0])[0]
    rng = np.random.default_rng(0)
    d = torch.from_numpy(rng.standard_normal(g.shape)).to(device)
    d = d / d.norm()
    eps = 0.5
    with torch.no_grad():
        plus = target([leaves64[0] + eps * d] + leaves64[1:]).item()
        minus = target([leaves64[0] - eps * d] + leaves64[1:]).item()
    num, ana = (plus - minus) / (2 * eps), (g * d).sum().item()
    check("gradient hook matches a finite-difference derivative (float64, random direction)",
          abs(num - ana) <= 0.02 * abs(ana) + 1e-12, f"analytic {ana:.5e} numeric {num:.5e}")

    # Along the gradient itself the derivative is |g|, not ~0 -- a much stricter probe.
    dg = g / g.norm()
    eps = 1e-3 * leaves64[0].norm().item() / 100
    with torch.no_grad():
        plus = target([leaves64[0] + eps * dg] + leaves64[1:]).item()
        minus = target([leaves64[0] - eps * dg] + leaves64[1:]).item()
    num, ana = (plus - minus) / (2 * eps), g.norm().item()
    check("gradient hook matches a finite-difference derivative (float64, along the gradient)",
          abs(num - ana) <= 0.05 * abs(ana), f"|g| {ana:.5e} numeric {num:.5e}")

    gf = torch.autograd.grad(target(leaves64), leaves64)
    gh = [torch.autograd.grad(target(leaves64, h), leaves64) for h in range(3)]
    ok = all(torch.allclose(gf[i], sum(gh[h][i] for h in range(3)) / 3, rtol=1e-6, atol=1e-12) for i in range(4))
    check("fused gradient == mean of the per-head gradients", ok)

    leaves = [h.detach().clone().requires_grad_(True) for h in hs]
    shares, per_head, maps = ex.stage_attribution(model, leaves, size, region_t)
    check("stage shares are a distribution (>=0, sum 1)", (shares >= 0).all() and abs(shares.sum() - 1) < 1e-5, np.round(shares, 3))
    check("per-head shares are distributions", np.allclose(per_head.sum(1), 1, atol=1e-5))
    check("attribution maps have each stage's own resolution",
          [m.shape for m in maps] == [tuple(h.shape[-2:]) for h in hs])

    flat = list(hs)
    flat[0] = hs[0].mean(dim=(2, 3), keepdim=True).expand_as(hs[0]).contiguous()
    rel, absd = ex.stage_ablation(model, flat, size, region_t)
    check("ablating a stage that is already spatially constant changes nothing", abs(rel[0]) < 1e-6, rel[0])
    rel, absd = ex.stage_ablation(model, hs, size, region_t)
    check("ablation returns one finite value per stage (relative and absolute)",
          rel.shape == (4,) and absd.shape == (4,) and np.isfinite(rel).all() and np.isfinite(absd).all(),
          f"rel {np.round(rel, 3)} abs {np.round(absd, 4)}")

    # Editing the features of a stage at a place with NO object must not move
    # the object's score (the edit is local).
    h_far = list(hs)
    sub = hs[2].clone()
    cell = torch.zeros(sub.shape[-2:], dtype=torch.bool, device=device)
    cell[:2, :] = True                                  # top rows of the image: far from any object
    sub[:, :, cell] = sub[:, :, 3:5, :].mean(dim=(2, 3)).unsqueeze(-1).expand(-1, -1, int(cell.sum()))
    h_far[2] = sub
    s0 = (ex.head_probs(model, hs, size).mean(0) * region_t).sum()
    s1 = (ex.head_probs(model, h_far, size).mean(0) * region_t).sum()
    check("editing features far from the object leaves its score unchanged",
          abs((s0 - s1).item()) < 1e-3 * max(abs(s0.item()), 1e-6) + 1e-5, f"{s0.item():.5f} vs {s1.item():.5f}")

    fused = p0.mean(0)
    base_peak, _ = ex.stats_in(fused, comp)
    _, p1, _ = ex.run_model(model, ex.fill_telea(rgb, np.zeros_like(comp)), device)
    check("an empty edit leaves the score map identical", np.allclose(p1, p0, atol=1e-6))
    check("scores are probabilities", 0.0 <= p0.min() and p0.max() <= 1.0 and 0 <= base_peak <= 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/phase2a_coco_run1/model_3head_best.pth")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    test_geometry()
    test_fills()
    test_stats()
    if not args.no_model:
        test_model(args.checkpoint)
    print("\n" + ("ALL EXPLAIN CHECKS PASS" if not failures else f"FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

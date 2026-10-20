"""Live, label-free explanation of one frame, for the demo's analyst panel.

Re-uses the evaluation-time explainability code (explain.py: X1 removal, X2 stage
ablation, X3 uncertainty split) on ONE image, with no ground truth. Everything the
panel says is built from the numbers measured here; there is no language model.

What is label-free and what is not (DISCUSSION_NOTES.md, sections 3 and 5):
  * measured here: score, head disagreement, segmentation class under the region,
    score after removing the region (X1), which encoder stage the score needs (X2),
    aleatoric / epistemic split (X3), and how a calibration changes the score.
  * NOT measurable here: accuracy or calibration ERROR, which need ground truth. So the
    calibration step says the score CHANGED, never that it improved.

All work happens at the model's input size (1024 x 512), like explain.py.
"""

import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

import config
from explain import (CONTEXT_PX, RING, ROAD_TRAINID, blur_context, find_road_shift, fill_patch,
                     fill_telea, run, stage_ablation)

THRESHOLD = 0.47658          # = server.BOX_THRESHOLD, fitted on the Fishyscapes val half
T_WHOLE = 1.2251             # calibrate_compare_T122.log, plain-NLL whole-image temperature
T_EDGE = 4.3592              # edge-band temperature (C2), same log
MIN_AREA_FRAC = 0.0005       # same rule as the drawn boxes
MAX_REGIONS = 3
DISPLAY_W = 720
ORANGE = (232, 80, 26)

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
               "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]
ROAD_USER = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
STAGES = ["fine edges and texture (1/4 resolution)", "local shape (1/8)",
          "object-level shape (1/16)", "scene-level context (1/32)"]


def _entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _calibrated(heads, temperature):
    p = np.clip(heads, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / temperature
    return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)


def _disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), _disk(radius)) > 0


def _position(box, w, h):
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    horiz = "left" if cx < 0.38 else "right" if cx > 0.62 else "centre"
    vert = "upper" if cy < 0.38 else "lower" if cy > 0.62 else "middle"
    return f"{vert} {horiz}" if horiz != "centre" or vert != "middle" else "centre"


def _display(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (DISPLAY_W, int(round(h * DISPLAY_W / w))), interpolation=cv2.INTER_AREA)


def _b64_png(arr):
    import base64
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _caption(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (23, 19, 15), -1)
    cv2.putText(out, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _regions(fused):
    h, w = fused.shape
    labels, n = ndimage.label(fused >= THRESHOLD, structure=np.ones((3, 3)))
    found = []
    for i in range(1, n + 1):
        region = labels == i
        area = int(region.sum())
        if area < MIN_AREA_FRAC * h * w:
            continue
        ys, xs = np.nonzero(region)
        found.append({"mask": region, "area": area, "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    found.sort(key=lambda r: r["area"], reverse=True)
    return found[:MAX_REGIONS]


@torch.no_grad()
def explain_image(model, device, pil_image):
    t0 = time.perf_counter()
    image = np.asarray(pil_image.convert("RGB").resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR),
                       dtype=np.uint8).copy()
    h, w = image.shape[:2]
    fused, heads, seg = run(model, image, device)
    disagreement = heads.std(axis=0)
    total_h = _entropy(heads.mean(axis=0))
    aleatoric = _entropy(heads).mean(axis=0)
    epistemic = np.clip(total_h - aleatoric, 0, None)
    cal_whole, cal_edge = _calibrated(heads, T_WHOLE), _calibrated(heads, T_EDGE)
    road = seg == ROAD_TRAINID

    regions = _regions(fused)
    out_regions = []
    removed_img = None
    for idx, reg in enumerate(regions):
        m = reg["mask"]
        grown = _dilate(m, 2)
        ring = _dilate(m, RING) & ~_erode(m, RING)
        classes, counts = np.unique(seg[m], return_counts=True)
        comp = sorted(((CLASS_NAMES[int(c)], float(n) / m.sum()) for c, n in zip(classes, counts)),
                      key=lambda t: t[1], reverse=True)[:3]

        # X1: fill the region from its surroundings and re-run the model
        filled = fill_telea(image, grown)
        f_after, _, _ = run(model, filled, device)
        shift = find_road_shift(grown, road, _dilate(m, 6))
        patch_score = None
        if shift is not None:
            f_patch, _, _ = run(model, fill_patch(image, grown, shift), device)
            patch_score = float(f_patch[m].mean())
        before = float(fused[m].mean())
        x1 = {"before": before, "after_telea": float(f_after[m].mean()), "after_patch": patch_score,
              "fill_peak": float(f_after[grown].max())}
        if idx == 0:
            removed_img = filled
            keep = _dilate(m, CONTEXT_PX)
            f_ctx, _, _ = run(model, blur_context(image, keep), device)
            x1["context_blur"] = float(f_ctx[m].mean())

        rel, absolute = stage_ablation(model, image, m, device)
        rel = [None if np.isnan(v) else float(v) for v in rel]
        top_stage = int(np.nanargmax(np.array([-1 if v is None else v for v in rel]))) if any(v is not None for v in rel) else None

        out_regions.append({
            "id": idx + 1, "box": reg["box"], "area_frac": reg["area"] / (h * w),
            "position": _position(reg["box"], w, h),
            "score_mean": before, "score_peak": float(fused[m].max()),
            "classes": comp,
            "disagreement_inside": float(disagreement[m].mean()), "disagreement_edge": float(disagreement[ring].mean()),
            "aleatoric_edge": float(aleatoric[ring].mean()), "epistemic_edge": float(epistemic[ring].mean()),
            "cal_whole": float(cal_whole[m].mean()), "cal_edge": float(cal_edge[m].mean()),
            "x1": x1, "x2": {"relative_drop": rel, "top_stage": top_stage},
        })

    annotated = _display(image).copy()
    sx = annotated.shape[1] / w
    for r in out_regions:
        x0, y0, x1_, y1_ = [int(v * sx) for v in r["box"]]
        cv2.rectangle(annotated, (x0, y0), (x1_, y1_), ORANGE, 3)
        cv2.circle(annotated, (x0 + 14, max(14, y0 - 2)), 13, ORANGE, -1)
        cv2.putText(annotated, str(r["id"]), (x0 + 8, max(14, y0 - 2) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    images = {"annotated": _b64_png(annotated)}
    if removed_img is not None:
        pair = np.concatenate([_caption(_display(image), "Before"),
                               _caption(_display(removed_img), "Region 1 filled from its surroundings")], axis=1)
        images["removed"] = _b64_png(pair)

    result = {"threshold": THRESHOLD, "max_score": float(fused.max()), "regions": out_regions, "images": images,
              "temperatures": {"whole": T_WHOLE, "edge": T_EDGE}}
    result["messages"] = narrate(result)
    result["timing_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return result


def _pct(x):
    return f"{100 * x:.0f}%"


def narrate(res):
    """Plain-language steps built only from measured numbers. Hedged on purpose."""
    regions = res["regions"]
    thr = res["threshold"]
    if not regions:
        return [{"id": "none", "title": "Nothing flagged",
                 "text": f"No region in this frame reaches the detection threshold of {thr:.3f} "
                         f"(the highest score anywhere is {res['max_score']:.3f}), so there is no box to explain. "
                         "A threshold fitted on road-scene validation data is deliberately conservative: it "
                         "avoids drawing boxes where nothing stands out.",
                 "facts": [("Highest score", f"{res['max_score']:.3f}"), ("Threshold", f"{thr:.3f}")], "image": None}]
    r = regions[0]
    msgs = []
    n = len(regions)
    msgs.append({
        "id": "found", "title": "What I flagged",
        "text": (f"{n} region{'s' if n > 1 else ''} reach{'' if n > 1 else 'es'} the detection threshold of {thr:.3f}. "
                 f"Region 1 is the largest, in the {r['position']} of the frame: it covers {100 * r['area_frac']:.1f}% of the image, "
                 f"with a mean anomaly score of {r['score_mean']:.2f} and a peak of {r['score_peak']:.2f}."
                 + (f" The other {'region is' if n == 2 else 'regions are'} smaller and are summarised at the end." if n > 1 else "")),
        "facts": [("Mean score", f"{r['score_mean']:.2f}"), ("Peak score", f"{r['score_peak']:.2f}"),
                  ("Area", f"{100 * r['area_frac']:.1f}%")], "image": "annotated"})

    top, top_frac = r["classes"][0]
    comp = ", ".join(f"{c} {_pct(f)}" for c, f in r["classes"])
    if top in ROAD_USER and top_frac >= 0.5:
        reading = (f"The segmentation network recognises most of this region as '{top}', a class it was trained on. "
                   "The anomaly heads flag it anyway, which can happen on an unusual part of a known object, "
                   "for example wheels or an odd texture, so this may be a false alarm.")
    elif top == "road" and top_frac >= 0.5:
        reading = "Most of the region lies on what the segmentation network calls road, which fits an object lying on the road surface."
    else:
        reading = f"The segmentation network does not settle on one class here (mostly '{top}')."
    msgs.append({"id": "scene", "title": "What the segmenter sees there",
                 "text": f"Inside the box the segmentation labels the pixels as: {comp}. " + reading,
                 "facts": [(c, _pct(f)) for c, f in r["classes"]], "image": None})

    x1 = r["x1"]
    d_telea = max((x1["before"] - x1["after_telea"]) / max(x1["before"], 1e-6), 0)
    d_patch = None if x1["after_patch"] is None else max((x1["before"] - x1["after_patch"]) / max(x1["before"], 1e-6), 0)
    drop = max(d_telea, d_patch if d_patch is not None else 0)
    how = (f"Smoothing it over from its surroundings gives {x1['after_telea']:.2f} ({_pct(d_telea)} lower), but that fill leaves a "
           f"smudge the model can itself flag, so it undersells the effect. "
           + (f"Pasting a patch of real road over it gives {x1['after_patch']:.2f} ({_pct(d_patch)} lower), the cleaner test. "
              if x1["after_patch"] is not None else "No clean patch of road was available in this frame to use as a second test. "))
    if drop >= 0.5:
        verdict = "So most of the evidence is the content of the region itself."
    elif drop >= 0.2:
        verdict = "So the region itself matters, but part of the score survives its removal."
    else:
        verdict = ("So the score barely moves: either the evidence lies in the surroundings, or the replacement is itself flagged. "
                   "Read removal numbers as a lower bound on the effect.")
    ctx = (f" Blurring everything more than {CONTEXT_PX} pixels away leaves the score at {x1['context_blur']:.2f}, "
           f"{'so the score is local to the region' if x1['context_blur'] >= 0.8 * x1['before'] else 'so the surroundings contribute to it'}."
           if "context_blur" in x1 else "")
    msgs.append({"id": "removal", "title": "Would the score survive without it?",
                 "text": (f"I erased region 1 and ran the model again; its mean score was {x1['before']:.2f}. {how}{verdict}{ctx}"),
                 "facts": [("Before", f"{x1['before']:.2f}"), ("Smoothed fill", f"{x1['after_telea']:.2f}")]
                          + ([("Road patch", f"{x1['after_patch']:.2f}")] if x1["after_patch"] is not None else []),
                 "image": "removed" if "removed" in res["images"] else None})

    rel, ts = r["x2"]["relative_drop"], r["x2"]["top_stage"]
    if ts is not None:
        parts = [f"stage {i + 1}: {_pct(max(v, 0))}" for i, v in enumerate(rel) if v is not None]
        msgs.append({"id": "stage", "title": "Which part of the network carries it",
                     "text": (f"The encoder has four stages, from fine detail to scene context. I switched each one off at the region, one at a time, "
                              f"and measured how much the score fell. It depends most on stage {ts + 1}, {STAGES[ts]}: "
                              f"removing it lowers the score by {_pct(max(rel[ts], 0))}. All four: {'; '.join(parts)}."),
                     "facts": [(f"Stage {i + 1}", _pct(max(v, 0))) for i, v in enumerate(rel) if v is not None], "image": None})

    tot = r["aleatoric_edge"] + r["epistemic_edge"]
    epi_share = r["epistemic_edge"] / tot if tot > 0 else 0.0
    agree = ("The three heads largely agree" if r["disagreement_inside"] < 0.05 else "The three heads differ noticeably")
    msgs.append({"id": "uncertainty", "title": "How sure are the three heads?",
                 "text": (f"{agree} inside the region (mean disagreement {r['disagreement_inside']:.3f}) and differ more along its outline "
                          f"({r['disagreement_edge']:.3f}). In that edge band, {_pct(1 - epi_share)} of the uncertainty is inherent "
                          f"(every head is unsure) and {_pct(epi_share)} is disagreement between heads. Three heads on one frozen encoder "
                          "underestimate the second kind, so treat it as a lower bound."),
                 "facts": [("Disagreement inside", f"{r['disagreement_inside']:.3f}"), ("At the outline", f"{r['disagreement_edge']:.3f}"),
                           ("Inherent", _pct(1 - epi_share))], "image": None})

    msgs.append({"id": "calibration", "title": "How calibration changes the number",
                 "text": (f"Region 1's mean score is {r['score_mean']:.2f} raw. With the whole-image temperature (T = {res['temperatures']['whole']}) "
                          f"it becomes {r['cal_whole']:.2f}; with the edge-fitted temperature (T = {res['temperatures']['edge']}) it becomes "
                          f"{r['cal_edge']:.2f}. That is how the score changes, not whether the change is an improvement: judging that needs "
                          "ground truth, which an uploaded image does not have. On the labelled test set the edge temperature improves "
                          "calibration at object outlines but worsens it over the whole image."),
                 "facts": [("Raw", f"{r['score_mean']:.2f}"), (f"T = {res['temperatures']['whole']}", f"{r['cal_whole']:.2f}"),
                           (f"T = {res['temperatures']['edge']}", f"{r['cal_edge']:.2f}")], "image": None})

    if n > 1:
        rows = []
        for o in regions[1:]:
            cls, fr = o["classes"][0]
            dd = (o["x1"]["before"] - o["x1"]["after_telea"]) / max(o["x1"]["before"], 1e-6)
            rows.append(f"Region {o['id']} ({o['position']}): score {o['score_mean']:.2f}, mostly '{cls}' ({_pct(fr)}), "
                        f"{_pct(max(dd, 0))} lower with the region removed.")
        known = [o for o in regions[1:] if o["classes"][0][0] in ROAD_USER and o["classes"][0][1] >= 0.5]
        if known:
            names = sorted({o["classes"][0][0] for o in known})
            rows.append(f"Region{'s' if len(known) > 1 else ''} {', '.join(str(o['id']) for o in known)} sit{'' if len(known) > 1 else 's'} on "
                        f"{'a recognised ' if len(names) == 1 else 'recognised '}{'/'.join(names)}, so {'they' if len(known) > 1 else 'it'} may be "
                        "false alarms on parts of a known vehicle such as its wheels. I can only see the class beneath a region, not what it is.")
        msgs.append({"id": "others", "title": "The other regions", "text": " ".join(rows),
                     "facts": [(f"Region {o['id']}", f"{o['score_mean']:.2f}") for o in regions[1:]], "image": None})
    return msgs

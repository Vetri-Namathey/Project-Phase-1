"""PLAN.md X4: turn explain.py's per-image JSON into what the /demo
explainability panel shows. CPU only, no model.

Numbers are computed from explain_out/*.json here rather than typed into the
frontend (the hand-copied EXPERIMENT_HISTORY in server.py is the cautionary
example). Writes static/explain/summary.json and copies one X1 example strip.

    python explain_export.py
"""

import json
import os
import shutil

import numpy as np

SRC = "explain_out"
DST = os.path.join("static", "explain")
# x1_03: the toy car. Removing it leaves an inpainted smudge that scores
# higher than the car did, which is the X1 confound in one picture.
EXAMPLE = "x1_03_01_Hanns_Klemm_Str_45_000010_000200_leftImg8bit.png"


def load(name):
    with open(os.path.join(SRC, name)) as f:
        return json.load(f)


def main():
    os.makedirs(DST, exist_ok=True)
    x1, x1b, x2, x3 = (load(n) for n in ("x1_per_image.json", "x1b_per_frame.json",
                                         "x2_per_image.json", "x3_per_image.json"))
    mean = lambda rows, k: float(np.mean([r[k] for r in rows if k in r]))

    summary = {
        "x1": {"n": len(x1),
               "object_before": mean(x1, "object_before"),
               "object_after_telea": mean(x1, "object_telea_after"),
               "fill_score_telea": mean(x1, "object_telea_fill_score"),
               "background": mean(x1, "bg_mean")},
        "x1b": {"n": len(x1b), "with_paste": mean(x1b, "pasted"), "true_removal": mean(x1b, "clean"),
                "telea_removal": mean(x1b, "telea"), "background": mean(x1b, "bg")},
        "x2": {},
        "x3": {},
    }
    for ds in ("fishyscapes", "carla_pasted"):
        rows = [r for r in x2 if r["dataset"] == ds]
        contrib = np.array([r["contrib"] for r in rows]).mean(1)  # (images, 4), mean over heads
        summary["x2"][ds] = {"n": len(rows),
                             "share": np.mean([r["share"] for r in rows], 0).tolist(),
                             "sign": np.sign(contrib).mean(0).tolist()}
        rows3 = [r for r in x3 if r["dataset"] == ds]
        summary["x3"][ds] = {
            region: dict({k: float(np.mean([r[region][k] for r in rows3 if region in r]))
                          for k in ("score", "std", "epistemic", "aleatoric")},
                         n=sum(region in r for r in rows3))
            for region in ("edge band", "interior", "far background", "true positive", "false positive")}

    # X2 faithfulness (explain.py --x2-faith, explain_x2_faith.log): detected objects only.
    if os.path.exists(os.path.join(SRC, "x2_faith_per_image.json")):
        xf = load("x2_faith_per_image.json")
        summary["x2_faith"] = {}
        for ds in ("fishyscapes", "carla_pasted"):
            rows = [r for r in xf if r["dataset"] == ds and r["detected"]]
            if rows:
                summary["x2_faith"][ds] = {
                    "n": len(rows),
                    "ablation_rel": np.nanmean([r["ablation_rel"] for r in rows], 0).tolist(),
                    "agreement": float(np.mean([r["top_share"] == r["top_ablation"] for r in rows]))}
    # X5 same-photo paste test (paste_test.py, paste_test.log): point estimates only;
    # CIs live in the log.
    if os.path.exists(os.path.join(SRC, "paste_test_per_image.json")):
        x5 = load("paste_test_per_image.json")
        arm_mean = lambda arm, k: float(np.mean([r[arm][k] for r in x5 if arm in r and r[arm].get(k) is not None]))
        both = [r for r in x5 if "real" in r and "coco_unseen" in r]
        summary["x5"] = {
            "n_photos": len(x5), "n_real": len(both),
            "arms": {arm: {"score": arm_mean(arm, "score"), "band_gap": arm_mean(arm, "band_gap"),
                           "px": arm_mean(arm, "px")}
                     for arm in ("coco_seen", "coco_unseen", "carla_bank", "real")},
            "edge_gap_unseen_minus_real": float(np.mean([r["coco_unseen"]["band_gap"] - r["real"]["band_gap"] for r in both])),
            "score_seen_minus_unseen": float(np.mean([r["coco_seen"]["score"] - r["coco_unseen"]["score"]
                                                      for r in x5 if "coco_seen" in r and "coco_unseen" in r])),
            "removed": arm_mean("coco_unseen", "score_original_photo")}

    with open(os.path.join(DST, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    shutil.copy(os.path.join(SRC, EXAMPLE), os.path.join(DST, "x1_example.png"))
    print(json.dumps(summary, indent=1))
    print(f"-> {DST}/summary.json, {DST}/x1_example.png")


if __name__ == "__main__":
    main()

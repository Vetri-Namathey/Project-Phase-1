"""Thesis figures (CPU only, read-only on data/). Run from the repo root:

    python paper/make_figures.py [--out ../Thesis_Template___Amrita_AIE/Thesis_Template___Amrita_AIE/images]

Plotted numbers are copied from the logs named next to them (rule 12); image
figures are copied or composed from files the pipeline already wrote.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # rule 13 ("" is unset on Windows)
sys.path.insert(0, os.getcwd())

import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
ACCENT, GREY, DARK = "#3D5AFE", "#96958E", "#0A0A0A"


def architecture(out):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=DARK, bold=False, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=DARK))

    box(0.1, 2.1, 1.6, 1.0, "RGB image\n1024 x 512")
    box(2.2, 1.4, 2.2, 2.4, "SegFormer-B5\nencoder\n(Cityscapes-\nfine-tuned,\nFROZEN)", fc="#EDEDEA", bold=True)
    arrow(1.7, 2.6, 2.2, 2.6)
    for i, (lab, y) in enumerate([("stage 1  1/4  64 ch", 3.9), ("stage 2  1/8  128 ch", 3.1),
                                  ("stage 3  1/16  320 ch", 2.3), ("stage 4  1/32  512 ch", 1.5)]):
        box(4.9, y - 0.3, 2.0, 0.6, lab, fs=9)
        arrow(4.4, 2.6, 4.9, y)
    box(7.4, 3.9, 2.7, 0.9, "Segmentation head\n19 Cityscapes classes", fs=9)
    for k, y in enumerate([2.75, 1.95, 1.15]):
        box(7.4, y - 0.32, 2.7, 0.64, f"OOD head {k + 1}  (seed {[42, 123, 7][k]})\nconv3x3-128, dropout 0.3, conv3x3-1", fs=7.5,
            fc="#DCE1FF", ec=ACCENT)
    for y in (4.35, 2.75, 1.95, 1.15):
        arrow(6.9, 2.6, 7.4, y)
    box(10.4, 3.95, 2.4, 0.8, "class map", fs=9)
    arrow(10.1, 4.35, 10.4, 4.35)
    box(10.4, 2.3, 2.4, 0.9, "anomaly score\nmean of 3 sigmoids", fc=ACCENT, ec=ACCENT, fs=9)
    ax.texts[-1].set_color("white")
    box(10.4, 0.9, 2.4, 0.9, "head disagreement\n(spread of 3 heads)", fs=9)
    for y in (2.75, 1.95, 1.15):
        arrow(10.1, y, 10.4, 2.75)
        arrow(10.1, y, 10.4, 1.35)
    ax.text(0.1, 0.35, "Training: Cityscapes train images with CutMix pastes from a 2000-object bank "
            "(500 CARLA tiles of 45 objects + 1500 COCO cutouts); only the heads are trained "
            "(L = L_seg + 1.0 * L_OOD, OOD BCE pos_weight 20; AdamW, lr 1e-4, 8 epochs, epoch 4 selected on val AP).",
            fontsize=8.5, color="#5C5C58", wrap=True)
    fig.savefig(os.path.join(out, "fig_architecture.png"))
    plt.close(fig)


def edge_band(out):
    from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
    from metrics import boundary_band
    _, test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    best = None
    for img_p, lab_p in test:
        lab = np.array(Image.open(lab_p))
        n = int((lab == 1).sum())
        if n and (best is None or abs(n - 6000) < abs(best[0] - 6000)):
            best = (n, img_p, lab_p)
    _, img_p, lab_p = best
    img = np.array(Image.open(img_p).convert("RGB"))
    lab = np.array(Image.open(lab_p))
    obj, valid = lab == 1, lab != 255
    band = boundary_band(obj, 8, valid=valid, side="both")
    lbl, _ = __import__("scipy.ndimage", fromlist=["label"]).label(obj)
    biggest = np.argmax(np.bincount(lbl.ravel())[1:]) + 1
    ys, xs = np.nonzero(lbl == biggest)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = 240, 480
    y0 = int(np.clip(cy - h // 2, 0, obj.shape[0] - h))
    x0 = int(np.clip(cx - w // 2, 0, obj.shape[1] - w))
    crop = img[y0:y0 + h, x0:x0 + w].astype(float)
    o, b = obj[y0:y0 + h, x0:x0 + w], band[y0:y0 + h, x0:x0 + w]
    over = crop.copy()
    over[o & ~b] = 0.5 * over[o & ~b] + 0.5 * np.array([31, 164, 99])
    over[b] = 0.35 * over[b] + 0.65 * np.array([61, 90, 254])
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.3))
    for a, im, t in zip(axs, (crop, over), ("Fishyscapes Lost & Found test crop",
                                             "object interior (green) and r = 8 px edge band (blue)")):
        a.imshow(im.astype(np.uint8))
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.savefig(os.path.join(out, "fig_edge_band.png"))
    plt.close(fig)


def tradeoff(out):
    # eval_fishyscapes_T122.log / calibrate_compare_T122.log (Fishyscapes test half)
    pts = [("raw", 0.0004, 0.2273), ("T = 1.2251 (whole)", 0.00012, 0.2095), ("T = 3.1516 (50/50)", 0.0177, 0.1277),
           ("T = 4.3592 (edge, C2)", 0.0420, 0.0942), ("C3 T(d)", 0.0193, 0.1312), ("C5 context", 0.0264, 0.0738),
           ("L_calib", 0.0005, 0.2396)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    temps = pts[1:4]
    ax.plot([p[1] for p in temps], [p[2] for p in temps], "--", color=GREY, zorder=1)
    for name, x, y in pts:
        c = ACCENT if name.startswith("T =") else (DARK if name == "raw" else "#D0452F")
        ax.scatter(x, y, s=60, color=c, zorder=2)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("whole-image ECE (lower is better)")
    ax.set_ylabel("edge-band ECE, r = 8 px (lower is better)")
    ax.set_title("No single temperature calibrates both the image and the edges", fontsize=11)
    ax.set_xlim(-0.003, 0.05)
    fig.savefig(os.path.join(out, "fig_temperature_tradeoff.png"))
    plt.close(fig)


def demo_outputs(out):
    d = os.path.join("static", "generated", "test_half", "demo_00")
    names = [("raw.png", "input"), ("ground_truth.png", "ground truth"), ("fused.png", "anomaly score (3-head mean)"),
             ("disagreement.png", "head disagreement"), ("segmentation.png", "segmentation"),
             ("detection.png", "detection (score >= 0.477)")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 4.8))
    for a, (f, t) in zip(axs.ravel(), names):
        a.imshow(Image.open(os.path.join(d, f)).convert("RGB"))
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.savefig(os.path.join(out, "fig_demo_outputs.png"))
    plt.close(fig)


def x5_scores(out):
    # paste_test.log
    arms = ["COCO paste,\nnever trained", "COCO paste,\nin training bank", "CARLA paste\n(aligned)", "real object,\nsame photos"]
    score = [0.939, 0.927, 0.842, 0.347]
    gap = [0.298, 0.292, 0.234, -0.179]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    cols = [DARK, DARK, GREY, ACCENT]
    axs[0].bar(arms, score, color=cols)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("mean anomaly score on the object", fontsize=11)
    axs[1].bar(arms, gap, color=cols)
    axs[1].axhline(0, color=DARK, lw=0.8)
    axs[1].set_title("edge gap: score - true rate in the edge band\n(> 0 over-confident, < 0 under-confident)", fontsize=10)
    for a, vals in zip(axs, (score, gap)):
        for i, v in enumerate(vals):
            a.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}" if a is axs[1] else f"{v:.3f}", ha="center", fontsize=9)
        a.tick_params(axis="x", labelsize=9)
    fig.savefig(os.path.join(out, "fig_x5_scores.png"))
    plt.close(fig)


def x2_ablation(out):
    # explain_x2_faith.log (detected objects only)
    st = ["stage 1\n(1/4)", "stage 2\n(1/8)", "stage 3\n(1/16)", "stage 4\n(1/32)"]
    data = {"Real objects (Fishyscapes, n = 24)": ([0.16, 0.21, 0.29, 0.33], [0.278, 0.517, 0.744, 0.282]),
            "Pasted objects (CARLA route, n = 75)": ([0.17, 0.35, 0.29, 0.19], [0.027, 0.158, 0.204, 0.124])}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    x = np.arange(4)
    for a, (t, (g, ab)) in zip(axs, data.items()):
        a.bar(x - 0.2, g, 0.4, color=GREY, label="gradient x activation share")
        a.bar(x + 0.2, ab, 0.4, color=ACCENT, label="score drop when the stage is ablated")
        a.set_xticks(x)
        a.set_xticklabels(st)
        a.set_title(t, fontsize=11)
    axs[0].legend(fontsize=9, frameon=False)
    fig.savefig(os.path.join(out, "fig_x2_ablation.png"))
    plt.close(fig)


def ra21(out):
    # eval_road_anomaly21.log
    imgs = [("0000", 8.32, 0.6094), ("0001", 7.24, 0.2417), ("0002", 7.94, 0.3395), ("0003", 17.99, 0.4371),
            ("0004", 36.76, 0.4449), ("0005", 10.82, 0.0976), ("0006", 6.78, 0.8133), ("0007", 22.19, 0.2686),
            ("0008", 27.87, 0.5135), ("0009", 1.69, 0.7291)]
    imgs.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar([f"#{i}\n{a:.1f}%" for i, a, _ in imgs], [ap for *_, ap in imgs], color=ACCENT)
    ax.axhline(0.3084, color=DARK, ls="--", lw=1)
    ax.text(9.4, 0.3084 + 0.02, "TwinGuard pooled AP 0.308", ha="right", fontsize=9)
    ax.axhline(0.4744, color=GREY, ls=":", lw=1.2)
    ax.text(9.4, 0.4744 + 0.02, "MSP pooled AP 0.474", ha="right", fontsize=9, color="#5C5C58")
    ax.set_ylabel("TwinGuard AP")
    ax.set_xlabel("RoadAnomaly21 validation image (anomaly share of pixels)")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(os.path.join(out, "fig_ra21_per_image.png"))
    plt.close(fig)


def video_frame(out):
    import av
    path = os.path.join("static", "video", "twinguard_video_town02_pasted.mp4")
    with av.open(path) as c:
        stream = c.streams.video[0]
        target = int(stream.frames * 0.4) if stream.frames else 150
        for i, frame in enumerate(c.decode(stream)):
            if i == target:
                frame.to_image().save(os.path.join(out, "fig_video_frame.png"))
                return


def copies(out):
    pairs = [(os.path.join("static", "calibration_reliability_T122.png"), "fig_reliability.png"),
             (os.path.join("static", "explain", "x1_example.png"), "fig_x1_removal.png"),
             (os.path.join("explain_out", "paste_test_00.png"), "fig_x5_paste_example.png"),
             (os.path.join("static", "generated", "training_cutmix", "sample_00", "overlay.png"), "fig_cutmix_sample.png")]
    for src, dst in pairs:
        shutil.copy(src, os.path.join(out, dst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("..", "Thesis_Template___Amrita_AIE", "Thesis_Template___Amrita_AIE", "images"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in (architecture, edge_band, tradeoff, demo_outputs, x5_scores, x2_ablation, ra21, video_frame, copies):
        fn(args.out)
        print("done:", fn.__name__)
    print(sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()

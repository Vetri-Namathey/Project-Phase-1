"""Measure the dual-mode inference claim instead of asserting it.

The abstract promises a "single deterministic pass sustains continuous
operation within real-time bounds" while dense uncertainty is reserved for
"safety-triggered events". Section III-A puts numbers on it: <35ms
continuous, <100ms safety-triggered. The project plan's own targets for the
trigger itself are <5% of normal frames and >90% of anomalous ones.

None of that is true by construction. It depends on the threshold, and the
right threshold depends on the trained model. This script measures all of
it against a real checkpoint:

  1. latency, both modes, on real images
  2. trigger rate on NORMAL frames   (Cityscapes val -- no anomalies present)
  3. trigger rate on ANOMALOUS frames (Fishyscapes -- anomalies present)
  4. a threshold sweep, so SAFETY_TRIGGER_THRESHOLD can be set from evidence

    python check_dual_mode.py
    python check_dual_mode.py --n-images 40 --passes 10

Latency numbers are only meaningful on the GPU the system would deploy on.
On CPU the script still reports trigger rates correctly, but it will say so
rather than let you quote a CPU timing as a real-time result.
"""

import argparse
import time

import numpy as np
import torch

import config
from data.cityscapes_dataset import CityscapesDataset
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from utils import get_device, load_trained_model


AREA_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
AREA_MIN_PIXELS = (10, 25, 50, 100, 200, 400, 800)


@torch.no_grad()
def peak_scores(model, tensors):
    """Continuous-mode statistics per image -- what a trigger can see.

    Returns (peak score, {t: number of pixels with fused score >= t}). The
    peak is what the current trigger uses; the counts are for an area
    trigger, which one stray pixel cannot fire.
    """
    peaks, counts = [], {t: [] for t in AREA_THRESHOLDS}
    for t in tensors:
        hidden = model.encode(t)
        scores = model.heads_from_features(hidden, t.shape[-2:], stochastic=False)
        fused = scores.mean(dim=1)
        peaks.append(float(fused.max()))
        for thr in AREA_THRESHOLDS:
            counts[thr].append(int((fused >= thr).sum()))
    return np.array(peaks), {k: np.array(v) for k, v in counts.items()}


@torch.no_grad()
def time_modes(model, tensor, n_passes, repeats=10):
    """Wall-clock for each mode. Warms up first, and synchronises CUDA --
    without both, the numbers are fiction."""
    device = tensor.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(3):                       # warm-up: cuDNN autotune, allocator
        model.predict_dual_mode(tensor, n_passes=n_passes, force_safety=True)
    sync()

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_dual_mode(tensor, trigger_threshold=1.01)   # never triggers
    sync()
    continuous_ms = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_dual_mode(tensor, n_passes=n_passes, force_safety=True)
    sync()
    safety_ms = (time.perf_counter() - t0) / repeats * 1000

    return continuous_ms, safety_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--passes", type=int, default=config.MC_DROPOUT_PASSES)
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_3HEAD,
                        help="checkpoint to inspect (default: config.CHECKPOINT_3HEAD)")
    args = parser.parse_args()

    device = get_device()
    model = load_trained_model(args.checkpoint, device=device, num_heads=len(config.OOD_HEAD_SEEDS_3HEAD))

    # NORMAL frames: Cityscapes val. Ordinary driving, no anomalies -- every
    # trigger here is a false alarm that costs latency for nothing.
    normal_ds = CityscapesDataset(split="val", normalize=True)
    stride = max(1, len(normal_ds) // args.n_images)
    normal = [normal_ds[i][0].unsqueeze(0).to(device)
              for i in range(0, stride * args.n_images, stride)]

    # ANOMALOUS frames: the Fishyscapes test half. Every missed trigger here
    # is an anomaly that never got dense uncertainty computed for it.
    _, test_pairs = split_fishyscapes_pairs(list_fishyscapes_pairs())
    anomalous = [load_image_tensor(p, device) for p, _ in test_pairs[:args.n_images]]

    print(f"\nmeasuring on {len(normal)} normal + {len(anomalous)} anomalous frames, "
          f"N={args.passes} passes")

    print("\n=== latency ===")
    cont_ms, safe_ms = time_modes(model, anomalous[0], args.passes)
    print(f"  continuous      {cont_ms:7.1f} ms   (paper target <35ms)")
    print(f"  safety-triggered{safe_ms:7.1f} ms   (paper target <100ms)")
    print(f"  cost of escalating: {safe_ms - cont_ms:.1f} ms for {args.passes} "
          f"passes x {len(model.ood_heads)} heads")
    if device.type != "cuda":
        print("  NOTE: CPU timings. Not comparable to the paper's targets --")
        print("        re-run on the deployment GPU before quoting these.")

    print("\n=== trigger rates by threshold ===")
    normal_peaks, normal_counts = peak_scores(model, normal)
    anom_peaks, anom_counts = peak_scores(model, anomalous)

    print(f"  peak score on normal frames   : "
          f"median {np.median(normal_peaks):.4f}  max {normal_peaks.max():.4f}")
    print(f"  peak score on anomalous frames: "
          f"median {np.median(anom_peaks):.4f}  min {anom_peaks.min():.4f}")

    print(f"\n  {'threshold':>10} {'normal FP':>11} {'anomaly TP':>11}   verdict")
    print("  " + "-" * 50)
    best = None
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        fp = float((normal_peaks >= thr).mean())
        tp = float((anom_peaks >= thr).mean())
        ok = fp <= 0.05 and tp >= 0.90
        if ok and best is None:
            best = thr
        print(f"  {thr:>10.2f} {fp:>10.1%} {tp:>11.1%}   {'MEETS TARGET' if ok else ''}")

    print(f"\n  targets: normal false-trigger <=5%, anomaly true-trigger >=90%")
    if best is None:
        print("  NO threshold meets both targets on this checkpoint.")
        print("  That is a reportable finding, not a script failure -- it means the")
        print("  dual-mode latency claim is not yet supported and needs either a")
        print("  better-separated detector or a different trigger statistic")
        print("  (peak pixel is sensitive to a single noisy pixel; an area-based")
        print("  trigger, e.g. 'N pixels above t', is the usual next thing to try).")
    else:
        print(f"  lowest threshold meeting both: {best:.2f}")
        print(f"  set SAFETY_TRIGGER_THRESHOLD = {best:.2f} in config.py "
              f"(currently {config.SAFETY_TRIGGER_THRESHOLD})")

    # Area trigger: fire only if at least N pixels score >= t. A single noisy
    # pixel can no longer escalate a whole frame.
    h, w = anomalous[0].shape[-2:]
    print(f"\n=== area trigger: >= N pixels with score >= t ({h}x{w} input) ===")
    print(f"  {'t':>5} {'N':>5} {'normal FP':>10} {'anomaly TP':>11}   verdict")
    print("  " + "-" * 48)
    met = []
    for thr in AREA_THRESHOLDS:
        for n in AREA_MIN_PIXELS:
            fp = float((normal_counts[thr] >= n).mean())
            tp = float((anom_counts[thr] >= n).mean())
            ok = fp <= 0.05 and tp >= 0.90
            if ok:
                met.append((tp - fp, thr, n, fp, tp))
            print(f"  {thr:>5.1f} {n:>5d} {fp:>9.1%} {tp:>11.1%}   {'MEETS TARGET' if ok else ''}")
    if met:
        _, thr, n, fp, tp = max(met)
        print(f"  best area trigger: >= {n} px at t={thr} (FP {fp:.1%}, TP {tp:.1%})")
    else:
        best_gap = max(((float((anom_counts[t] >= n).mean()) - float((normal_counts[t] >= n).mean()), t, n)
                        for t in AREA_THRESHOLDS for n in AREA_MIN_PIXELS))
        print(f"  NO area trigger meets both targets either. Widest TP-FP gap: "
              f"{best_gap[0]:.1%} at t={best_gap[1]}, N={best_gap[2]}.")


if __name__ == "__main__":
    main()

"""P0b: does a full-resolution forward pass fit on the 8 GB GPU? Eval-only.

PLAN.md P3 asks whether feeding the encoder 2048x1024 (its Cityscapes
fine-tuning resolution) instead of the 1024x512 the heads were trained at
changes detection. Before spending a 15-30 min run on that, this measures
one Fishyscapes VAL image per scale: peak GPU memory, wall time, the logit
shape and the fused-score range. The decision rule (PLAN.md P0b) reads the
scale-2 RESERVED peak (max_memory_reserved: what the caching allocator holds,
fragmentation included -- the number that decides an OOM on an 8 GB card):
< 6.5 GiB -> whole image; else retry with --window, same 6.5 GiB rule. The
~1.5 GiB margin covers the CUDA context (counted in neither number) and the
second model eval_spatial loads. Allocated peak is printed for reference.

Never touches the test half.

Usage:
    python smoke_fullres.py --raw model_3head_best.pth --scales 1 2
    python smoke_fullres.py --raw model_3head_best.pth --scales 2 --window
"""

import argparse
import time

import torch

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from eval_spatial import forward_logits
from utils import get_device, load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True,
                        help=f"checkpoint, e.g. {config.PRIMARY_RAW} (explicit, MISTAKES.md M15)")
    parser.add_argument("--scales", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    parser.add_argument("--window", action="store_true",
                        help="run scale 2 as two overlapping 1024-tall windows")
    args = parser.parse_args()

    device = get_device()
    if device.type != "cuda":
        raise SystemExit("this measures GPU memory; it needs CUDA")
    model = load_trained_model(args.raw, device)

    fishy_val, fishy_test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    pairs = fishy_val
    assert pairs is fishy_val  # val only: P3's decision must never see the test half
    image_path = pairs[0][0]
    print(f"image (val half): {image_path}")

    for scale in args.scales:
        window = args.window and scale == 2
        size = (config.INPUT_WIDTH * scale, config.INPUT_HEIGHT * scale)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            logits = forward_logits(model, image_path, device, scale, window)
        except torch.cuda.OutOfMemoryError:
            print(f"scale {scale} {size[0]}x{size[1]}{' windowed' if window else ''}: OOM")
            continue
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        fused = torch.sigmoid(logits).mean(dim=0)
        print(f"scale {scale} {size[0]}x{size[1]}{' windowed' if window else ''}: "
              f"reserved peak {reserved:.2f} GiB (rule: < 6.5)  allocated peak {peak:.2f} GiB  time {seconds:.2f}s  "
              f"logit shape {tuple(logits.unsqueeze(0).shape)}  "
              f"fused min/mean/max {fused.min():.4f}/{fused.mean():.4f}/{fused.max():.4f}")


if __name__ == "__main__":
    main()

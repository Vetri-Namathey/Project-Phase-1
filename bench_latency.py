"""Single-pass latency of the deployed model ("yes" item 14). Eval only.

Times the one pass the system actually runs (frozen SegFormer-B5 + seg head + 3
OOD heads, 1024x512, the same amp path as evaluation) on Fishyscapes test
images. cuda.synchronize brackets every timed region, warm-up passes are
discarded, and the GPU name is printed: the number belongs to this GPU, not to
an in-vehicle one (PLAN.md: no deployment claim from laptop timings).

    python bench_latency.py --checkpoint model_3head_best.pth | Tee-Object -FilePath bench_latency.log
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from train import amp_context
from utils import get_device, load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. model_3head_best.pth (rule 11)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    if device.type != "cuda":
        raise SystemExit("latency is only meaningful on the GPU")
    model = load_trained_model(args.checkpoint, device).eval()
    _, test = split_fishyscapes_pairs(list_fishyscapes_pairs())
    images = [load_image_tensor(p, device) for p, _ in test[:10]]
    label_hw = (1024, 2048)

    def sync():
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(x):
        with amp_context(device):
            return model(x)

    @torch.no_grad()
    def full(path):
        x = load_image_tensor(path, device)
        out = forward(x)
        logits = F.interpolate(out["ood_logits"].float(), size=label_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits).mean(dim=1)

    for i in range(args.warmup):
        forward(images[i % len(images)])
    sync()

    fwd = []
    for i in range(args.runs):
        sync()
        t = time.perf_counter()
        forward(images[i % len(images)])
        sync()
        fwd.append(1000 * (time.perf_counter() - t))

    e2e = []
    paths = [p for p, _ in test[:10]]
    for i in range(min(args.runs, 100)):
        sync()
        t = time.perf_counter()
        full(paths[i % len(paths)])
        sync()
        e2e.append(1000 * (time.perf_counter() - t))

    def stats(v):
        v = np.array(v)
        return f"median {np.median(v):7.1f} ms   mean {v.mean():7.1f}   p95 {np.percentile(v, 95):7.1f}   ({1000 / np.median(v):5.1f} fps)"

    print(f"GPU: {torch.cuda.get_device_name(0)}   checkpoint: {args.checkpoint}   input 1024x512, amp")
    print(f"model forward (encoder + seg head + 3 OOD heads), n={len(fwd)}: {stats(fwd)}")
    print(f"end-to-end (disk read + resize + forward + upsample to 2048x1024 + sigmoid + mean), n={len(e2e)}: {stats(e2e)}")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    print("Laptop GPU number, not an in-vehicle deployment figure.")


if __name__ == "__main__":
    main()

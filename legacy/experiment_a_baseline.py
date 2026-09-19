"""SUPERSEDED -- use `python experiment_a.py --image PATH` instead.

Pre-merge original, kept for reference only. `experiment_a.py` (from the
audited exp_coco pipeline) does this single-image check AND the full
evaluation, on the same val/test split and the same metric implementations
Experiment B uses. Nothing imports this except its sibling
`legacy/experiment_a_eval.py`.

Experiment A (Section 12, revised): off-the-shelf SegFormer-B5 baseline.

No training. Loads nvidia/segformer-b5-finetuned-cityscapes-1024-1024 exactly
as published and derives a per-pixel OOD score from its existing 19-class
softmax output via Max Softmax Probability (MSP). This script is for pipeline
validation on a single image -- real evaluation (AUROC/ECE over Fishyscapes)
comes later once that dataset is downloaded.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# This file lives in legacy/, the modules it imports live at the repo root.
# Running a script directly puts the SCRIPT's directory on sys.path, not the
# working directory, so the root has to be added explicitly for
# `python legacy/experiment_a_baseline.py` to resolve `import config`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def load_baseline():
    processor = AutoImageProcessor.from_pretrained(config.BASELINE_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(config.BASELINE_MODEL_NAME)
    model.eval()
    return processor, model


def msp_ood_score(logits):
    """logits: (1, num_classes, H, W) -> ood score (H, W) in [0,1], 1 = anomalous."""
    probs = torch.softmax(logits, dim=1)
    max_prob, _ = probs.max(dim=1)
    return 1.0 - max_prob.squeeze(0)


def run_on_image(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - t0

    logits = outputs.logits  # (1, 19, h, w) -- lower res than input
    logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)

    seg_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    ood_map = msp_ood_score(logits).cpu().numpy()

    print(f"inference time: {elapsed * 1000:.1f} ms on {device}")
    print(f"segmentation map shape: {seg_map.shape}, classes present: {sorted(np.unique(seg_map).tolist())}")
    print(f"OOD score map (MSP): min={ood_map.min():.4f} max={ood_map.max():.4f} mean={ood_map.mean():.4f}")
    return seg_map, ood_map


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/images/anomaly_000.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor, model = load_baseline()
    model.to(device)
    run_on_image(image_path, processor, model, device)

"""SUPERSEDED -- use `python experiment_a.py` instead.

Pre-merge original, kept for reference only. It scores ALL 100 Fishyscapes
images, i.e. it reports on the same images checkpoint selection reads, which
is exactly what `config.FISHYSCAPES_VAL_FRACTION` was introduced to stop.
`experiment_a.py` reports the TEST half only, so its numbers are the ones
comparable with Experiment B's.

Experiment A (Section 12, revised): real Fishyscapes evaluation.

No training. Loads nvidia/segformer-b5-finetuned-cityscapes-1024-1024 and
scores every Fishyscapes Lost & Found validation image with MSP
(1 - max softmax probability), then computes AUROC / ECE / FPR@95 over all
valid (non-ignore) pixels across all 100 images and logs the run to MLflow.
"""

import os
import sys
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# This file lives in legacy/, the modules it imports live at the repo root.
# Running a script directly puts the SCRIPT's directory on sys.path, not the
# working directory, so the root has to be added explicitly for
# `python legacy/experiment_a_eval.py` to resolve `import config`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from data.fishyscapes_dataset import list_fishyscapes_pairs  # noqa: E402
from experiment_a_baseline import load_baseline, msp_ood_score  # noqa: E402
from metrics import compute_auroc, compute_ece, compute_fpr_at_tpr  # noqa: E402


def score_image(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = F.interpolate(outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    return msp_ood_score(logits).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    processor, model = load_baseline()
    model.to(device)

    pairs = list_fishyscapes_pairs()
    assert len(pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(pairs)}"

    all_scores = []
    all_labels = []

    t0 = time.time()
    for i, (image_path, label_path) in enumerate(pairs):
        score_map = score_image(image_path, processor, model, device)
        label_map = np.array(Image.open(label_path))

        valid = label_map != 255
        all_scores.append(score_map[valid])
        all_labels.append(label_map[valid])

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i + 1}/{len(pairs)}] valid_px={valid.sum()} ood_px={(label_map[valid] == 1).sum()}")

    elapsed = time.time() - t0

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    auroc = compute_auroc(scores, labels)
    ece = compute_ece(scores, labels)
    fpr95 = compute_fpr_at_tpr(scores, labels)

    print(f"\ndone in {elapsed:.1f}s over {len(pairs)} images, {len(labels)} valid pixels")
    print(f"AUROC={auroc:.4f}  ECE={ece:.4f}  FPR@95={fpr95:.4f}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="experiment_a_baseline"):
        mlflow.log_params({
            "model": config.BASELINE_MODEL_NAME,
            "ood_score_method": config.OOD_SCORE_METHOD,
            "num_images": len(pairs),
            "device": str(device),
        })
        mlflow.log_metrics({
            "auroc": auroc,
            "ece": ece,
            "fpr95": fpr95,
            "num_valid_pixels": len(labels),
        })

    return auroc, ece, fpr95


if __name__ == "__main__":
    main()

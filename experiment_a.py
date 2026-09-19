"""Experiment A: the off-the-shelf baseline.

No training. Loads nvidia/segformer-b5-finetuned-cityscapes-1024-1024 exactly
as published and derives a per-pixel OOD score from its 19-class softmax via
Max Softmax Probability (MSP).

Reported on the SAME val/test split Experiment B uses, with the SAME metric
implementations, so the two are directly comparable.

    python experiment_a.py                 # full Fishyscapes evaluation
    python experiment_a.py --image PATH    # single-image pipeline check

Note on ECE: here the "confidence" is MSP, which is not a probability that a
pixel is anomalous -- it is one minus the model's confidence in its top
class. Its ECE therefore does not measure the same quantity as Experiment
B's, and the two ECE columns must not be presented as a head-to-head
comparison without saying so.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from metrics import ScoreHistogram
from utils import get_device


def load_baseline(device=None):
    """The published model plus its own preprocessing, which includes
    ImageNet normalisation (do_normalize=True, mean [0.485,0.456,0.406],
    std [0.229,0.224,0.225])."""
    processor = AutoImageProcessor.from_pretrained(config.BASELINE_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(config.BASELINE_MODEL_NAME)
    model.eval()
    if device is not None:
        model.to(device)
    return processor, model


def msp_ood_score(logits):
    """logits (1,C,H,W) -> OOD score (H,W) in [0,1], 1 = anomalous."""
    return 1.0 - torch.softmax(logits, dim=1).max(dim=1).values.squeeze(0)


def predict(image_path, processor, model, device):
    """Returns (seg_map, ood_map) at the image's native resolution.

    The single scoring path -- the evaluation loop and the single-image debug
    mode both go through it, so they cannot drift apart.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = F.interpolate(outputs.logits, size=image.size[::-1],
                           mode="bilinear", align_corners=False)
    return (logits.argmax(dim=1).squeeze(0).cpu().numpy(),
            msp_ood_score(logits).cpu().numpy())


def evaluate_split(pairs, processor, model, device, label):
    hist = ScoreHistogram()
    for i, (image_path, label_path) in enumerate(pairs):
        _, ood_map = predict(image_path, processor, model, device)
        label_map = np.array(Image.open(label_path))
        valid = label_map != 255
        hist.update(ood_map[valid], label_map[valid])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label} {i + 1}/{len(pairs)}]")
    return hist


def run_single_image(image_path, processor, model, device):
    t0 = time.time()
    seg_map, ood_map = predict(image_path, processor, model, device)
    elapsed = (time.time() - t0) * 1000
    print(f"inference: {elapsed:.1f} ms on {device}")
    print(f"segmentation {seg_map.shape}, classes present: "
          f"{sorted(np.unique(seg_map).tolist())}")
    print(f"MSP OOD map: min={ood_map.min():.4f} max={ood_map.max():.4f} "
          f"mean={ood_map.mean():.4f}")
    return seg_map, ood_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="score one image instead of the full set")
    args = parser.parse_args()

    device = get_device()
    processor, model = load_baseline(device)

    if args.image:
        run_single_image(args.image, processor, model, device)
        return

    import mlflow

    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    val_pairs, test_pairs = split_fishyscapes_pairs(all_pairs)
    print(f"split: {len(val_pairs)} val / {len(test_pairs)} test")

    t0 = time.time()
    val_hist = evaluate_split(val_pairs, processor, model, device, "val")
    test_hist = evaluate_split(test_pairs, processor, model, device, "test")
    elapsed = time.time() - t0

    val_metrics = val_hist.summary(prefix="val_")
    test_metrics = test_hist.summary(prefix="test_")
    total_px = test_hist.n_pos + test_hist.n_neg

    print(f"\ndone in {elapsed:.1f}s over {len(all_pairs)} images")
    print(f"  test half: {total_px:,} valid pixels, positive rate "
          f"{test_hist.n_pos / total_px:.4%}")
    print("\nREPORTED (test half):")
    print(f"  AUROC={test_metrics['test_auroc']:.4f}  AP={test_metrics['test_ap']:.4f}  "
          f"FPR@95={test_metrics['test_fpr95']:.4f}  ECE={test_metrics['test_ece']:.4f}")
    print("val half:")
    print(f"  AUROC={val_metrics['val_auroc']:.4f}  AP={val_metrics['val_ap']:.4f}  "
          f"FPR@95={val_metrics['val_fpr95']:.4f}  ECE={val_metrics['val_ece']:.4f}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="experiment_a_baseline"):
        mlflow.log_params({
            "model": config.BASELINE_MODEL_NAME,
            "ood_score_method": config.OOD_SCORE_METHOD,
            "num_images": len(all_pairs),
            "device": str(device),
            "trained": False,
        })
        mlflow.set_tag("ece_note",
                       "Experiment A's ECE is over MSP, not over a probability "
                       "of anomaly -- not directly comparable to Experiment B's.")
        mlflow.log_metrics({
            **val_metrics, **test_metrics,
            "auroc": test_metrics["test_auroc"],
            "ap": test_metrics["test_ap"],
            "ece": test_metrics["test_ece"],
            "fpr95": test_metrics["test_fpr95"],
            "num_valid_pixels": total_px,
        })

    return test_metrics


if __name__ == "__main__":
    main()

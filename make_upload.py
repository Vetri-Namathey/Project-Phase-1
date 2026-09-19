"""Package the code for upload to a RunPod pod.

Builds twinguard_code.zip containing source only -- no datasets, no
checkpoints, no mlruns, no COCO bank. Those are either downloaded on the pod
(COCO, Cityscapes, model weights) or produced by the run itself.

    python make_upload.py
    python make_upload.py --out somewhere/else.zip

Datasets are NOT included on purpose. Cityscapes plus Fishyscapes is roughly
30GB; uploading that over a home connection is far slower than fetching it
on the pod, and RunPod network volumes persist between pods anyway.
"""

import argparse
import os
import zipfile

INCLUDE_FILES = [
    "config.py",
    "utils.py",
    "losses.py",
    "metrics.py",
    "train.py",
    "experiment_a.py",
    "preflight.py",
    "validate_metrics.py",
    "check_collapse.py",
    "check_data.py",
    "check_runs.py",
    "download_coco_anomalies.py",
    "generate_anomalies.py",
    "requirements.txt",
    "requirements-runpod.txt",
    "README.md",
    "RUNPOD_GUIDE.md",
    "pod_survey.sh",
    "setup_env.sh",
]
INCLUDE_DIRS = ["data", "model"]
SKIP_SUFFIXES = (".pyc",)
SKIP_DIRS = {"__pycache__", "coco_objects", "_coco_download", "images", "masks",
             "_excluded_objects", "test_images", "test_masks"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="twinguard_code.zip")
    args = parser.parse_args()

    added = []
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE_FILES:
            if os.path.exists(name):
                zf.write(name)
                added.append(name)
            else:
                print(f"  WARNING: {name} not found, skipping")

        for directory in INCLUDE_DIRS:
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in files:
                    if fname.endswith(SKIP_SUFFIXES):
                        continue
                    path = os.path.join(root, fname)
                    zf.write(path)
                    added.append(path)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.2f} MB, {len(added)} files)")
    print("\nDeliberately excluded -- fetched or produced on the pod:")
    print("  data/coco_objects/   download_coco_anomalies.py rebuilds it")
    print("  data/images|masks/   CARLA output -- copy these to the pod yourself if")
    print("                       ANOMALY_SOURCE is 'carla' or 'both' (the default)")
    print("  data/_excluded_objects/  the 5 curated-out CARLA props, never trained on")
    print("  checkpoints/         produced by train.py")
    print("  mlruns/              produced by train.py")
    print("  Cityscapes/Fishyscapes -- see the RunPod guide for fetching these")


if __name__ == "__main__":
    main()

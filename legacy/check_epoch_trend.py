"""SUPERSEDED -- use `python check_runs.py --trend` instead.

Pre-merge original, kept for reference only. It hardcodes the run name
`experiment_b_3head`; `check_runs.py --trend` walks the latest run whatever
it is called, and prints AP alongside AUROC.
"""

import os
import sys

import mlflow

# This file lives in legacy/, config.py lives at the repo root. Running a
# script directly puts the SCRIPT's directory on sys.path, not the working
# directory, so the root has to be added explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
runs = client.search_runs(
    experiment.experiment_id,
    filter_string="tags.mlflow.runName = 'experiment_b_3head'",
    order_by=["start_time DESC"],
)

latest = runs[0]
print(f"run_id: {latest.info.run_id}  status: {latest.info.status}")

for key in ["auroc", "ece", "fpr95", "train_loss", "auroc_head0", "auroc_head1", "auroc_head2"]:
    history = client.get_metric_history(latest.info.run_id, key)
    if not history:
        print(f"{key}: no history")
        continue
    values = ", ".join(f"{m.step}:{m.value:.4f}" for m in sorted(history, key=lambda m: m.step))
    print(f"{key}: {values}")

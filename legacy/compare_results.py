"""SUPERSEDED -- use `python check_runs.py` instead.

Pre-merge original, kept for reference only. `check_runs.py` prints the same
side-by-side run table plus the per-epoch trend (`--trend`, which replaces
`legacy/check_epoch_trend.py`), and it reports the AP / disagreement-AUROC
columns and the val-vs-test split that this version predates.
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
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])

for run in runs:
    name = run.data.tags.get("mlflow.runName", "?")
    m = run.data.metrics
    print(f"{name:<25} auroc={m.get('auroc')}  ece={m.get('ece')}  fpr95={m.get('fpr95')}")

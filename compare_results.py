import mlflow

import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])

for run in runs:
    name = run.data.tags.get("mlflow.runName", "?")
    m = run.data.metrics
    print(f"{name:<25} auroc={m.get('auroc')}  ece={m.get('ece')}  fpr95={m.get('fpr95')}")

import mlflow

import config

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

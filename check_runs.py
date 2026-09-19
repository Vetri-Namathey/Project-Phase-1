"""Read results out of MLflow.

    python check_runs.py            # every run, side by side
    python check_runs.py --trend    # per-epoch history for the latest run

Reported metrics are the TEST-half numbers. Checkpoints are selected on the
val half, shown separately so selection bias is visible rather than implicit.
"""

import argparse

import config
from utils import mlflow_experiment


def show_all_runs(client, experiment):
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    if not runs:
        raise SystemExit("no runs logged yet")

    header = (f"{'run':<22} {'src':<5} {'ep':>3} {'AUROC':>7} {'AP':>7} "
              f"{'FPR@95':>7} {'ECE':>7} {'mIoU':>6} {'disagAUC':>9}")
    print(header)
    print("-" * len(header))

    def fmt(value, width=7, places=4):
        return (f"{value:>{width}.{places}f}" if isinstance(value, (int, float))
                else f"{'--':>{width}}")

    for run in runs:
        name = run.data.tags.get("mlflow.runName", "?")
        m = run.data.metrics
        # Prefer the SELECTED checkpoint's numbers. Falling back to the bare
        # keys would report whatever the last epoch scored, which describes a
        # different model from the .pth file that was actually saved.
        def pick(key):
            return m.get(f"selected_{key}", m.get(key))
        epoch = m.get("best_epoch")
        epoch_s = f"{int(epoch) + 1:>3}" if isinstance(epoch, (int, float)) else "  -"
        print(f"{name[:22]:<22} {run.data.params.get('anomaly_source', '--')[:5]:<5} "
              f"{epoch_s} {fmt(pick('auroc'))} {fmt(pick('ap'))} "
              f"{fmt(pick('fpr95'))} {fmt(pick('ece'))} {fmt(m.get('miou'), 6)} "
              f"{fmt(pick('auroc_disagreement'), 9)}")

    print()
    print("  ep       = epoch of the SELECTED checkpoint (blank for runs with no selection)")
    print("  disagAUC = AUROC of head disagreement alone -- the ensemble")
    print("             uncertainty signal scored as a detector in its own right")


def show_trend(client, experiment):
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.mlflow.runName = 'experiment_b_3head'",
        order_by=["start_time DESC"],
    )
    if not runs:
        raise SystemExit("no experiment_b_3head runs found")

    latest = runs[0]
    print(f"run_id: {latest.info.run_id}  status: {latest.info.status}")
    print(f"encoder: {latest.data.params.get('encoder')}  "
          f"source: {latest.data.params.get('anomaly_source')}\n")

    # test_-prefixed, because the bare aliases only cover a few metrics --
    # an earlier version queried the bare names and printed "--" for the
    # disagreement AUROC, which is the project's core uncertainty claim.
    keys = ["train_loss", "train_l_ood", "val_ap", "val_auroc", "auroc", "ap",
            "fpr95", "ece", "miou", "test_auroc_disagreement",
            "test_min_head_separation"]
    history = {k: {m.step: m.value for m in
                   client.get_metric_history(latest.info.run_id, k)} for k in keys}

    steps = sorted({s for h in history.values() for s in h})
    if not steps:
        raise SystemExit("run has no logged metric history yet")

    short = {"test_auroc_disagreement": "disagAUC",
             "test_min_head_separation": "minSep"}
    header = f"{'ep':>3} " + " ".join(
        f"{short.get(k, k.replace('train_', '')):>10}" for k in keys)
    print(header)
    print("-" * len(header))
    for step in steps:
        row = f"{step + 1:>3} "
        for key in keys:
            value = history[key].get(step)
            row += f" {value:>10.4f}" if value is not None else f" {'--':>10}"
        print(row)

    # Collapse is "the heads stopped telling the classes apart", which needs
    # BOTH a flat separation and a near-chance AUROC. Output std is not the
    # test -- a healthy model at a 0.24% positive rate has a low std by
    # construction. Read the last epoch that actually has test numbers, since
    # the test half is only scored when val improves.
    def last_value(key):
        for step in reversed(steps):
            value = history.get(key, {}).get(step)
            if value is not None:
                return value
        return None

    final_sep = last_value("test_min_head_separation")
    final_auroc = last_value("auroc")
    if (final_sep is not None and final_sep < 0.01
            and final_auroc is not None and final_auroc < 0.70):
        print(f"\nWARNING: head separation {final_sep:.4f} with AUROC "
              f"{final_auroc:.4f} -- the heads are not telling anomalous from "
              f"normal pixels apart. See check_collapse.py.")
    elif final_sep is not None:
        print(f"\nhead separation {final_sep:.4f}, AUROC {final_auroc:.4f} "
              f"-- healthy, no collapse.")
    print("\nBlank test columns are epochs where val did not improve, so the "
          "test half was not scored (EVAL_TEST_ON_IMPROVEMENT_ONLY).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trend", action="store_true",
                        help="per-epoch history for the latest training run")
    args = parser.parse_args()

    client, experiment = mlflow_experiment()
    if args.trend:
        show_trend(client, experiment)
    else:
        show_all_runs(client, experiment)
        print()
        print(f"gates: {config.PRECALIBRATION_AUROC_GATE} pre-calibration "
              f"(proceed to Phase 2b), {config.POSTCALIBRATION_AUROC_TARGET} "
              f"post-calibration target")
        print(config.CALIBRATION_TRADEOFF_NOTE)


if __name__ == "__main__":
    main()

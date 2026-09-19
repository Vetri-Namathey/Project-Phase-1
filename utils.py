"""Small shared helpers.

Everything here existed as copy-pasted blocks in three or four scripts
before. Kept deliberately thin -- this is for boilerplate that must behave
identically everywhere, not a dumping ground.
"""

import torch

import config


def get_device(verbose=True):
    """Selects CUDA when available and reports what was chosen."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"device: {device} ({torch.cuda.get_device_name(0)}, {total:.0f}GB)")
        else:
            print("device: cpu (no CUDA available)")
    return device


def mlflow_experiment():
    """Returns (client, experiment) for the configured tracking store.

    Fails with a useful message instead of an AttributeError on None, which
    is what happens when the script is run from the wrong directory -- the
    tracking URI is a relative file path.
    """
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(
            f"no MLflow experiment named {config.MLFLOW_EXPERIMENT_NAME!r} at "
            f"{config.MLFLOW_TRACKING_URI}. Run training first, and run this "
            f"from the repository root (the tracking URI is a relative path)."
        )
    return client, experiment


def load_trained_model(checkpoint_path=None, device=None, num_heads=3):
    """Builds TwinGuard and loads a saved checkpoint into it."""
    from model.twinguard_model import TwinGuardModel

    checkpoint_path = checkpoint_path or config.CHECKPOINT_3HEAD
    device = device or get_device(verbose=False)
    seeds = (config.OOD_HEAD_SEEDS_3HEAD if num_heads == 3
             else config.OOD_HEAD_SEEDS_1HEAD)

    model = TwinGuardModel(num_ood_heads=num_heads, ood_seeds=seeds).to(device)
    try:
        # weights_only=True: the checkpoint is a plain state_dict, and the
        # unrestricted default emits a FutureWarning and is a real risk on
        # any file you did not produce yourself.
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise SystemExit(
            f"no checkpoint at {checkpoint_path} -- run train.py first."
        )
    model.load_state_dict(state)
    model.eval()
    return model

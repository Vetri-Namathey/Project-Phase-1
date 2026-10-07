"""Phase 2b: L_calib, and the temperature-scaling baseline it has to beat.

Three modes. Exactly one must be named, so no flag combination trains by
accident: a bare run used to start an L_calib fine-tune on the stale
config.CHECKPOINT_3HEAD (MISTAKES.md M15).

  --compare-only --raw R --calib C [--temp-whole T]
      Eval only. Fishyscapes TEST-half table: raw vs temp(whole) vs L_calib,
      whole-image metrics plus band-ECE r=8, and a NEW reliability png. T is
      fitted on the val half (plain NLL) unless --temp-whole passes the T
      that eval_spatial.py already fitted (PLAN.md U2-3).
  --temp-only --checkpoint R
      Eval only. Fits whole + band T on the val half, reports the test half.
  --train-lcalib --checkpoint R
      The only route into run_calib_finetune. Writes
      config.CHECKPOINT_3HEAD_CALIB, a training-output path that is never the
      primary model file.

Temperature fit: plain binary NLL with a bounded 1-D search over log T,
ported from exp_v2. The previous fit minimised the TRAINING loss
(pos_weight=20) for 3 Adam epochs and stopped while T was still rising; its
T=1.7142 is invalid (MISTAKES.md M1). It survives only as
_fit_temperature_weighted_DEPRECATED so that number can be reproduced; no
flag reaches it.

eval_spatial.py adds r=4/8/16 bands, UBQ and paired bootstrap CIs on top.

Usage:
    python calibrate.py --compare-only --raw model_3head_best.pth --calib model_3head_calib_best.pth --temp-whole <T>
    python calibrate.py --temp-only --checkpoint model_3head_best.pth
    python calibrate.py --train-lcalib --checkpoint model_3head_best.pth
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

import config
from data.cityscapes_dataset import CityscapesDataset
from data.cutmix import CutMixAugmentedDataset
from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
from data.transforms import load_image_tensor
from losses import SoftECELoss, build_seg_criterion, compute_total_loss, ood_bce_loss
from metrics import ScoreHistogram, boundary_band
from train import amp_context
from utils import get_device, load_trained_model

REGIONS = ("whole", "band")
# The png calibrate.py used to write. It holds the invalid T=1.7142 curve and
# stays on disk as the record of M1 (no-delete rule), so nothing overwrites it.
LEGACY_PNG = "calibration_reliability.png"


def _cache_logits(model, pairs, device):
    """One forward pass per image, cached -- reused across every temperature
    -fit epoch without re-running the (frozen) encoder each time. Used only
    by the deprecated weighted fit."""
    cached = []
    model.eval()
    with torch.no_grad():
        for image_path, label_path in pairs:
            label_map = np.array(Image.open(label_path))
            valid = label_map != 255
            with amp_context(device):
                out = model(load_image_tensor(image_path, device))
            logits = F.interpolate(out["ood_logits"].float(), size=label_map.shape,
                                   mode="bilinear", align_corners=False)
            num_heads = logits.shape[1]
            logits_flat = logits.squeeze(0).reshape(num_heads, -1).cpu()
            valid_flat = torch.from_numpy(valid.reshape(-1))
            target = torch.from_numpy((label_map[valid] == 1).astype(np.float32))
            cached.append((logits_flat[:, valid_flat], target))
    return cached


def _fit_temperature_weighted_DEPRECATED(model, val_pairs, device):
    """DEPRECATED -- produced the invalid T=1.7142 (MISTAKES.md M1). Do not use.

    Minimises the training loss (ood_bce_loss, pos_weight=20) for
    config.TEMPERATURE_EPOCHS Adam epochs from T=1.5. pos_weight=20 rewards
    higher scores on positives, so the optimum sits above the NLL optimum,
    and 3 epochs stop before convergence (calibrate.log: 1.5688 -> 1.6421 ->
    1.7142, loss still falling). Kept only so that number can be reproduced;
    no command-line flag reaches it. Use fit_temperatures.
    """
    print("fitting temperature with the DEPRECATED weighted objective (M1)...")
    cached = _cache_logits(model, val_pairs, device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.Adam([temperature], lr=config.TEMPERATURE_LEARNING_RATE)

    for epoch in range(config.TEMPERATURE_EPOCHS):
        total_loss = 0.0
        for logits_valid, target in cached:
            logits_valid = logits_valid.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            num_heads = logits_valid.shape[0]
            loss = torch.stack([
                ood_bce_loss(logits_valid[h] / temperature, target)
                for h in range(num_heads)
            ]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{config.TEMPERATURE_EPOCHS}: "
              f"T={temperature.item():.4f}  loss={total_loss / len(cached):.4f}")
    return float(temperature.detach().item())


def temperature_cache_entry(logits, anomaly, valid, radius=config.TEMPERATURE_BAND_RADIUS_PX):
    """(heads, H, W) logits at label resolution -> one entry of the
    temperature-fit cache: per-head logits over valid pixels, their targets,
    and which of them fall in the r-band. Shared with eval_spatial.py so both
    scripts fit T on exactly the same pixels."""
    band = boundary_band(anomaly, radius_px=radius, valid=valid)
    valid_t = torch.from_numpy(valid)
    return (logits[:, valid_t].cpu(),
            torch.from_numpy(anomaly[valid].astype(np.float32)),
            torch.from_numpy(band[valid]))


@torch.no_grad()
def _cache_val_logits(model, pairs, device):
    """One forward pass per val image, so every temperature the 1-D search
    tries is scored without re-running the model."""
    cached = []
    model.eval()
    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid, anomaly = label_map != 255, label_map == 1
        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=anomaly.shape,
                               mode="bilinear", align_corners=False).squeeze(0).cpu()
        cached.append(temperature_cache_entry(logits, anomaly, valid))
    return cached


def _mean_nll(cached, temperature, region):
    """Plain (unweighted) binary NLL of sigmoid(logit / T), averaged over
    every head and every pixel in the region -- i.e. T is applied per head,
    before the fusion mean, exactly as the scores are later computed."""
    total, count = 0.0, 0
    for logits, target, band in cached:
        if region == "band":
            logits, target = logits[:, band], target[band]
        if target.numel() == 0:
            continue
        total += F.binary_cross_entropy_with_logits(
            logits / temperature, target.expand_as(logits), reduction="sum").item()
        count += logits.numel()
    return total / count


def fit_temperatures_from_cache(cached, verbose=True):
    """Fits T separately for each region in REGIONS. Returns {region: T}.

    Plain NLL, NOT the training loss: the training loss carries
    pos_weight=20, which rewards pushing scores UP (MISTAKES.md M1). A
    bounded scalar search is deterministic, so the same logits always give
    the same T.
    """
    lo, hi = config.TEMPERATURE_LOG_BOUNDS
    temperatures = {}
    for region in REGIONS:
        res = minimize_scalar(lambda log_t: _mean_nll(cached, float(np.exp(log_t)), region),
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        t = float(np.exp(res.x))
        if abs(res.x - lo) < 1e-3 or abs(res.x - hi) < 1e-3:
            print(f"  WARNING: {region} T={t:.4f} hit the search bound -- widen "
                  f"config.TEMPERATURE_LOG_BOUNDS")
        if verbose:
            print(f"  {region:<5}  T={t:.4f}  NLL {_mean_nll(cached, 1.0, region):.6f} "
                  f"(T=1) -> {res.fun:.6f}  [{res.nfev} evaluations]")
        temperatures[region] = t
    return temperatures


def fit_temperatures(model, val_pairs, device):
    print("fitting temperature scaling baselines (plain NLL, Fishyscapes val half)...")
    return fit_temperatures_from_cache(_cache_val_logits(model, val_pairs, device))


@torch.no_grad()
def evaluate_fused(model, pairs, device, temperature=1.0):
    """Fused-score AUROC/AP/FPR@95/ECE on a Fishyscapes split, with an
    optional post-hoc temperature applied to each head's logit BEFORE fusing
    -- the same mean-of-sigmoids fusion the model itself uses, just fed
    logit/T instead of the raw logit when temperature != 1.0.
    """
    model.eval()
    hist = ScoreHistogram()
    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid = label_map != 255
        labels = label_map[valid]

        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=label_map.shape,
                               mode="bilinear", align_corners=False)
        scores = torch.sigmoid(logits / temperature).mean(dim=1)
        hist.update(scores.squeeze(0).cpu().numpy()[valid], labels)
    return hist


@torch.no_grad()
def evaluate_regions(model, pairs, device, temperature=1.0,
                     radius=config.TEMPERATURE_BAND_RADIUS_PX):
    """evaluate_fused, plus a second histogram over the r-band, so the
    comparison table carries the edge number the paper is about."""
    model.eval()
    hists = {region: ScoreHistogram() for region in REGIONS}
    for image_path, label_path in pairs:
        label_map = np.array(Image.open(label_path))
        valid, anomaly = label_map != 255, label_map == 1
        band = boundary_band(anomaly, radius_px=radius, valid=valid)
        with amp_context(device):
            out = model(load_image_tensor(image_path, device))
        logits = F.interpolate(out["ood_logits"].float(), size=label_map.shape,
                               mode="bilinear", align_corners=False)
        scores = torch.sigmoid(logits / temperature).mean(dim=1).squeeze(0).cpu().numpy()
        labels = anomaly.astype(np.int64)
        hists["whole"].update(scores[valid], labels[valid])
        hists["band"].update(scores[band], labels[band])
    return hists


def run_calib_finetune(base_checkpoint, device, beta=config.BETA_CALIB, out_path=config.CHECKPOINT_3HEAD_CALIB):
    """Continues training an already-converged checkpoint with SoftECELoss
    added to the existing loss. Selection reads val ECE (what this fine-tune
    actually exists to fix), guarded by the AUROC-drop limit -- restart, do
    not save, if the drop exceeds what CALIBRATION_TRADEOFF_NOTE accounts for.
    """
    model = load_trained_model(base_checkpoint, device)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)  # frozen invariant holds through Phase 2b too

    train_base = CityscapesDataset(split="train", normalize=False)
    train_dataset = CutMixAugmentedDataset(train_base, p=config.CUTMIX_PROB)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
        pin_memory=device.type == "cuda")
    print(f"calib dataloader: {len(train_loader)} steps/epoch, "
          f"num_workers={config.NUM_WORKERS}")

    all_pairs = list_fishyscapes_pairs()
    fishy_val, _ = split_fishyscapes_pairs(all_pairs)

    seg_criterion = build_seg_criterion().to(device)
    soft_ece = SoftECELoss().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.seg_head.parameters(), "lr": config.CALIB_LEARNING_RATE},
        {"params": model.ood_heads.parameters(), "lr": config.CALIB_LEARNING_RATE},
    ])

    start_hist = evaluate_fused(model, fishy_val, device)
    start_auroc = start_hist.auroc()
    print(f"base checkpoint val AUROC: {start_auroc:.4f} (fine-tune stops if "
          f"this drops by more than {config.CALIB_AUROC_DROP_LIMIT})")

    best_score, best_epoch = -1.0, -1
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.CALIB_EPOCHS):
        model.train()
        model.encoder.eval()
        running_total, running_calib = 0.0, 0.0
        n_steps = len(train_loader)

        for step, (images, seg_labels, ood_target) in enumerate(train_loader):
            images = images.to(device)
            seg_labels = seg_labels.to(device)
            ood_target = ood_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                out = model(images)
                base_loss, l_seg, l_ood = compute_total_loss(
                    out["seg_logits"].float(), seg_labels,
                    out["ood_logits"].float(), ood_target, seg_criterion)
                l_calib = soft_ece(out["ood_fused"].float(), ood_target)
                loss = base_loss + beta * l_calib
            loss.backward()
            optimizer.step()

            running_total += loss.item()
            running_calib += l_calib.item()
            if step % 100 == 0:
                print(f"  calib epoch {epoch + 1}/{config.CALIB_EPOCHS} "
                      f"step {step}/{n_steps} loss={loss.item():.4f} "
                      f"(base={base_loss.item():.4f} calib={l_calib.item():.4f})")

        val_hist = evaluate_fused(model, fishy_val, device)
        val_auroc, val_ece = val_hist.auroc(), val_hist.ece()
        auroc_drop = start_auroc - val_auroc
        print(f"calib epoch {epoch + 1}/{config.CALIB_EPOCHS} done -- "
              f"val AUROC={val_auroc:.4f} (drop {auroc_drop:+.4f}) "
              f"val ECE={val_ece:.4f} avg_calib_loss={running_calib / n_steps:.4f}")

        if auroc_drop > config.CALIB_AUROC_DROP_LIMIT:
            print(f"  STOPPING: AUROC dropped {auroc_drop:.4f} > "
                  f"{config.CALIB_AUROC_DROP_LIMIT} limit -- per "
                  f"CALIBRATION_TRADEOFF_NOTE this is no longer an "
                  f"accounted-for cost. This epoch's weights are NOT saved.")
            break

        score = -val_ece
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), out_path)
            print(f"  -> new best val ECE {val_ece:.4f}, checkpoint saved")

    if best_epoch == -1:
        raise SystemExit(
            "no epoch improved val ECE within the AUROC-drop budget -- no "
            "calibrated checkpoint was saved. See the CALIB_* settings in "
            "config.py (BETA_CALIB, CALIB_EPOCHS, CALIB_AUROC_DROP_LIMIT)."
        )
    print(f"L_calib fine-tune done (beta={beta}). selected epoch {best_epoch + 1} -> {out_path}")
    return out_path


def comparison_table(rows):
    """rows: [(name, {"whole": hist, "band": hist})], all on the test half."""
    r = config.TEMPERATURE_BAND_RADIUS_PX
    header = (f"{'model':<20} {'AUROC':>7} {'AP':>7} {'FPR@95':>7} "
              f"{'ECE':>8} {f'band-ECE r={r}':>14}")
    print("\n" + header)
    print("-" * len(header))
    results = {}
    for name, hists in rows:
        m = hists["whole"].summary()
        m["band_ece"] = hists["band"].ece()
        print(f"{name:<20} {m['auroc']:>7.4f} {m['ap']:>7.4f} {m['fpr95']:>7.4f} "
              f"{m['ece']:>8.5f} {m['band_ece']:>14.4f}")
        results[name] = m
    print("Whole-image ECE is dominated by easy background pixels; the band column "
          "is the edge number. For r=4/8/16, UBQ and confidence intervals run "
          "eval_spatial.py.")
    return results


def reliability_diagram(hists, out_path):
    """Whole-image and band panels, one curve per model. Built from the same
    per-bin data ScoreHistogram.ece() uses, so the diagram and the reported
    ECE numbers can never silently disagree."""
    if os.path.basename(out_path) == LEGACY_PNG:
        raise SystemExit(f"refusing to overwrite {LEGACY_PNG}: it is the T=1.7142 record "
                         f"(MISTAKES.md M1). Pass a different --png.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, region in zip(axes, REGIONS):
        ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
        for name, per_region in hists.items():
            conf, acc, weight = per_region[region].reliability_curve()
            mask = weight > 0
            ax.plot(conf[mask], acc[mask], marker="o", label=name)
        title = ("whole image" if region == "whole"
                 else f"boundary band r={config.TEMPERATURE_BAND_RADIUS_PX}px")
        ax.set_title(f"Reliability -- {title} (Fishyscapes test half)")
        ax.set_xlabel("mean predicted score (confidence)")
        ax.set_ylabel("empirical positive rate (accuracy)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def _fishyscapes_split():
    all_pairs = list_fishyscapes_pairs()
    assert len(all_pairs) == 100, f"expected 100 Fishyscapes pairs, found {len(all_pairs)}"
    return split_fishyscapes_pairs(all_pairs)


def run_compare_only(args):
    """Eval only: never constructs an optimiser or touches a checkpoint file
    for writing."""
    device = get_device()
    raw_model = load_trained_model(args.raw, device)
    calib_model = load_trained_model(args.calib, device)
    fishy_val, fishy_test = _fishyscapes_split()

    if args.temp_whole is not None:
        temperature = args.temp_whole
        source = "--temp-whole (fitted by eval_spatial.py on the val half)"
    else:
        temperature = fit_temperatures(raw_model, fishy_val, device)["whole"]
        source = "fitted here on the val half (plain NLL)"
    print(f"temp(whole) T = {temperature:.4f}  [{source}]")

    rows = [("raw", evaluate_regions(raw_model, fishy_test, device, 1.0)),
            (f"temp(whole) T={temperature:.4f}",
             evaluate_regions(raw_model, fishy_test, device, temperature)),
            ("L_calib", evaluate_regions(calib_model, fishy_test, device, 1.0))]
    comparison_table(rows)
    reliability_diagram(dict(rows), args.png)


def run_temp_only(args):
    device = get_device()
    raw_model = load_trained_model(args.checkpoint, device)
    fishy_val, fishy_test = _fishyscapes_split()
    temperatures = fit_temperatures(raw_model, fishy_val, device)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="temperature_scaling_only"):
        mlflow.log_params({"base_checkpoint": args.checkpoint,
                           "temperature_whole": temperatures["whole"],
                           "temperature_band": temperatures["band"]})
        rows = [("raw", evaluate_regions(raw_model, fishy_test, device, 1.0))]
        for region in REGIONS:
            rows.append((f"temp({region}) T={temperatures[region]:.4f}",
                         evaluate_regions(raw_model, fishy_test, device, temperatures[region])))
        for name, m in comparison_table(rows).items():
            key = name.split(" ")[0].replace("(", "_").replace(")", "")
            mlflow.log_metrics({f"test_{key}_{k}": v for k, v in m.items()})


def run_train_lcalib(args):
    # Belt and braces for M15: the fine-tune writes CHECKPOINT_3HEAD_CALIB,
    # which must never resolve to a primary model file.
    out = os.path.abspath(args.out)
    assert out not in (os.path.abspath(config.PRIMARY_RAW), os.path.abspath(config.PRIMARY_CALIB)), \
        f"--out points at a primary model file ({out}) -- see M15"
    if os.path.exists(out):
        raise SystemExit(f"--out {out} already exists; refusing to overwrite a checkpoint (pick a new name)")
    device = get_device()
    raw_model = load_trained_model(args.checkpoint, device)
    fishy_val, fishy_test = _fishyscapes_split()
    temperature = fit_temperatures(raw_model, fishy_val, device)["whole"]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="l_calib_finetune"):
        mlflow.log_params({
            "base_checkpoint": args.checkpoint,
            "beta_calib": args.beta,
            "calib_epochs": config.CALIB_EPOCHS,
            "calib_lr": config.CALIB_LEARNING_RATE,
            "auroc_drop_limit": config.CALIB_AUROC_DROP_LIMIT,
            "fitted_temperature": temperature,
        })
        calib_checkpoint = run_calib_finetune(args.checkpoint, device, beta=args.beta, out_path=out)
        calib_model = load_trained_model(calib_checkpoint, device)
        rows = [("raw", evaluate_regions(raw_model, fishy_test, device, 1.0)),
                (f"temp(whole) T={temperature:.4f}",
                 evaluate_regions(raw_model, fishy_test, device, temperature)),
                ("L_calib", evaluate_regions(calib_model, fishy_test, device, 1.0))]
        for name, m in comparison_table(rows).items():
            key = name.split(" ")[0].replace("(", "_").replace(")", "")
            mlflow.log_metrics({f"{key}_{k}": v for k, v in m.items()})
        reliability_diagram(dict(rows), args.png)
        mlflow.log_artifact(args.png)
        mlflow.log_artifact(calib_checkpoint)


def build_parser():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-only", action="store_true",
                      help="eval only: test-half table raw / temp(whole) / L_calib + new png")
    mode.add_argument("--temp-only", action="store_true",
                      help="eval only: fit whole + band T (plain NLL) on val, report test")
    mode.add_argument("--train-lcalib", action="store_true",
                      help="TRAINING: L_calib fine-tune of --checkpoint into "
                           "config.CHECKPOINT_3HEAD_CALIB")
    parser.add_argument("--raw", help=f"--compare-only: raw checkpoint, e.g. {config.PRIMARY_RAW}")
    parser.add_argument("--calib", help=f"--compare-only: L_calib checkpoint, e.g. {config.PRIMARY_CALIB}")
    parser.add_argument("--checkpoint",
                        help="--temp-only / --train-lcalib: base checkpoint (explicit, M15)")
    parser.add_argument("--temp-whole", type=float, default=None,
                        help="--compare-only: use this whole-image T instead of refitting")
    parser.add_argument("--beta", type=float, default=config.BETA_CALIB,
                        help="--train-lcalib: weight on L_calib (PLAN.md beta sweep)")
    parser.add_argument("--out", default=config.CHECKPOINT_3HEAD_CALIB,
                        help="--train-lcalib: output checkpoint; must not exist yet")
    parser.add_argument("--png", default="calibration_reliability_T122.png",
                        help=f"reliability diagram output (never {LEGACY_PNG})")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.compare_only or args.temp_only or args.train_lcalib):
        parser.error("no mode given. The training path is disabled by default (MISTAKES.md "
                     "M15): pass exactly one of --compare-only, --temp-only, --train-lcalib")
    if args.compare_only and not (args.raw and args.calib):
        parser.error("--compare-only requires --raw and --calib")
    if (args.temp_only or args.train_lcalib) and not args.checkpoint:
        parser.error("--temp-only / --train-lcalib require --checkpoint")
    if args.temp_whole is not None and not args.compare_only:
        parser.error("--temp-whole is only used by --compare-only")
    if args.train_lcalib and args.png == parser.get_default("png"):
        # never overwrite the signed T122 diagram from a sweep run
        args.png = f"calibration_reliability_lcalib_beta{args.beta:g}.png"
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.compare_only:
        run_compare_only(args)
    elif args.temp_only:
        run_temp_only(args)
    else:
        run_train_lcalib(args)


if __name__ == "__main__":
    main()

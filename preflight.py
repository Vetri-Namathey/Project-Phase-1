"""End-to-end preflight. Run this before paying for GPU time.

Checks every assumption training depends on, in dependency order, and stops
at the first real failure with a specific instruction. Each of these has
already been an actual bug in this project at least once.

    python preflight.py
"""

import os
import sys
import traceback

import numpy as np

import config

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, fn, fatal=True):
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 -- preflight reports, never raises
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        if os.environ.get("PREFLIGHT_TRACE"):
            traceback.print_exc()
    results.append((name, status, detail, fatal))
    marker = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN"}[status]
    print(f"[{marker}] {name}\n       {detail}")
    return status


def c_versions():
    """Catch library combinations that break in a misleading way.

    transformers 5.x requires torch>=2.5. On older torch it does not raise at
    import -- it prints a notice, disables its PyTorch backend, and keeps
    going. The first symptom is "SegformerModel requires the PyTorch library
    but it was not found" from somewhere deep in model construction, which
    reads like torch is missing when torch is fine and transformers is the
    problem. Checking it here turns three cascading ImportErrors into one
    line naming the fix.
    """
    try:
        import torch
    except ImportError as exc:
        if "cannot open shared object file" in str(exc) or "libcudnn" in str(exc):
            return FAIL, (
                f"torch imports but cannot load a CUDA shared library:\n"
                f"         {exc}\n"
                f"       This is the classic broken venv: a torch was pip-installed "
                f"into a\n"
                f"       --system-site-packages venv, and pip skipped some CUDA deps as\n"
                f"       'already satisfied' from the system. The venv's own nvidia/ "
                f"directory\n"
                f"       then shadows the system one, hiding the libraries pip skipped.\n"
                f"       Cleanest fix is to rebuild the venv rather than patch it:\n"
                f"         deactivate && rm -rf .venv\n"
                f"         python -m venv --system-site-packages .venv\n"
                f"         source .venv/bin/activate && pip install -r requirements-runpod.txt"
            )
        raise

    import transformers
    from transformers.utils import is_torch_available

    tv, trv = torch.__version__, transformers.__version__

    if not is_torch_available():
        return FAIL, (
            f"transformers {trv} cannot see torch {tv}.\n"
            f"       transformers 5.x needs torch>=2.5; this torch is older, so "
            f"the PyTorch backend is disabled.\n"
            f"       Fix (inside your venv):  pip install 'transformers>=4.46.3,<5'\n"
            f"       Do NOT upgrade torch on a pod -- it is the CUDA-matched build."
        )

    torch_v = tuple(int(x) for x in tv.split("+")[0].split(".")[:2])
    tr_v = tuple(int(x) for x in trv.split(".")[:2])

    if tr_v >= (5, 0) and torch_v < (2, 5):
        return FAIL, (
            f"transformers {trv} requires torch>=2.5 but torch is {tv}.\n"
            f"       Fix (inside your venv):  pip install 'transformers>=4.46.3,<5'"
        )

    # transformers 4.51+ refuses to call torch.load on torch<2.6 (CVE-2025-
    # 32434) and tells you to use safetensors instead. That advice does not
    # apply here: nvidia/segformer-* ships pytorch_model.bin ONLY, with no
    # safetensors file in the repo, so there is nothing to switch to. The
    # combination is simply unusable and has to be resolved by version.
    if tr_v >= (4, 51) and torch_v < (2, 6):
        return FAIL, (
            f"torch {tv} + transformers {trv} cannot load the encoder.\n"
            f"       transformers 4.51+ blocks torch.load on torch<2.6 (CVE-2025-32434),\n"
            f"       and nvidia/segformer-* ships only pytorch_model.bin -- there is no\n"
            f"       safetensors file to fall back to.\n"
            f"       Fix, inside your venv:\n"
            f"         pip install 'transformers>=4.46.3,<4.51'\n"
            f"       Do NOT fix this by upgrading torch. Installing torch into a\n"
            f"       --system-site-packages venv leaves it unable to find cuDNN:\n"
            f"       pip skips CUDA deps it sees in system site-packages, and the\n"
            f"       venv's own nvidia/ directory then shadows them."
        )

    return PASS, f"torch {tv}, transformers {trv}, numpy {np.__version__}"


def c_paths():
    names = ("CITYSCAPES_IMAGES_ROOT", "CITYSCAPES_LABELS_ROOT",
             "FISHYSCAPES_LABELS_DIR", "FISHYSCAPES_IMAGES_ROOT")
    missing = [n for n in names if not os.path.isdir(getattr(config, n))]
    if missing:
        detail = "\n".join(f"         {n} -> {getattr(config, n)}" for n in missing)
        if os.environ.get("TWINGUARD_DATA_ROOT"):
            hint = ("\n       TWINGUARD_DATA_ROOT is set but the paths under it do "
                    "not exist.\n       Check the layout, or override each root "
                    "individually by env var of the same name.")
        else:
            # The overwhelmingly common cause on a pod. ~/.bashrc lives on the
            # container disk, not the /workspace network volume, so exports
            # added there vanish when the container is replaced -- and the
            # paths silently revert to config.py's Windows defaults.
            hint = ("\n       TWINGUARD_DATA_ROOT is NOT SET, so these are "
                    "config.py's built-in defaults.\n       On a pod this almost "
                    "always means the container restarted: ~/.bashrc is on the\n"
                    "       container disk, not the /workspace network volume, so "
                    "exports there are lost.\n"
                    "         now:     export TWINGUARD_DATA_ROOT=/workspace/data\n"
                    "         durably: bash setup_env.sh   "
                    "(writes it into .venv/bin/activate, which is on /workspace)")
        return FAIL, f"{len(missing)} dataset root(s) not found:\n{detail}{hint}"
    root = os.environ.get("TWINGUARD_DATA_ROOT")
    where = f"TWINGUARD_DATA_ROOT={root}" if root else "config.py defaults"
    return PASS, f"all four dataset roots exist ({where})"


def c_cityscapes():
    from data.cityscapes_dataset import CityscapesDataset
    train = CityscapesDataset(split="train", normalize=False)
    val = CityscapesDataset(split="val", normalize=True)

    # Cityscapes train/val is 2975/500. Anything substantially short means a
    # partial unzip -- the directory tree exists and `ls` looks fine, but the
    # files are not all there.
    if len(train) < 2900 or len(val) < 490:
        return FAIL, (
            f"only {len(train)} train / {len(val)} val images found "
            f"(expected 2975 / 500). The dataset tree exists but is "
            f"incomplete -- most likely an unzip that did not finish."
        )
    bad = [p for p in train.labels[:200] if not os.path.exists(p)]
    if bad:
        return FAIL, f"{len(bad)} of first 200 train labels missing, e.g. {bad[0]}"

    # Read one image end to end. Directory listings can succeed against files
    # that are truncated or unreadable on a network volume.
    try:
        image, label = train[0]
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"first training sample failed to load: {type(exc).__name__}: {exc}"
    if image.shape != (3, config.INPUT_HEIGHT, config.INPUT_WIDTH):
        return FAIL, f"unexpected image shape {tuple(image.shape)}"

    return PASS, (f"{len(train)} train / {len(val)} val images, labels resolve, "
                  f"sample loads as {tuple(image.shape)}")


def c_fishyscapes():
    from data.fishyscapes_dataset import list_fishyscapes_pairs, split_fishyscapes_pairs
    pairs = list_fishyscapes_pairs()
    val, test = split_fishyscapes_pairs(pairs)
    overlap = {p[1] for p in val} & {p[1] for p in test}
    if overlap:
        return FAIL, f"val/test split overlaps on {len(overlap)} images"
    if len(pairs) != 100:
        return WARN, f"expected 100 pairs, found {len(pairs)}"
    return PASS, f"100 pairs, split {len(val)} val / {len(test)} test, no overlap"


def c_anomaly_bank():
    from data.anomaly_sources import build_anomaly_bank
    bank = build_anomaly_bank()
    if len(bank) < 50:
        return WARN, f"only {len(bank)} objects in the '{bank.name}' bank -- thin"
    sample = bank.sample()
    if sample is None:
        return FAIL, "bank returned an empty mask on first sample"
    rgb, mask = sample
    if rgb.shape[:2] != mask.shape:
        return FAIL, f"rgb {rgb.shape[:2]} != mask {mask.shape}"
    return PASS, (f"source='{bank.name}', {len(bank)} objects, "
                  f"sample crop {rgb.shape[:2]}, {int(mask.sum())} object px")


def c_no_excluded_categories():
    """The single most damaging possible mistake with COCO: leaking a
    category Cityscapes already knows would train the model that cars are
    anomalies."""
    if config.ANOMALY_SOURCE != "coco":
        return PASS, "source is not coco -- exclusion check not applicable"
    import glob
    names = [os.path.basename(p) for p in
             glob.glob(os.path.join(config.COCO_OBJECTS_DIR, "*_rgb.png"))]
    excluded = {e.lower().replace(" ", "_") for e in config.COCO_EXCLUDED_CATEGORIES}
    leaked = set()
    for n in names:
        parts = n.replace("_rgb.png", "").split("_", 1)
        if len(parts) == 2 and parts[1].lower() in excluded:
            leaked.add(parts[1].lower())
    if leaked:
        return FAIL, (f"Cityscapes-known categories present in the anomaly "
                      f"bank: {sorted(leaked)} -- regenerate the bank")
    present = sorted({n.replace("_rgb.png", "").split("_", 1)[1] for n in names
                      if len(n.split("_", 1)) == 2})
    return PASS, (f"{len(names)} cutouts, {len(present)} categories, "
                  f"none overlapping Cityscapes")


def c_cutmix():
    from data.cityscapes_dataset import CityscapesDataset
    from data.cutmix import CutMixAugmentedDataset
    base = CityscapesDataset(split="train", normalize=False)
    aug = CutMixAugmentedDataset(base, p=1.0)

    # Sizes MUST be measured per connected component. Taking the bounding box
    # of all anomaly pixels at once spans the gap between separate objects and
    # reports ~0.5 regardless of the real object size -- which is exactly what
    # an earlier version of this check did, and it reported PASS on it.
    from scipy import ndimage

    n_samples = 24
    pos_rates, sizes, counts = [], [], []
    with_objects, ignore_ok = 0, 0
    for i in range(n_samples):
        _, seg, ood = aug[i]
        ood_np = ood.numpy() > 0
        pos_rates.append(float(ood_np.mean()))
        if not ood_np.any():
            # No object landed. Counted separately -- an empty sample is not
            # an ignore-label failure, and conflating the two reported a
            # spurious FAIL in an earlier version of this check.
            continue
        with_objects += 1
        if (seg.numpy()[ood_np] == config.CUTMIX_SEG_IGNORE_INDEX).all():
            ignore_ok += 1
        labelled, n = ndimage.label(ood_np)
        counts.append(n)
        for j in range(1, n + 1):
            ys, xs = np.where(labelled == j)
            sizes.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))

    mean_rate = float(np.mean(pos_rates))
    if mean_rate == 0:
        return FAIL, "no anomaly pixels produced at p=1.0 -- pasting is broken"
    if ignore_ok < with_objects:
        return FAIL, (f"seg label not set to ignore under pasted objects in "
                      f"{with_objects - ignore_ok}/{with_objects} samples that "
                      f"actually received one")
    empty_frac = 1.0 - with_objects / n_samples
    if empty_frac > 0.10:
        return WARN, (f"{empty_frac:.0%} of samples received no object at "
                      f"p=1.0 -- positive signal is being wasted")

    frac = np.array(sizes) / config.INPUT_HEIGHT
    median = float(np.median(frac))
    # Real Fishyscapes: p25 0.022, median 0.043, p75 0.072. Training outside
    # this band is the mismatch that plateaued three earlier attempts.
    if not 0.015 <= median <= 0.12:
        return FAIL, (f"per-object median size {median:.3f} of short side is "
                      f"outside the real anomaly distribution (median 0.043) "
                      f"-- adjust CUTMIX_SCALE_MIN/MAX")
    if mean_rate > 0.02:
        return WARN, (f"anomaly pixel rate {mean_rate:.4%}/image is far above "
                      f"the real 0.280% -- objects may be too large or too many")

    return PASS, (f"{np.mean(counts):.2f} objects/image, {empty_frac:.0%} empty, pos rate "
                  f"{mean_rate:.4%} (real 0.280%), per-object longest side "
                  f"p25={np.percentile(frac, 25):.3f} median={median:.3f} "
                  f"p75={np.percentile(frac, 75):.3f} "
                  f"(real 0.022/0.043/0.072), seg->ignore verified")


def c_normalization():
    from data.cityscapes_dataset import CityscapesDataset
    ds = CityscapesDataset(split="val", normalize=True)
    image, _ = ds[0]
    mean, std = float(image.mean()), float(image.std())
    if abs(mean) > 1.5 or std < 0.3:
        return FAIL, f"normalised tensor looks wrong: mean={mean:.3f} std={std:.3f}"
    if image.min() >= 0.0:
        return FAIL, ("normalised tensor has no negative values -- looks like "
                      "raw [0,1] input, normalisation is not being applied")
    return PASS, (f"encoder-space input: mean={mean:.3f} std={std:.3f} "
                  f"range [{float(image.min()):.2f}, {float(image.max()):.2f}]")


def c_model():
    import torch
    from model.twinguard_model import (TwinGuardModel, verify_encoder_frozen,
                                       verify_heads_independent)
    model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD)
    if not verify_encoder_frozen(model):
        return FAIL, "encoder is NOT frozen"
    if not verify_heads_independent(model):
        return FAIL, "OOD heads are not independently initialised"

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 128, 256))
    expected = {"seg_logits", "ood_logits", "ood_scores", "ood_fused", "ood_disagreement"}
    if not expected.issubset(out):
        return FAIL, f"model output missing {expected - set(out)}"
    logits = out["ood_logits"]
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return WARN, "ood_logits look bounded to [0,1] -- is a sigmoid still applied?"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return PASS, (f"encoder frozen ({frozen / 1e6:.1f}M params), heads independent, "
                  f"{trainable / 1e6:.1f}M trainable, logits emitted")


def c_global_rng_not_hijacked():
    """OOD head init used to call torch.manual_seed(seed), which RESET the
    global RNG to the last head's seed. Everything downstream -- data
    shuffling, dropout -- then ran from seed 7 no matter what
    config.GLOBAL_SEED said, so runs were not reproducible in the way the
    config claimed.

    Constructing any nn.Conv2d consumes global RNG draws, and that is fine.
    The invariant that actually matters is that global randomness after
    construction still DEPENDS on the seed set before it. Under the old bug
    it did not: every pre-seed produced the same post-construction stream.
    """
    import torch
    from model.twinguard_model import TwinGuardModel

    def stream_after_construction(seed):
        torch.manual_seed(seed)
        TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD)
        return torch.randn(4)

    a, b = stream_after_construction(0), stream_after_construction(999)
    if torch.equal(a, b):
        return FAIL, ("global RNG is hijacked by model construction -- "
                      "GLOBAL_SEED has no effect on shuffling or dropout")

    # Head weights must still be reproducible from their own seeds, and must
    # not drift when the global seed changes.
    torch.manual_seed(0)
    m1 = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD)
    torch.manual_seed(12345)
    m2 = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD)
    for h, (h1, h2) in enumerate(zip(m1.ood_heads, m2.ood_heads)):
        for p1, p2 in zip(h1.parameters(), h2.parameters()):
            if not torch.equal(p1, p2):
                return FAIL, (f"head {h} weights changed with the global seed -- "
                              f"per-head seeding is not isolated")
    return PASS, ("GLOBAL_SEED still controls global randomness; head weights "
                  "reproducible from their own seeds and independent of it")


def c_loss_step():
    """One real optimiser step -- catches shape/dtype errors that would
    otherwise surface an hour into a paid run."""
    import torch
    from losses import build_seg_criterion, compute_total_loss
    from model.twinguard_model import TwinGuardModel

    model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD)
    model.train()
    model.encoder.eval()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    criterion = build_seg_criterion()

    images = torch.randn(2, 3, 128, 256)
    seg_labels = torch.randint(0, config.NUM_SEG_CLASSES, (2, 128, 256))
    seg_labels[0, :10, :10] = 255  # exercise ignore_index
    ood_target = torch.zeros(2, 128, 256)
    ood_target[:, 40:60, 40:60] = 1.0

    out = model(images)
    loss, l_seg, l_ood = compute_total_loss(
        out["seg_logits"], seg_labels, out["ood_logits"], ood_target, criterion)
    if not torch.isfinite(loss):
        return FAIL, f"loss is not finite: {loss.item()}"
    loss.backward()

    grads = [p.grad for p in model.ood_heads.parameters() if p.grad is not None]
    if not grads:
        return FAIL, "OOD heads received no gradient"
    gnorm = float(torch.cat([g.flatten() for g in grads]).norm())
    if gnorm == 0.0:
        return FAIL, "OOD head gradient norm is exactly zero"
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    if enc_grads:
        return FAIL, "frozen encoder received gradients"
    optimizer.step()
    return PASS, (f"loss={loss.item():.4f} (seg={l_seg.item():.4f} "
                  f"ood={l_ood.item():.4f}), ood grad norm={gnorm:.4f}, "
                  f"encoder got no gradient")


def c_metrics():
    from metrics import ScoreHistogram
    rng = np.random.default_rng(0)
    labels = (rng.random(50_000) < 0.05).astype(np.int64)
    scores = np.clip(rng.normal(np.where(labels == 1, 0.7, 0.3), 0.15), 0, 1)
    hist = ScoreHistogram()
    hist.update(scores, labels)
    auroc = hist.auroc()
    if not (0.9 < auroc < 1.0):
        return FAIL, f"AUROC on a separable synthetic case is {auroc:.4f}"
    return PASS, (f"AUROC={auroc:.4f} AP={hist.average_precision():.4f} "
                  f"on synthetic separable data (run validate_metrics.py for "
                  f"the full sklearn comparison)")


def c_device():
    import torch
    if not torch.cuda.is_available():
        return WARN, ("no CUDA device -- training will run on CPU and take "
                      "days. Fine for preflight, not for the real run.")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total < 20:
        return WARN, (f"{name}, {total:.0f}GB -- the b5 encoder at "
                      f"{config.INPUT_HEIGHT}x{config.INPUT_WIDTH} and batch "
                      f"{config.BATCH_SIZE} wants ~20GB+. Reduce BATCH_SIZE "
                      f"or use a larger GPU.")
    return PASS, f"{name}, {total:.0f}GB"


def main():
    print(f"TwinGuard preflight\n  encoder: {config.ENCODER_NAME}\n"
          f"  anomaly source: {config.ANOMALY_SOURCE}\n")

    checks = [
        ("library versions", c_versions, True),
        ("dataset paths exist", c_paths, True),
        ("cityscapes loads", c_cityscapes, True),
        ("fishyscapes pairs + split", c_fishyscapes, True),
        ("anomaly bank", c_anomaly_bank, True),
        ("no Cityscapes-known categories in bank", c_no_excluded_categories, True),
        ("input normalization", c_normalization, True),
        ("cutmix output", c_cutmix, True),
        ("model wiring", c_model, True),
        ("global RNG not hijacked", c_global_rng_not_hijacked, True),
        ("one training step", c_loss_step, True),
        ("metrics", c_metrics, True),
        ("compute device", c_device, False),
    ]
    for name, fn, fatal in checks:
        check(name, fn, fatal)

    fatal_failures = [r for r in results if r[1] == FAIL and r[3]]
    warnings = [r for r in results if r[1] in (WARN,) or (r[1] == FAIL and not r[3])]

    print("\n" + "=" * 68)
    if fatal_failures:
        print(f"{len(fatal_failures)} BLOCKING FAILURE(S) -- do not start training:")
        for name, _, detail, _ in fatal_failures:
            print(f"  - {name}: {detail}")
        return 1
    if warnings:
        print(f"READY, with {len(warnings)} warning(s):")
        for name, _, detail, _ in warnings:
            print(f"  - {name}: {detail}")
    else:
        print("ALL CHECKS PASSED -- ready to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# RunPod guide — TwinGuard Experiment B on a **shared** pod

Written for: running this on a pod someone else set up and is still using.

The pod already exists, the datasets are already uploaded, and your friend's run lives in `/workspace/TwinGuard_CARLA`. **You are a guest on this machine.** Everything below keeps your work in its own directory and out of theirs.

`local$` = your Windows machine (PowerShell). `pod$` = the pod over SSH.

---

## The five ways you could break your friend's run

Read this once — everything in the guide follows from it.

| Risk | What happens | How this guide avoids it |
|---|---|---|
| **Shared Python** | `pip install` upgrades packages system-wide; their next run breaks | Step 3 builds a venv that borrows torch but installs everything else privately |
| **Shared GPU** | Two trainings at once → CUDA OOM, both die | Step 1 checks free VRAM before you start |
| **Same directory** | Overwriting their `config.py`, checkpoints or MLflow data | Step 2 gives you `/workspace/twinguard_v2` |
| **Same port** | `mlflow ui` collides with RunPod's nginx or JupyterLab | Step 8 checks first, then binds a free port on loopback only |
| **Disk full** | Both writing checkpoints fills the volume; their run crashes mid-epoch | Step 1 checks free space |

The one rule: **never write anything inside `/workspace/TwinGuard_CARLA`.** Read it if you like. Do not touch it.

---

## Step 0a — Clear the previous run (re-runs only)

Skip on a first run. Repeating training? Clear the old outputs so the MLflow table is not a mix of runs, including any run you killed part-way.

```
cd /workspace/twinguard_v2 && rm -rf mlflow.db mlartifacts mlruns checkpoints train.log __pycache__ check_data_cutmix.png
```

Then confirm they are actually gone — `rm -rf` is silent about everything, including doing nothing:

```
ls
```

`mlruns` is worth naming explicitly: it is the old **file-based** MLflow store. The project now uses `mlflow.db` (SQLite), so a leftover `mlruns/` directory is stale data from an earlier version that nothing reads but which is confusing to find later.

Only do this **after** pulling anything you want to keep (Step 9). On your own machine, archive rather than delete, so the previous run stays comparable:

```
mkdir run1_archive ; move model_3head_best.pth run1_archive ; move train.log run1_archive ; move mlflow.db run1_archive
```

If `move mlflow.db` reports the file does not exist, you simply never downloaded it — harmless.

---

## Step 0 — Local prep

```
local$ cd "D:\Academics (D)\SEM-7\PROJECTS\FinalYearProject\TwinGuard_CARLA"
local$ python preflight.py
local$ python make_upload.py
```

`preflight.py` must end in `READY` (the "no CUDA" warning is expected locally). `make_upload.py` writes `twinguard_code.zip`, ~60KB — code only, since the datasets are already on the pod.

> Use `make_upload.py`, **not** `Compress-Archive`. `Compress-Archive -Path "<folder>"` nests everything under a `TwinGuard_CARLA/` directory inside the zip — that is how the datasets got their structure, and it is correct for them. Here you want the files at the zip root so they land directly in your own folder.

---

## Step 1 — Connect and survey (read-only)

```
local$ ssh root@194.xxx.xxx.xxx -p 40123 -i ~/.ssh/id_ed25519
```

Find out what is on the machine **before** adding anything. Everything here is read-only.

**If you have already uploaded the code**, just run the bundled script:

```
pod$ bash /workspace/twinguard_v2/pod_survey.sh
```

**On a first visit**, paste this single line. It is deliberately one line with `;` separators — pasting a multi-line block into an SSH session is unreliable, because some terminals collapse the newlines and turn `echo "=== GPU ==="` + newline + `nvidia-smi` into `echo "=== GPU ===" nvidia-smi`, which prints the command name instead of running it:

```
nvidia-smi; echo "--- gpu users ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; echo "--- python procs ---"; ps aux | grep -i python | grep -v grep; echo "--- disk ---"; df -h /workspace; echo "--- workspace ---"; ls -la /workspace; echo "--- sizes ---"; du -sh /workspace/* 2>/dev/null | sort -h; echo "--- datasets ---"; ls /workspace/data; echo "--- ports ---"; ss -tlnp 2>/dev/null | head -20; echo "--- cpu/ram ---"; nproc; free -h; echo "--- python ---"; command -v python; python --version; python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

### How to read the output

**GPU free memory** — in `nvidia-smi`, look at `MiB / 46068MiB`. You need roughly **20GB free** for the b5 encoder at batch 4.

- Under 20GB free → your friend is training. **Do not start.** Wait, agree a slot with them, or drop `BATCH_SIZE` to 2 in your own `config.py` and re-check.
- `nvidia-smi --query-compute-apps` names the exact processes holding memory. An empty list means the GPU is idle and you are clear.

**Disk** — you need ~5GB: ~1GB for the COCO download plus checkpoints (~350MB each, only the best is kept). If `df -h` shows under 10GB available, tell your friend before filling it. A full volume crashes their run mid-epoch, which costs them more than it costs you.

**Datasets** — confirm all four are present:

```
ls /workspace/data
```

Expect exactly these four names:

```
fishyscapes_lostandfound   gtFine_trainvaltest   leftImg8bit   leftImg8bit_trainvaltest
```

**Empty folders with the right names are the trap here.** An interrupted `unzip` leaves the directory tree behind with no files in it, and `ls` looks perfectly healthy. Count actual files:

```
du -sh /workspace/data 2>/dev/null; echo "--- file counts ---"; for d in leftImg8bit_trainvaltest gtFine_trainvaltest fishyscapes_lostandfound leftImg8bit; do echo "$d: $(find /workspace/data/$d -type f 2>/dev/null | wc -l)"; done
```

Expect roughly:

```
18G     /workspace/data
--- file counts ---
leftImg8bit_trainvaltest: 5002
gtFine_trainvaltest: 20002
fishyscapes_lostandfound: 100
leftImg8bit: 2239
```

(These are the exact counts from the same zips, verified against a local copy. `gtFine` carries four files per image, hence 20002.)

`du` in the **tens of GB** is the number that matters. A few kilobytes means the folders exist but the data does not.

> **Two things about these commands.** They are deliberately **one line each** with `;` separators, and they use `-type f` instead of filename patterns. Multi-line blocks get mangled when pasted into SSH — some terminals collapse the newlines, turning `du -sh /workspace/data` + newline + `echo "..."` into `du -sh /workspace/data echo "..."`. And filename patterns containing `*` get silently eaten if you copy from a *rendered* view of this file, because a matched pair of asterisks is markdown emphasis. `-type f` avoids the problem entirely.

> **`Permission denied` on `gtFine/test` is harmless.** Cityscapes withholds the test-split labels, nothing reads that folder, and `2>/dev/null` above hides the noise. Train and val are what matter.

### If the datasets are not actually there

The source zips normally remain in `/workspace`. Unzipping ~17GB takes a while, so run it detached — again as one line:

```
cd /workspace && mkdir -p data && nohup bash -c 'cd /workspace/data && for z in cityscapes_images cityscapes_labels fishyscapes_labels fishyscapes_images; do unzip -q -o ../$z.zip -d . || exit 1; done && echo UNZIP_DONE' > /workspace/unzip.log 2>&1 &
```

Watch it climb:

```
watch -n 30 'du -sh /workspace/data 2>/dev/null; tail -2 /workspace/unzip.log'
```

Wait for `UNZIP_DONE`, then re-run the counts above. **Check with your friend first** — two `unzip` processes writing the same tree will corrupt it.

If the counts are right but the shape differs, do not move anything (the files are shared) — find the real paths and use the per-path overrides in Step 4:

```
find /workspace -maxdepth 4 -type d -name leftImg8bit 2>/dev/null; find /workspace -maxdepth 4 -type d -name gtFine 2>/dev/null; find /workspace -maxdepth 3 -type d -name fishyscapes_lostandfound 2>/dev/null
```

---

## Step 2 — Claim your own directory

```
pod$ mkdir -p /workspace/twinguard_v2
pod$ cd /workspace/twinguard_v2
pod$ pwd
```

Everything you do happens here. `config.py` writes `checkpoints/` and `mlflow.db` as **relative** paths, so simply running from this directory keeps your outputs entirely separate from your friend's — no extra configuration needed.

Upload your code from a **local** terminal (open a second PowerShell window; leave the SSH session running):

```
local$ cd "D:\Academics (D)\SEM-7\PROJECTS\FinalYearProject\TwinGuard_CARLA"
local$ scp -P 40123 -i ~/.ssh/id_ed25519 twinguard_code.zip root@194.xxx.xxx.xxx:/workspace/twinguard_v2/
```

Back on the pod:

```
pod$ cd /workspace/twinguard_v2
pod$ unzip -o twinguard_code.zip && rm twinguard_code.zip
pod$ ls
```

You should see `train.py`, `preflight.py`, `config.py`, `utils.py`, `data/`, `model/` directly in this folder — **not** nested inside another `TwinGuard_CARLA/`.

---

## Step 3 — A Python environment that cannot break your friend's

This is the step that matters most on a shared pod.

```
pod$ cd /workspace/twinguard_v2
pod$ python -m venv --system-site-packages .venv
pod$ source .venv/bin/activate
```

`--system-site-packages` is the important flag: the venv **inherits** the pod's existing packages — including the CUDA-matched PyTorch, which is several GB and must not be reinstalled — while anything you install now lands in `.venv` and is invisible to your friend.

```
(.venv) pod$ pip install -r requirements-runpod.txt
```

Then check what actually landed — the pod's torch sets a ceiling on transformers:

```
(.venv) pod$ python -c "
import torch, transformers
from transformers.utils import is_torch_available
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| transformers sees torch:', is_torch_available())"
```

`transformers sees torch: True` is what you need.

Two version rules apply, and `preflight.py` checks both:

| pod torch | required transformers | why |
|---|---|---|
| >= 2.6 | `<5` | transformers 5.x needs torch>=2.5; below that it silently disables its torch backend |
| < 2.6 | `<4.51` | transformers 4.51+ blocks `torch.load` (CVE-2025-32434), and nvidia/segformer ships no safetensors |

This pod has torch 2.4.1, so it needs `transformers>=4.46.3,<4.51`:

```
(.venv) pod$ pip install 'transformers>=4.46.3,<4.51'
```

**Never fix this by upgrading torch.** See the troubleshooting section — it breaks cuDNN in a way that is not obvious.

Confirm the GPU still works inside the venv:

```
(.venv) pod$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and name the GPU. If it prints `False`, a CPU-only torch got installed into the venv and is shadowing the system build:

```
(.venv) pod$ pip uninstall -y torch torchvision torchaudio
```

That removes only the venv's copy and falls back to the inherited system one. Re-check.

> **Every future session must re-activate the venv:**
> ```
> cd /workspace/twinguard_v2 && source .venv/bin/activate
> ```
> Forgetting means you are on the system Python with different package versions. The `(.venv)` prefix in your prompt is the confirmation.

---

## Step 4 — Point at the existing datasets

Do not re-upload anything — reuse what is already there.

```
(.venv) pod$ bash setup_env.sh
(.venv) pod$ source .venv/bin/activate
```

`setup_env.sh` appends the exports to `.venv/bin/activate`, which lives on `/workspace`.

> **Do not use `~/.bashrc` for this.** It looks permanent and is not. On RunPod only `/workspace` is a network volume — `/root` is on the container disk and is destroyed whenever the pod restarts or redeploys. The next session then has no `TWINGUARD_DATA_ROOT`, `config.py` silently falls back to its built-in Windows default, and preflight reports four missing dataset roots pointing at a `D:\` path. This has already happened once in this project.

To point somewhere else, or use fewer workers:

```
(.venv) pod$ bash setup_env.sh /some/other/datasets 8
```

Verify the code resolves them:

```
(.venv) pod$ cd /workspace/twinguard_v2
(.venv) pod$ python -c "
import os, config
for n in ('CITYSCAPES_IMAGES_ROOT','CITYSCAPES_LABELS_ROOT','FISHYSCAPES_LABELS_DIR','FISHYSCAPES_IMAGES_ROOT'):
    p = getattr(config, n)
    print(('OK   ' if os.path.isdir(p) else 'MISS '), n, '->', p)"
```

All four must say `OK`.

**If the pod layout differs**, override each path individually rather than moving any data — those files are shared, and moving them breaks your friend's run:

```
(.venv) pod$ export CITYSCAPES_IMAGES_ROOT=/actual/path/to/leftImg8bit
(.venv) pod$ export CITYSCAPES_LABELS_ROOT=/actual/path/to/gtFine
(.venv) pod$ export FISHYSCAPES_LABELS_DIR=/actual/path/to/fishyscapes_lostandfound
(.venv) pod$ export FISHYSCAPES_IMAGES_ROOT=/actual/path/to/laf/leftImg8bit
```

Add whichever you need to `.venv/bin/activate` (the same file `setup_env.sh` writes to) so they persist across restarts -- not to `~/.bashrc`, which does not.

---

## Step 5 — Build the anomaly bank

```
(.venv) pod$ cd /workspace/twinguard_v2
(.venv) pod$ python download_coco_anomalies.py
```

~1GB download, a few minutes. It writes to `/workspace/twinguard_v2/data/coco_objects/` — your directory, not shared. The last lines must include:

```
verified: no Cityscapes-overlapping category present
```

That assertion is the one that matters: a leaked COCO `car` or `person` would train the model that cars are anomalies and make the whole run worthless.

> **Using CARLA objects instead?** Copy them into `/workspace/twinguard_v2/data/images` and `/workspace/twinguard_v2/data/masks`, then set `ANOMALY_SOURCE = "carla"` in your `config.py`. Note `/workspace/TwinGuard_CARLA/data/images` is empty — that is the original problem, not a source to copy from.

---

## Step 6 — Preflight on the pod

**Do not skip this.** Two minutes now, versus finding a problem three hours into a shared GPU slot.

```
(.venv) pod$ python preflight.py
(.venv) pod$ python validate_metrics.py
```

Every line must read `PASS` ending in `ALL CHECKS PASSED`, and the metrics script must end in `ALL METRICS MATCH SKLEARN`. The compute-device check should now name the GPU instead of warning.

| Failure | Fix |
|---|---|
| dataset root(s) not found | Step 4 — the message prints the exact path it tried |
| no cutouts found | Step 5 |
| Cityscapes-known categories in bank | `rm -rf data/coco_objects`, rerun Step 5 |
| expected 100 Fishyscapes pairs | labels and images not both present — recheck Step 1 counts |
| GPU memory warning | your friend is on the GPU, or lower `BATCH_SIZE` |

---

## Step 7 — Baseline, then train

Run Experiment A first. Its published 0.8304 was measured over all 100 images; you need the **test-half** number to compare fairly against Experiment B.

```
(.venv) pod$ python experiment_a.py
```

~10 minutes. Note the `REPORTED (test half)` AUROC — that is your real bar.

**Re-check the GPU right before training**, in case your friend started something while you were setting up:

```
(.venv) pod$ nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Keeping the GPU bill down

Three settings control how long a run takes. Defaults are already set for a pod; this is what they do and when to change them.

| Setting | Default | Effect |
|---|---|---|
| `EPOCHS` | 8 | The first full run converged by epoch 5–6; 15 epochs added ~0.02 AP for double the cost. Raise only if the val metric is still clearly climbing at the end. |
| `USE_AMP` | `True` | bfloat16 autocast. The frozen b5 encoder runs 2976 images per epoch and dominates runtime; bf16 roughly halves it on an A40. |
| `EVAL_TEST_ON_IMPROVEMENT_ONLY` | `True` | Scores the test half only when val improves, plus the final epoch. Saves ~25% of epoch time, and only the selected checkpoint's test numbers are ever reported anyway. |
| `TWINGUARD_NUM_WORKERS` | env var | CutMix does real per-sample CPU work (decode, resize, paste, harmonise). Too low and the GPU waits on the CPU. |

A 15-epoch run at `num_workers=8` with no AMP took **57 minutes**. With these defaults expect roughly **15–20 minutes**.

If epochs are still slower than ~100s, the dataloader is the bottleneck rather than the GPU — raise workers (the pod has 96 cores, but leave some for your friend):

```
(.venv) pod$ export TWINGUARD_NUM_WORKERS=16
```

Confirm which side is the bottleneck before turning knobs — low `sm%` here means the GPU is idle waiting for data:

```
(.venv) pod$ nvidia-smi dmon -s u -c 20
```

**Set the DataLoader workers.** The default is 0 — the main process does every PNG decode, resize and CutMix paste while the GPU idles. A pod with dozens of free cores should not be left doing that:

`setup_env.sh` in Step 4 already sets this to 16. To check or change it:

```
(.venv) pod$ echo "$TWINGUARD_NUM_WORKERS"; nproc
(.venv) pod$ bash setup_env.sh /workspace/data 8 && source .venv/bin/activate
```

8 is a deliberate choice rather than the maximum: it is enough to keep an A40 fed, and it leaves the rest of the machine for whoever else is on it. The value is logged to MLflow, and `train.py` prints it at startup so a slow run is easy to diagnose later.

Then launch under `nohup` so an SSH drop cannot kill it:

```
(.venv) pod$ cd /workspace/twinguard_v2
(.venv) pod$ nohup python -u train.py > train.log 2>&1 &
(.venv) pod$ echo $!          # note this process id
(.venv) pod$ tail -f train.log
```

`Ctrl-C` stops watching, not training. To actually stop: `kill <pid>`.

### Healthy output

```
epoch 1/15 done in 228.3s -- loss=0.2794 | VAL auroc=0.9857 | TEST auroc=0.9919 ap=0.6813 ece=0.0006 fpr95=0.0309 | mIoU=0.7337
  head separation (mean score on anomalous - on normal): h0=+0.3512  h1=+0.3387  h2=+0.3410
  -> new best VAL AUROC 0.9857 (test 0.9919), checkpoint saved
```

What to watch:

- **`head separation`** — the collapse alarm: mean score on anomalous pixels minus mean on normal ones. Positive and growing is healthy. The `WARNING` fires only when separation is under 0.01 **and** AUROC is under 0.70, i.e. the heads genuinely cannot tell the classes apart.
- **Do not read the global output std as collapse.** The positive rate is 0.24%, so in a healthy detector nearly every pixel is a correctly-near-zero negative and the global std is low by construction — a model at 0.99 AUROC still shows std ~0.04. An earlier version of this warning used std and fired on a perfectly good run.
- **`VAL` vs `TEST` auroc** — should track closely.
- **`mIoU`** — should climb then hold. Falling while AUROC climbs means the OOD heads are eating the segmentation head's capacity.
- **Gates** — 0.75 is the pre-calibration gate for Phase 2b; 0.83 is the real post-calibration target.
- **Checkpoint selection uses val AP, not AUROC** (`SELECTION_METRIC` in config). Once the model works AUROC saturates — it moved only 0.012 across a full run while AP moved 0.116 — so selecting on AUROC is close to selecting on noise. In the first full run it picked the epoch with the *worst* AP of all 15. AP is also what the Fishyscapes benchmark ranks on.
- **The closing summary prints the SELECTED checkpoint's numbers.** Quote those. The last epoch's line describes a different model from the `.pth` on disk.

---

## Step 8 — Viewing results

```
(.venv) pod$ grep "done in" train.log      # one line per epoch
(.venv) pod$ python check_runs.py          # all runs, with the gates
(.venv) pod$ python check_runs.py --trend  # per-epoch history
(.venv) pod$ python check_collapse.py      # per-head mean/std on the checkpoint
(.venv) pod$ python check_data.py          # CutMix composites + size stats
```

### MLflow UI — pick a free port, check first

On this pod RunPod's own nginx already holds **8081, 8001, 7861, 7270, 3001 and 9091**, and JupyterLab holds **8888**. Do not assume any particular port is free — check, then bind:

```
(.venv) pod$ ss -tlnp | grep -E ':(8050|8080)\s'   # no output = free
(.venv) pod$ cd /workspace/twinguard_v2
(.venv) pod$ mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8050
```

`--host 127.0.0.1` rather than `0.0.0.0`: the SSH tunnel reaches it either way, and binding only to loopback keeps it off the pod's public interfaces.

**Use an SSH tunnel to reach it.** From a local terminal:

```
local$ ssh -N -L 8050:localhost:8050 root@194.xxx.xxx.xxx -p 40123 -i ~/.ssh/id_ed25519
```

Leave it running and open <http://localhost:8050>.

> Prefer the tunnel over RunPod's HTTP-port proxy here. Changing a pod's exposed ports needs console access and can require a restart — which would kill your friend's training. The tunnel needs nothing on the pod and affects no one.

Your `mlflow.db` lives in `/workspace/twinguard_v2/`, so you only ever see your own runs.

---

## Step 9 — Get your results off

Do this as soon as training finishes. **You do not control this pod** — if your friend terminates it, everything is gone.

```
local$ scp -P 40123 -i ~/.ssh/id_ed25519 root@194.xxx.xxx.xxx:/workspace/twinguard_v2/checkpoints/model_3head_best.pth .
local$ scp -P 40123 -i ~/.ssh/id_ed25519 root@194.xxx.xxx.xxx:/workspace/twinguard_v2/train.log .
local$ scp -P 40123 -i ~/.ssh/id_ed25519 root@194.xxx.xxx.xxx:/workspace/twinguard_v2/mlflow.db .
```

`mlflow.db` is the entire experiment history in one file — every metric, every epoch, every parameter. View it locally:

```
local$ mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Tidying up

Free the shared disk once your results are safe — **only inside your own directory**:

```
(.venv) pod$ rm -rf /workspace/twinguard_v2/data/_coco_download
```

Leave `/workspace/data` alone. That is the shared dataset your friend is also using.

---

## Reconnecting later

```
local$ ssh root@194.xxx.xxx.xxx -p 40123 -i ~/.ssh/id_ed25519
pod$   cd /workspace/twinguard_v2 && source .venv/bin/activate
pod$   tail -f train.log
```

**Check whether the pod restarted.** The hostname in your prompt is the container ID, and it changes when the pod is restarted or redeployed:

```
pod$ hostname
```

A different ID than last session means the container was replaced. `/workspace` survives that (it is a network volume) so your venv, code and datasets are still there — but nothing outside `/workspace` is, any running job is gone, and your shell exports are reset. Re-activate the venv and re-check:

```
pod$ cd /workspace/twinguard_v2 && source .venv/bin/activate
pod$ bash pod_survey.sh
```

A prompt without the `(.venv)` prefix means you are on the system Python, not yours.

Check your training survived:

```
pod$ ps aux | grep train.py | grep -v grep
```

No output means it finished or died — the end of `train.log` says which.

---

## Troubleshooting

**`Due to a serious vulnerability issue in torch.load ... upgrade torch to at least v2.6` (CVE-2025-32434)**
transformers 4.51+ refuses to call `torch.load` when torch is older than 2.6. Its suggestion to "use safetensors instead" does not apply here: `nvidia/segformer-*` publishes `pytorch_model.bin` **only**, with no safetensors file in the repo, so there is nothing to switch to. It has to be resolved by version.

**Downgrade transformers. Do not upgrade torch.**

```
(.venv) pod$ pip install 'transformers>=4.46.3,<4.51'
(.venv) pod$ python preflight.py
```

12MB, instant, and it cannot disturb CUDA. transformers 4.50 supports SegFormer identically — that API has been stable for years.

> **Upgrading torch instead looks tempting and is a trap on this pod.** `pip install 'torch>=2.6'` into a `--system-site-packages` venv appears to succeed, then fails at import with `libcudnn.so.9: cannot open shared object file`. pip marks CUDA dependencies it finds in system site-packages as "already satisfied" and skips them, but the venv-local `nvidia/` directory it *does* create shadows the system one — so the skipped libraries become invisible. The pod's `torchvision`/`torchaudio` are also pinned to the old torch and break at the same time. If you have already done this, rebuild the venv rather than patching it:
>
> ```
> deactivate && rm -rf .venv
> python -m venv --system-site-packages .venv
> source .venv/bin/activate
> pip install -r requirements-runpod.txt
> pip install 'transformers>=4.46.3,<4.51'
> ```

**`libcudnn.so.9: cannot open shared object file`**
A torch was pip-installed into the venv and is missing CUDA libraries — see the box above. Rebuild the venv. `preflight.py` detects this on its first check and prints the same instructions.

**`SegformerModel requires the PyTorch library but it was not found`**
torch is fine — `transformers` is the problem. transformers 5.x requires torch>=2.5; on older torch it does not error at import, it just disables its PyTorch backend, so the failure surfaces later and points at the wrong library. Look for this line earlier in the output:

```
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1+cu124
```

Fix inside your venv, and never by upgrading torch:

```
(.venv) pod$ pip install 'transformers>=4.46.3,<5'
```

`preflight.py`'s first check now catches this before anything else runs.

**`torch.cuda.is_available()` is False inside the venv**
A CPU torch got installed into the venv, shadowing the system CUDA build. `pip uninstall -y torch torchvision torchaudio` inside the venv.

**CUDA out of memory**
Someone else is on the GPU. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` shows who. Wait, or set `BATCH_SIZE = 2` in your `config.py`.

**Your friend's run died right after you started**
Almost certainly GPU memory or disk. Check `nvidia-smi` and `df -h`, stop your run, and tell them. This is exactly what Step 1 is for.

**preflight shows dataset paths starting with `D:\`**
`TWINGUARD_DATA_ROOT` is not set, so `config.py` fell back to its Windows defaults. Almost always means the pod restarted and took `~/.bashrc` with it. `bash setup_env.sh && source .venv/bin/activate`.

**`cityscapes train+val labels: 3475` — is that a partial unzip?**
No, that is correct. 2975 train + 500 val = 3475. Cityscapes withholds the test-split labels, and on this pod `gtFine/test` is also permission-denied. Nothing reads it.

**`no MLflow experiment named ...`**
Wrong directory. The tracking URI is relative — run from `/workspace/twinguard_v2`.

**`The filesystem tracking backend ... is in maintenance mode`**
An old `MLFLOW_TRACKING_URI` is set in your shell. `unset MLFLOW_TRACKING_URI`. This project uses `sqlite:///mlflow.db`, which every MLflow version accepts.

**Training much slower than ~15 min/epoch**
Almost always the DataLoader starving the GPU. Check what the run actually used — `train.py` prints `num_workers=N` at startup. If it says 0, you missed the `TWINGUARD_NUM_WORKERS` export in Step 7. Confirm the GPU is the bottleneck and not the CPU with `nvidia-smi dmon -s u` (low `sm%` with high CPU means data loading). If your friend is also running, leave them cores.

**SSH drops and training dies**
You did not use `nohup`. Relaunch with the Step 7 command.

**AUROC near 0.5 and `head output std` under 0.05**
The collapse signature. Stop — more epochs will not help. Save `check_collapse.py` output and the first three epochs of `train.log`. The mean says which failure it is: near 0.0 is collapse to "always normal", near 0.5 is genuine indecision, and they need different fixes.

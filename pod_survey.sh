#!/usr/bin/env bash
# Read-only survey of a shared pod. Changes nothing.
#
#     bash pod_survey.sh
#
# Exists as a file because pasting a multi-line block into an SSH session is
# unreliable -- some terminals collapse the newlines, which turns
#     echo "=== GPU ==="
#     nvidia-smi
# into `echo "=== GPU ===" nvidia-smi`, printing the command name instead of
# running it. A script cannot be mangled that way.

section() { printf '\n===== %s =====\n' "$1"; }

section "GPU"
nvidia-smi

section "WHO IS USING THE GPU"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

section "RUNNING PYTHON PROCESSES"
ps aux | grep -i python | grep -v grep || echo "(none)"

section "DISK"
df -h /workspace

section "TOP LEVEL OF /workspace"
ls -la /workspace

section "SIZE OF EACH TOP-LEVEL FOLDER"
du -sh /workspace/* 2>/dev/null | sort -h

section "DATASETS"
ls -la /workspace/data 2>/dev/null
# Only train+val matter. Cityscapes withholds the test-split labels, and on
# this pod the test directory is also permission-denied -- which is why the
# label count is 3475 (2975 train + 500 val) and not 5000. That is correct,
# not a partial unzip.
echo "cityscapes train images: $(find /workspace/data/leftImg8bit_trainvaltest -name '*_leftImg8bit.png' 2>/dev/null | wc -l)  (expect 5000)"
echo "cityscapes train+val labels: $(find /workspace/data/gtFine_trainvaltest -name '*_gtFine_labelIds.png' 2>/dev/null | wc -l)  (expect 3475 = 2975 train + 500 val)"
echo "fishyscapes labels:      $(find /workspace/data/fishyscapes_lostandfound -name '*_labels.png' 2>/dev/null | wc -l)  (expect 100)"
echo "lost&found images:       $(find /workspace/data/leftImg8bit -name '*_leftImg8bit.png' 2>/dev/null | wc -l)  (expect 2239)"

section "OTHER PROJECT ON THIS POD (look, do not touch)"
ls -la /workspace/TwinGuard_CARLA 2>/dev/null || echo "(not present)"

section "PORTS IN USE"
ss -tlnp 2>/dev/null | head -20

section "CPU / RAM"
echo "cores: $(nproc)"
free -h

section "PYTHON / TORCH / TRANSFORMERS"
echo "interpreter: $(command -v python)"
python --version
python - <<'PY' 2>&1 || echo "  (torch or transformers not importable -- see the guide)"
import torch
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         ", torch.cuda.get_device_name(0))
try:
    import transformers
    from transformers.utils import is_torch_available
    print("transformers", transformers.__version__, "| sees torch:", is_torch_available())
except ImportError:
    print("transformers  NOT INSTALLED")
PY

section "VENV + ENVIRONMENT"
if [ -n "$VIRTUAL_ENV" ]; then
  echo "venv active: $VIRTUAL_ENV"
else
  echo "NO VENV ACTIVE -- run: source /workspace/twinguard_v2/.venv/bin/activate"
fi
if [ -n "$TWINGUARD_DATA_ROOT" ]; then
  echo "TWINGUARD_DATA_ROOT=$TWINGUARD_DATA_ROOT"
else
  echo "TWINGUARD_DATA_ROOT NOT SET -- preflight will look for the datasets on"
  echo "  a Windows path and fail. ~/.bashrc does NOT survive a pod restart"
  echo "  (/root is container disk, not the /workspace network volume)."
  echo "  Fix: bash /workspace/twinguard_v2/setup_env.sh"
fi
echo "TWINGUARD_NUM_WORKERS=${TWINGUARD_NUM_WORKERS:-(not set, defaults to 0)}"
echo "container id (changes on restart): $(hostname)"

printf '\n===== DONE =====\n'

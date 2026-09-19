#!/usr/bin/env bash
# Make TwinGuard's environment variables survive a pod restart.
#
#     bash setup_env.sh
#     source .venv/bin/activate      # now sets everything automatically
#
# The problem this solves: `echo 'export ...' >> ~/.bashrc` looks permanent
# and is not. On RunPod only /workspace is a network volume; /root lives on
# the container disk and is destroyed when the pod restarts or redeploys.
# The next session then runs with no TWINGUARD_DATA_ROOT, config.py falls
# back to its built-in Windows default, and preflight reports four missing
# dataset roots pointing at a D:\ path.
#
# The venv is inside /workspace, so appending the exports to its activate
# script puts them somewhere that actually persists -- and ties them to the
# venv, so you cannot end up with one without the other.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE="$HERE/.venv/bin/activate"

DATA_ROOT="${1:-/workspace/data}"
NUM_WORKERS="${2:-16}"

if [ ! -f "$ACTIVATE" ]; then
  echo "ERROR: no venv at $ACTIVATE"
  echo "Create it first:"
  echo "  cd $HERE && python -m venv --system-site-packages .venv"
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: dataset root '$DATA_ROOT' does not exist."
  echo "Pass the right one:  bash setup_env.sh /path/to/datasets"
  exit 1
fi

MARKER="# --- twinguard env (added by setup_env.sh) ---"

# Idempotent: strip any previous block before appending, so running this
# repeatedly does not stack duplicate exports.
if grep -qF "$MARKER" "$ACTIVATE"; then
  python3 - "$ACTIVATE" "$MARKER" <<'PY'
import sys
path, marker = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
out, skip = [], False
for line in lines:
    if line.strip() == marker.strip():
        skip = True
        continue
    if skip and line.startswith("export TWINGUARD_"):
        continue
    skip = False
    out.append(line)
open(path, "w").write("\n".join(out))
PY
  echo "removed previous twinguard env block"
fi

{
  echo ""
  echo "$MARKER"
  echo "export TWINGUARD_DATA_ROOT=$DATA_ROOT"
  echo "export TWINGUARD_NUM_WORKERS=$NUM_WORKERS"
} >> "$ACTIVATE"

echo "Added to $ACTIVATE:"
echo "  TWINGUARD_DATA_ROOT=$DATA_ROOT"
echo "  TWINGUARD_NUM_WORKERS=$NUM_WORKERS"
echo
echo "Apply to the current shell:"
echo "  source $ACTIVATE"
echo
echo "This lives on /workspace, so it survives pod restarts -- unlike ~/.bashrc."

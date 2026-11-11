#!/usr/bin/env bash
# Offline FID + DINO eval. Default checkpoint: model_2000.pkl under output_dir.
# Usage: bash scripts/eval_checkpoint.sh <run_dir>/config.yaml [checkpoint.pkl]
# Pass the run's own config.yaml (written by launch_train.sh), not configs/course/*.yaml:
# output_dir is read from it, and the course configs only hold a placeholder.
# Env overrides: PYTHON, CUDA_VISIBLE_DEVICES (default 0), HF_ENDPOINT.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?usage: eval_checkpoint.sh <run_dir>/config.yaml [checkpoint.pkl]}"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
PYTHON="${PYTHON:-$(command -v python || true)}"
VENDOR="$FROST_ROOT/vendor/img2img-turbo"

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found ('$PYTHON'). Activate your env or set PYTHON=/path/to/python." >&2
  exit 1
fi
HF_EXPORT=""
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  HF_EXPORT="export HF_ENDPOINT='$HF_ENDPOINT'"
fi

# Pin tmux under workspace; allow CUDA_VISIBLE_DEVICES override (default GPU 0).
export TMUX_TMPDIR="${TMUX_TMPDIR:-$FROST_ROOT/.tmux}"
mkdir -p "$TMUX_TMPDIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# cleanfid expects /tmp/inception-2015-12-05.pt and skips download if any file exists
# (including incomplete). Prefer durable frost cache copy when size matches.
INCEPTION_EXPECTED_BYTES=95607719
INCEPTION_CACHE="$FROST_ROOT/.cache/inception-2015-12-05.pt"
INCEPTION_TMP="/tmp/inception-2015-12-05.pt"
_inception_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  local sz
  sz="$(stat -c '%s' "$p" 2>/dev/null || echo 0)"
  [[ "$sz" -eq "$INCEPTION_EXPECTED_BYTES" ]]
}
if ! _inception_ok "$INCEPTION_TMP"; then
  if _inception_ok "$INCEPTION_CACHE"; then
    cp -f "$INCEPTION_CACHE" "$INCEPTION_TMP"
    echo "Restored Clean-FID Inception from $INCEPTION_CACHE -> $INCEPTION_TMP"
  else
    echo "WARNING: missing/incomplete Inception at $INCEPTION_TMP (and no good cache). FID may hang on download." >&2
  fi
fi

read -r OUT_DIR DATASET_FOLDER METHOD_NAME DATASET_NAME SEED < <(
  "$PYTHON" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "$FROST_ROOT/scripts")
from config_lib import load_frost_config
c = load_frost_config("$CFG")
print(c["output_dir"], c.get("dataset_folder", "data/horse2zebra"),
      c.get("method", "unknown"), c.get("dataset", "unknown"), c.get("seed", 42))
PY
)

if [[ "$OUT_DIR" == *"OUTPUT_DIR_PLACEHOLDER"* ]]; then
  echo "ERROR: $CFG has no resolved output_dir. Pass output/fresh_YYYYMMDD/<dataset>/<method>/config.yaml." >&2
  exit 1
fi

CKPT="${2:-$OUT_DIR/checkpoints/model_2000.pkl}"
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint $CKPT" >&2
  exit 1
fi

SESSION="frost_eval_${DATASET_NAME}_${METHOD_NAME}"
LOG="$OUT_DIR/eval.log"
mkdir -p "$OUT_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$FROST_ROOT" bash -lc "
set -euo pipefail
export PATH='$(dirname "$PYTHON")':\$PATH
export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
export TMUX_TMPDIR='$TMUX_TMPDIR'
$HF_EXPORT
export PYTHONUNBUFFERED=1
export PYTHONPATH='$FROST_ROOT/src:$VENDOR/src'\${PYTHONPATH:+:\$PYTHONPATH}
echo \"EVAL_START \$(date -Iseconds) ckpt=$CKPT cuda=\$CUDA_VISIBLE_DEVICES\" | tee -a '$LOG'
$PYTHON -u -m frost.evaluate_unpaired \\
  --checkpoint '$CKPT' \\
  --dataset_folder '$DATASET_FOLDER' \\
  --output_dir '$OUT_DIR' \\
  --metrics_json '$OUT_DIR/metrics.json' \\
  --dataset_name '$DATASET_NAME' \\
  --method_name '$METHOD_NAME' \\
  --seed $SEED \\
  --image_prep no_resize \\
  2>&1 | tee -a '$LOG'
echo \"EVAL_END \$(date -Iseconds)\" | tee -a '$LOG'
"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

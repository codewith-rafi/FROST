#!/usr/bin/env bash
# Precompute granular masks for a dataset folder (train_A + train_B).
# Usage: bash scripts/precompute_masks.sh horse2zebra
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-horse2zebra}"
VENDOR="$FROST_ROOT/vendor/img2img-turbo"
DATA_DIR="$FROST_ROOT/data/$DATASET"
MASK_DIR="$DATA_DIR/masks"
SESSION="frost_masks_${DATASET}"
PYTHON="${PYTHON:-/home/user0/.conda/envs/rafi/bin/python}"
LOG="$FROST_ROOT/output/mask_precompute_${DATASET}.log"
mkdir -p "$(dirname "$LOG")" "$MASK_DIR"

if [[ ! -d "$DATA_DIR/train_A" ]]; then
  echo "ERROR: missing $DATA_DIR/train_A" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists" >&2
  exit 1
fi

WORKERS="${MASK_WORKERS:-12}"
tmux new-session -d -s "$SESSION" -c "$FROST_ROOT" bash -lc "
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate rafi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH='$FROST_ROOT/src:$VENDOR/src'\${PYTHONPATH:+:\$PYTHONPATH}
echo \"PRECOMPUTE_START \$(date -Iseconds) dataset=$DATASET workers=$WORKERS\" | tee -a '$LOG'
$PYTHON -u -m frost.precompute_granular_masks \
  --dataset_folder data/$DATASET \
  --mask_dir data/$DATASET/masks \
  --seed 42 \
  --workers $WORKERS \
  2>&1 | tee -a '$LOG'
echo \"PRECOMPUTE_END \$(date -Iseconds)\" | tee -a '$LOG'
"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Masks → $MASK_DIR"

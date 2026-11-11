#!/usr/bin/env bash
# BDD-only course primary: baseline → binary → fuzzy → fuzzy_rough.
# Skips runs that already have model_2000.pkl (>=1G). Aborts if a new run
# finishes without a large model_2000.pkl.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_DIR="$FROST_ROOT/configs/course"
LAUNCH="$FROST_ROOT/scripts/launch_train.sh"
DATA="$FROST_ROOT/data"
FRESH_DATE="${FRESH_DATE:-$(date +%Y%m%d)}"
FRESH_ROOT="${FRESH_ROOT:-$FROST_ROOT/output/fresh_${FRESH_DATE}}"
export FRESH_DATE FRESH_ROOT

# Pin tmux server under workspace — /tmp/tmux-UID gets wiped and false-aborts waits.
export TMUX_TMPDIR="${TMUX_TMPDIR:-$FROST_ROOT/.tmux}"
mkdir -p "$TMUX_TMPDIR"

MIN_PKL_BYTES=$((1024 * 1024 * 1024))  # 1GiB

pkl_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  local sz
  sz="$(stat -c '%s' "$p" 2>/dev/null || echo 0)"
  [[ "$sz" -ge "$MIN_PKL_BYTES" ]]
}

check_masks() {
  local ds="$1"
  local mask_dir="$DATA/$ds/masks"
  if [[ ! -d "$mask_dir" ]]; then
    echo "ERROR: masks missing at $mask_dir. Run: bash scripts/precompute_masks.sh $ds" >&2
    exit 1
  fi
  local n
  n="$(find "$mask_dir" -type f -name '*_mask_binary.npz' 2>/dev/null | wc -l)"
  if [[ "$n" -lt 10 ]]; then
    echo "ERROR: mask dir incomplete ($n binary masks). Precompute first." >&2
    exit 1
  fi
}

RUNS=(
  bdd100k_day_night:baseline:frost_bdd_baseline
  bdd100k_day_night:binary:frost_bdd_binary
  bdd100k_day_night:fuzzy:frost_bdd_fuzzy
  bdd100k_day_night:fuzzy_rough:frost_bdd_fuzzy_rough
)

check_masks "bdd100k_day_night"

for entry in "${RUNS[@]}"; do
  IFS=: read -r ds method session <<<"$entry"
  CFG="$CFG_DIR/${ds}_${method}.yaml"
  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: missing $CFG" >&2
    exit 1
  fi
  OUT="$FRESH_ROOT/$ds/$method"
  PKL="$OUT/checkpoints/model_2000.pkl"

  if pkl_ok "$PKL"; then
    echo "=== COURSE SKIP $ds / $method (existing model_2000.pkl $(stat -c '%s' "$PKL") bytes) ==="
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "ERROR: session $session already running; refuse overlap." >&2
    exit 1
  fi

  echo "=== COURSE LAUNCH $ds / $method ==="
  bash "$LAUNCH" "$CFG"
  echo "Waiting for tmux session $session to finish..."
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
  sync
  sleep 5
  if ! pkl_ok "$PKL"; then
    echo "ERROR: $session finished without usable $PKL — aborting remaining BDD runs." >&2
    exit 1
  fi
  echo "=== COURSE DONE $ds / $method (ok: model_2000.pkl) ==="
done

echo "ALL_COURSE_BDD_DONE $(date -Iseconds)"

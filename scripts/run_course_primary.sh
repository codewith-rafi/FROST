#!/usr/bin/env bash
# Run eight course trainings: Horse×4 then BDD×4.
# All eight runs share one FRESH_DATE folder, even if the sequence crosses midnight.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_DIR="$FROST_ROOT/configs/course"
LAUNCH="$FROST_ROOT/scripts/launch_train.sh"
DATA="$FROST_ROOT/data"
FRESH_DATE="${FRESH_DATE:-$(date +%Y%m%d)}"
FRESH_ROOT="${FRESH_ROOT:-$FROST_ROOT/output/fresh_${FRESH_DATE}}"
export FRESH_DATE FRESH_ROOT

export TMUX_TMPDIR="${TMUX_TMPDIR:-$FROST_ROOT/.tmux}"
mkdir -p "$TMUX_TMPDIR"

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
  horse2zebra:baseline:frost_h2z_baseline
  horse2zebra:binary:frost_h2z_binary
  horse2zebra:fuzzy:frost_h2z_fuzzy
  horse2zebra:fuzzy_rough:frost_h2z_fuzzy_rough
  bdd100k_day_night:baseline:frost_bdd_baseline
  bdd100k_day_night:binary:frost_bdd_binary
  bdd100k_day_night:fuzzy:frost_bdd_fuzzy
  bdd100k_day_night:fuzzy_rough:frost_bdd_fuzzy_rough
)

for entry in "${RUNS[@]}"; do
  IFS=: read -r ds method session <<<"$entry"
  if [[ "$method" != "baseline" ]]; then
    check_masks "$ds"
  fi
done

for entry in "${RUNS[@]}"; do
  IFS=: read -r ds method session <<<"$entry"
  CFG="$CFG_DIR/${ds}_${method}.yaml"
  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: missing $CFG" >&2
    exit 1
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
  echo "=== COURSE DONE $ds / $method ==="
done

echo "ALL_COURSE_PRIMARY_DONE $(date -Iseconds)"

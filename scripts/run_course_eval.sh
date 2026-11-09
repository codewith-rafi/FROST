#!/usr/bin/env bash
# Sequential offline FID+DINO eval for all 8 course runs (horse×4 then BDD×4).
# Uses resolved run config.yaml (not configs/course placeholders).
# Requires CUDA_VISIBLE_DEVICES=0: the final evaluation pass ran on GPU 0 while GPU 1 was occupied.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_LAUNCH="$FROST_ROOT/scripts/eval_checkpoint.sh"
PYTHON="${PYTHON:-/home/user0/.conda/envs/rafi/bin/python}"

export TMUX_TMPDIR="${TMUX_TMPDIR:-$FROST_ROOT/.tmux}"
mkdir -p "$TMUX_TMPDIR"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "ERROR: run_course_eval.sh hard-requires CUDA_VISIBLE_DEVICES=0 (got '${CUDA_VISIBLE_DEVICES:-unset}')." >&2
  exit 1
fi

MIN_PKL_BYTES=$((1024 * 1024 * 1024))

pkl_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  local sz
  sz="$(stat -c '%s' "$p" 2>/dev/null || echo 0)"
  [[ "$sz" -ge "$MIN_PKL_BYTES" ]]
}

metrics_ok() {
  local m="$1"
  [[ -f "$m" ]] || return 1
  "$PYTHON" - "$m" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
    a = d["a2b"]["fid"]
    b = d["b2a"]["fid"]
    assert isinstance(a, (int, float)) and isinstance(b, (int, float))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

# dataset:method:run_dir (relative to FROST_ROOT/output)
RUNS=(
  horse2zebra:baseline:fresh_20261024/horse2zebra/baseline
  horse2zebra:binary:fresh_20261024/horse2zebra/binary
  horse2zebra:fuzzy:fresh_20261024/horse2zebra/fuzzy
  horse2zebra:fuzzy_rough:fresh_20261024/horse2zebra/fuzzy_rough
  bdd100k_day_night:baseline:fresh_20261027/bdd100k_day_night/baseline
  bdd100k_day_night:binary:fresh_20261027/bdd100k_day_night/binary
  bdd100k_day_night:fuzzy:fresh_20261027/bdd100k_day_night/fuzzy
  bdd100k_day_night:fuzzy_rough:fresh_20261027/bdd100k_day_night/fuzzy_rough
)

echo "=== COURSE EVAL START $(date -Iseconds) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
"$PYTHON" -c "import sys; print(sys.executable)"
which python || true

for entry in "${RUNS[@]}"; do
  IFS=: read -r ds method rel <<<"$entry"
  OUT="$FROST_ROOT/output/$rel"
  CFG="$OUT/config.yaml"
  PKL="$OUT/checkpoints/model_2000.pkl"
  METRICS="$OUT/metrics.json"
  LOG="$OUT/eval.log"
  SESSION="frost_eval_${ds}_${method}"

  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: missing $CFG" >&2
    exit 1
  fi
  if ! pkl_ok "$PKL"; then
    echo "ERROR: missing/small checkpoint $PKL" >&2
    exit 1
  fi

  if metrics_ok "$METRICS"; then
    echo "=== COURSE EVAL SKIP $ds / $method (existing metrics.json) ==="
    continue
  fi

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: session $SESSION already running; refuse overlap." >&2
    exit 1
  fi

  echo "=== COURSE EVAL LAUNCH $ds / $method ==="
  echo "CFG=$CFG PKL=$PKL SESSION=$SESSION"
  bash "$EVAL_LAUNCH" "$CFG" "$PKL"

  echo "Waiting for tmux session $SESSION to finish..."
  while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 60
  done
  sync
  sleep 3

  if ! grep -q 'EVAL_END' "$LOG" 2>/dev/null; then
    echo "ERROR: $SESSION finished without EVAL_END in $LOG — aborting." >&2
    exit 1
  fi
  if ! metrics_ok "$METRICS"; then
    echo "ERROR: $SESSION finished without valid $METRICS — aborting." >&2
    exit 1
  fi
  echo "=== COURSE EVAL DONE $ds / $method ==="
  "$PYTHON" -c "import json; d=json.load(open('$METRICS')); print(f\"  fid_a2b={d['a2b']['fid']:.4f} fid_b2a={d['b2a']['fid']:.4f} dino_a2b={d['a2b']['dino_struct_mean']} dino_b2a={d['b2a']['dino_struct_mean']}\")"
done

echo "ALL_COURSE_EVAL_DONE $(date -Iseconds)"

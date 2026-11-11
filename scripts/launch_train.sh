#!/usr/bin/env bash
# Launch one frost train run from a YAML config (tmux + tee).
# Usage: bash scripts/launch_train.sh configs/course/horse2zebra_baseline.yaml
# Env overrides: PYTHON, ACCEL, CUDA_VISIBLE_DEVICES (default 0), HF_ENDPOINT,
#                FRESH_DATE / FRESH_ROOT, RESUME_FROM, MAIN_PROCESS_PORT.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?usage: launch_train.sh <config.yaml>}"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"

PYTHON="${PYTHON:-$(command -v python || true)}"
ACCEL="${ACCEL:-$(dirname "$PYTHON")/accelerate}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
FRESH_DATE="${FRESH_DATE:-$(date +%Y%m%d)}"
FRESH_ROOT="${FRESH_ROOT:-$FROST_ROOT/output/fresh_${FRESH_DATE}}"

# Pin tmux under workspace so /tmp cleanup cannot drop the server mid-run.
export TMUX_TMPDIR="${TMUX_TMPDIR:-$FROST_ROOT/.tmux}"
mkdir -p "$TMUX_TMPDIR"

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found ('$PYTHON'). Activate your env or set PYTHON=/path/to/python." >&2
  exit 1
fi
if [[ ! -x "$ACCEL" ]]; then
  echo "ERROR: accelerate not found at $ACCEL. Install it in the same env or set ACCEL=/path/to/accelerate." >&2
  exit 1
fi
HF_EXPORT=""
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  HF_EXPORT="export HF_ENDPOINT='$HF_ENDPOINT'"
fi

eval "$("$PYTHON" - <<PY
import ast
from pathlib import Path
text = Path("$CFG").read_text()
cfg = {}
for line in text.splitlines():
    line = line.split("#", 1)[0].rstrip()
    if not line or ":" not in line:
        continue
    k, v = line.split(":", 1)
    k, v = k.strip(), v.strip()
    if not k:
        continue
    if v in ("null", "Null", "~", ""):
        cfg[k] = None
    elif v.lower() in ("true", "false"):
        cfg[k] = v.lower() == "true"
    else:
        try:
            cfg[k] = ast.literal_eval(v)
        except Exception:
            cfg[k] = v
need = ["dataset", "method", "dataset_folder", "tmux_session", "max_train_steps", "seed"]
miss = [k for k in need if k not in cfg]
if miss:
    raise SystemExit(f"config missing keys: {miss}")
def sh(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
for k, v in cfg.items():
    print(f'CFG_{k.upper()}={repr(sh(v))}')
print(f'FRESH_ROOT={repr("$FRESH_ROOT")}')
print(f'DATASET={repr(cfg["dataset"])}')
print(f'METHOD={repr(cfg["method"])}')
PY
)"

OUT_DIR="${CFG_OUTPUT_DIR:-}"
if [[ -z "$OUT_DIR" || "$OUT_DIR" == *"experiments/"* || "$OUT_DIR" == *"OUTPUT_DIR_PLACEHOLDER"* ]]; then
  OUT_DIR="$FRESH_ROOT/$DATASET/$METHOD"
fi
SESSION="$CFG_TMUX_SESSION"
mkdir -p "$OUT_DIR/checkpoints"

RESUME_FROM="${RESUME_FROM:-}"
shopt -s nullglob
existing=("$OUT_DIR"/checkpoints/*.pkl)
shopt -u nullglob
if [[ -n "$RESUME_FROM" ]]; then
  if [[ ! -f "$RESUME_FROM" ]]; then
    echo "ERROR: RESUME_FROM not found: $RESUME_FROM" >&2
    exit 1
  fi
  if [[ -f "$OUT_DIR/checkpoints/model_2000.pkl" ]]; then
    echo "ERROR: $OUT_DIR/checkpoints/model_2000.pkl already exists; refuse resume." >&2
    exit 1
  fi
  echo "RESUME_FROM=$RESUME_FROM (allowing nonempty checkpoint dir)"
elif ((${#existing[@]} > 0)); then
  # Refuse reuse of nonempty checkpoint dirs (fresh runs only)
  echo "ERROR: refuse nonempty checkpoint dir: $OUT_DIR/checkpoints (${#existing[@]} pkl). Use a new FRESH_DATE, delete, or set RESUME_FROM." >&2
  exit 1
fi

cp -f "$CFG" "$OUT_DIR/config.yaml"
# rewrite snapshot output_dir to actual
"$PYTHON" - <<PY
from pathlib import Path
p = Path("$OUT_DIR/config.yaml")
lines = []
for line in p.read_text().splitlines():
    if line.startswith("output_dir:"):
        lines.append("output_dir: $OUT_DIR")
    else:
        lines.append(line)
p.write_text("\\n".join(lines) + "\\n")
PY

VENDOR="$FROST_ROOT/vendor/img2img-turbo"
VENDOR_SRC="$VENDOR/src"
FROST_SRC="$FROST_ROOT/src"
if [[ ! -d "$VENDOR_SRC" ]]; then
  echo "ERROR: missing $VENDOR_SRC" >&2
  exit 1
fi

ARGS=(
  -m frost.train_cyclegan_turbo
  --pretrained_model_name_or_path "${CFG_PRETRAINED_MODEL_NAME_OR_PATH:-stabilityai/sd-turbo}"
  --output_dir "$OUT_DIR"
  --dataset_folder "$CFG_DATASET_FOLDER"
  --train_img_prep "${CFG_TRAIN_IMG_PREP:-resize_286_randomcrop_256x256_hflip}"
  --val_img_prep "${CFG_VAL_IMG_PREP:-no_resize}"
  --learning_rate "${CFG_LEARNING_RATE:-1e-5}"
  --max_train_steps "$CFG_MAX_TRAIN_STEPS"
  --max_train_epochs "${CFG_MAX_TRAIN_EPOCHS:-1}"
  --train_batch_size "${CFG_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${CFG_GRADIENT_ACCUMULATION_STEPS:-1}"
  --seed "$CFG_SEED"
  --report_to "${CFG_REPORT_TO:-wandb}"
  --tracker_project_name "${CFG_TRACKER_PROJECT_NAME:-granular_cycle_turbo}"
  --validation_steps "${CFG_VALIDATION_STEPS:-2000}"
  --checkpointing_steps "${CFG_CHECKPOINTING_STEPS:-500}"
  --validation_num_images "${CFG_VALIDATION_NUM_IMAGES:-0}"
  --viz_freq "${CFG_VIZ_FREQ:-0}"
  --dataloader_num_workers "${CFG_DATALOADER_NUM_WORKERS:-8}"
  --lambda_gan "${CFG_LAMBDA_GAN:-0.5}"
  --lambda_idt "${CFG_LAMBDA_IDT:-1}"
  --lambda_cycle "${CFG_LAMBDA_CYCLE:-1}"
  --lambda_cycle_lpips "${CFG_LAMBDA_CYCLE_LPIPS:-10}"
  --lambda_idt_lpips "${CFG_LAMBDA_IDT_LPIPS:-1}"
  --granular_mode "${CFG_GRANULAR_MODE:-none}"
  --lambda_granular "${CFG_LAMBDA_GRANULAR:-0}"
  --lora_rank_unet "${CFG_LORA_RANK_UNET:-128}"
  --lora_rank_vae "${CFG_LORA_RANK_VAE:-4}"
)

if [[ "${CFG_ALLOW_TF32:-true}" == "true" ]]; then
  ARGS+=(--allow_tf32)
fi
if [[ "${CFG_ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION:-false}" == "true" ]]; then
  ARGS+=(--enable_xformers_memory_efficient_attention)
fi
if [[ -n "${CFG_GRANULAR_MASK_DIR:-}" && "${CFG_GRANULAR_MASK_DIR}" != "null" && "${CFG_GRANULAR_MODE}" != "none" ]]; then
  ARGS+=(--granular_mask_dir "$CFG_GRANULAR_MASK_DIR")
fi
if [[ -n "$RESUME_FROM" ]]; then
  ARGS+=(--resume_from "$RESUME_FROM")
fi

CMD_STR="$ACCEL launch --main_process_port ${MAIN_PROCESS_PORT:-29501} ${ARGS[*]}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists. Attach or kill it first." >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$FROST_ROOT" bash -lc "
set -euo pipefail
export PATH='$(dirname "$PYTHON")':\$PATH
export CUDA_VISIBLE_DEVICES='$GPU'
export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
$HF_EXPORT
export PYTHONUNBUFFERED=1
export PYTHONPATH='$FROST_SRC:$VENDOR_SRC'\${PYTHONPATH:+:\$PYTHONPATH}
which python | tee -a '$OUT_DIR/train.log'
nvidia-smi -L | tee -a '$OUT_DIR/train.log'
echo \"TRAIN_CMD $CMD_STR\" | tee -a '$OUT_DIR/train.log'
echo \"CONFIG $CFG\" | tee -a '$OUT_DIR/train.log'
echo \"OUTPUT_DIR $OUT_DIR\" | tee -a '$OUT_DIR/train.log'
echo \"LAUNCH_START \$(date -Iseconds) method=$METHOD steps=$CFG_MAX_TRAIN_STEPS\" | tee -a '$OUT_DIR/train.log'
$ACCEL launch --main_process_port ${MAIN_PROCESS_PORT:-29501} ${ARGS[*]} 2>&1 | tee -a '$OUT_DIR/train.log'
echo \"LAUNCH_END \$(date -Iseconds)\" | tee -a '$OUT_DIR/train.log'
"

echo "Started tmux session: $SESSION"
echo "Output: $OUT_DIR"
echo "Log: $OUT_DIR/train.log"
echo "Attach: tmux attach -t $SESSION"
echo "CUDA_VISIBLE_DEVICES=$GPU"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
PYTHON_BIN=${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3/bin/python}
MODEL_PATH=${MODEL_PATH:-$ROOT}

TEST_DATA=${TEST_DATA:-/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-radvqa/3drad_test123456_1.jsonl}
IMAGE_BASE_DIR=${IMAGE_BASE_DIR:-/volume/med-train/data/CT-RATE-DATA/dataset}

TRAINED_WEIGHTS=${TRAINED_WEIGHTS:-/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/work_dir/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate/20260529_m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate_lora}
MODEL_NAME=${MODEL_NAME:-m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate_lora}

GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_SLICES=${NUM_SLICES:-8}
BATCH_SIZE=${BATCH_SIZE:-96}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
NUM_SAMPLES=${NUM_SAMPLES:--1}
DTYPE=${DTYPE:-bfloat16}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}
OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/results/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate_full/inference_results.jsonl}
SKIP_AUTO_METRIC=${SKIP_AUTO_METRIC:-0}

mkdir -p "$(dirname "$OUTPUT_PATH")"

NEED_REMAP=$("$PYTHON_BIN" - <<'PY' "$TRAINED_WEIGHTS"
import sys
from pathlib import Path
from safetensors.torch import load_file

adapter_dir = Path(sys.argv[1])
adapter_file = adapter_dir / "adapter_model.safetensors"
if not adapter_file.exists():
    raise SystemExit(f"Missing adapter weights: {adapter_file}")
state = load_file(str(adapter_file), device="cpu")
need = any(k.startswith("base_model.model.base_model.model.") for k in state.keys())
print("1" if need else "0")
PY
)

RUN_WEIGHTS="$TRAINED_WEIGHTS"
RUN_MODEL_NAME="$MODEL_NAME"
if [ "$NEED_REMAP" = "1" ]; then
  REMAP_DIR="${TRAINED_WEIGHTS}_autoremapped"
  "$PYTHON_BIN" "$ROOT/remap_lora_checkpoint_keys.py" \
    --src "$TRAINED_WEIGHTS" \
    --dst "$REMAP_DIR" \
    --force
  RUN_WEIGHTS="$REMAP_DIR"
  RUN_MODEL_NAME="${MODEL_NAME}_autoremapped"
fi

"$PYTHON_BIN" "$ROOT/inference_compare_9_v15.py" \
  --model_path "$MODEL_PATH" \
  --data_path "$TEST_DATA" \
  --trained_weights "$RUN_WEIGHTS" \
  --model_names "$RUN_MODEL_NAME" \
  --image_base_dir "$IMAGE_BASE_DIR" \
  --output_path "$OUTPUT_PATH" \
  --gpu_ids "$GPU_IDS" \
  --num_slices "$NUM_SLICES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples "$NUM_SAMPLES" \
  --dtype "$DTYPE" \
  $( [ "$USE_FLASH_ATTN" = "1" ] && echo "--use_flash_attn" ) \
  $( [ "$SKIP_AUTO_METRIC" = "1" ] && echo "--skip_auto_metric" )

echo "Full eval done."
echo "Test data: $TEST_DATA"
echo "Adapter: $RUN_WEIGHTS"
echo "Output: $OUTPUT_PATH"

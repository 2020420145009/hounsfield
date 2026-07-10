#!/usr/bin/env bash
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
PYTHON_BIN=${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3/bin/python}
MODEL_PATH=${MODEL_PATH:-$ROOT}

TRAIN_DATA=${TRAIN_DATA:-$ROOT/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl}
TEST_DATA=${TEST_DATA:-/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-radvqa/3drad_test123456_1.jsonl}
IMAGE_BASE_DIR=${IMAGE_BASE_DIR:-/volume/med-train/data/CT-RATE-DATA/dataset}

SMOKE_DIR=${SMOKE_DIR:-$ROOT/smoke_datasets/m3d_ctrate_lidc_mix_v1_open_only_test}
TRAIN_SUBSET=${TRAIN_SUBSET:-$SMOKE_DIR/train_smoke.jsonl}
TEST_SUBSET=${TEST_SUBSET:-$SMOKE_DIR/test_smoke.jsonl}

TRAIN_LIMIT=${TRAIN_LIMIT:-64}
TEST_LIMIT=${TEST_LIMIT:-32}

GPU_IDS=${GPU_IDS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

SAVE_ROOT=${SAVE_ROOT:-$ROOT/work_dir/m3d_ctrate_lidc_mix_v1_smoke}
MODEL_NAME=${MODEL_NAME:-m3d_ctrate_lidc_mix_v1_smoke_lora}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-$ROOT/results/m3d_ctrate_lidc_mix_v1_smoke}
EVAL_OUT_BASE=${EVAL_OUT_BASE:-$EVAL_OUT_DIR/inference_results_smoke.jsonl}

NUM_SLICES=${NUM_SLICES:-8}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-5}
DTYPE=${DTYPE:-bfloat16}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-1024}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:--1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
EXISTING_LORA_DIR=${EXISTING_LORA_DIR:-}

REMAP_SCRIPT="$ROOT/remap_lora_checkpoint_keys.py"

mkdir -p "$SMOKE_DIR" "$SAVE_ROOT" "$EVAL_OUT_DIR"

"$PYTHON_BIN" - <<'PY' "$TRAIN_DATA" "$TEST_DATA" "$TRAIN_SUBSET" "$TEST_SUBSET" "$TRAIN_LIMIT" "$TEST_LIMIT" "$SKIP_TRAIN"
import json
import sys

train_src, test_src, train_dst, test_dst, train_limit, test_limit, skip_train = sys.argv[1:]
train_limit = int(train_limit)
test_limit = int(test_limit)
skip_train = int(skip_train)

def copy_head(src, dst, limit):
    count = 0
    with open(src, 'r', encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= limit:
                break
    return count

train_count = 0
if not skip_train:
    train_count = copy_head(train_src, train_dst, train_limit)
test_count = copy_head(test_src, test_dst, test_limit)
print({
    "train_subset": train_dst,
    "train_count": train_count,
    "test_subset": test_dst,
    "test_count": test_count,
    "skip_train": bool(skip_train),
})
PY

if [ "$SKIP_TRAIN" != "1" ]; then
  torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/finetune_3d_strict_cot_prompt_onlylora_v15.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$TRAIN_SUBSET" \
    --save_dir "$SAVE_ROOT" \
    --model_name "$MODEL_NAME" \
    --gpu_ids "$GPU_IDS" \
    --num_slices "$NUM_SLICES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --dtype "$DTYPE" \
    --use_llm_lora 1 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    $( [ "$USE_FLASH_ATTN" = "1" ] && echo "--use_flash_attn" )

  LATEST_ADAPTER=$(find "$SAVE_ROOT" -maxdepth 1 -mindepth 1 -type d -name "*_${MODEL_NAME}" | sort | tail -n 1)
  if [ -z "${LATEST_ADAPTER:-}" ]; then
    echo "Failed to find latest adapter under $SAVE_ROOT" >&2
    exit 1
  fi
else
  if [ -z "$EXISTING_LORA_DIR" ]; then
    echo "SKIP_TRAIN=1 requires EXISTING_LORA_DIR" >&2
    exit 1
  fi
  LATEST_ADAPTER="$EXISTING_LORA_DIR"
fi

RUN_LORA_DIR="$LATEST_ADAPTER"
RUN_MODEL_NAME="$MODEL_NAME"

NEED_REMAP=$("$PYTHON_BIN" - <<'PY' "$LATEST_ADAPTER"
import sys
from safetensors.torch import load_file

path = sys.argv[1] + "/adapter_model.safetensors"
state = load_file(path, device="cpu")
need = any(k.startswith("base_model.model.base_model.model.") for k in state.keys())
print("1" if need else "0")
PY
)

if [ "$NEED_REMAP" = "1" ]; then
  REMAP_DIR="${LATEST_ADAPTER}_autoremapped"
  "$PYTHON_BIN" "$REMAP_SCRIPT" \
    --src "$LATEST_ADAPTER" \
    --dst "$REMAP_DIR" \
    --force
  RUN_LORA_DIR="$REMAP_DIR"
  RUN_MODEL_NAME="${MODEL_NAME}_autoremapped"
fi

"$PYTHON_BIN" "$ROOT/inference_compare_9_v15.py" \
  --model_path "$MODEL_PATH" \
  --data_path "$TEST_SUBSET" \
  --trained_weights "$RUN_LORA_DIR" \
  --model_names "$RUN_MODEL_NAME" \
  --image_base_dir "$IMAGE_BASE_DIR" \
  --output_path "$EVAL_OUT_BASE" \
  --gpu_ids "$GPU_IDS" \
  --num_slices "$NUM_SLICES" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$EVAL_NUM_WORKERS" \
  --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --debug_print_every 1 \
  $( [ "$USE_FLASH_ATTN" = "1" ] && echo "--use_flash_attn" )

echo "Smoke test done."
echo "Train subset: $TRAIN_SUBSET"
echo "Test subset: $TEST_SUBSET"
echo "Latest adapter: $RUN_LORA_DIR"
echo "Eval outputs dir: $EVAL_OUT_DIR"

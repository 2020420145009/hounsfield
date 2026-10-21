#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
MODEL_PATH="$ROOT"
TRAIN_DATA="$ROOT/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl"
TEST_DATA="$ROOT/mixed_datasets_rebalanced/m3dtest_like_v2/m3dtest_like_mcq_plus_open_v2.jsonl"

# `image_base_dir` is only used by inference_compare_9_v15.py to build a
# basename->fullpath index. Absolute paths in TEST_DATA still work directly.
IMAGE_BASE_DIR=/volume/med-train/data/CT-RATE-DATA/dataset

GPU_IDS=0,1,2,3
NPROC_PER_NODE=4

SAVE_ROOT="$ROOT/work_dir/m3d_ctrate_lidc_mix_v1"
MODEL_NAME=m3d_ctrate_lidc_mix_v1_lora
EVAL_OUT_DIR="$ROOT/results/m3d_ctrate_lidc_mix_v1"
EVAL_OUT_BASE="$EVAL_OUT_DIR/inference_results.jsonl"

# Finetune hyperparameters
NUM_SLICES=8
EPOCHS=2
BATCH_SIZE=4
GRAD_ACCUM=8
LR=1e-5
DTYPE=bfloat16
USE_FLASH_ATTN=1
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Eval hyperparameters
EVAL_BATCH_SIZE=16
EVAL_NUM_WORKERS=8
EVAL_MAX_NEW_TOKENS=1024
EVAL_NUM_SAMPLES=-1

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
mkdir -p "$SAVE_ROOT" "$EVAL_OUT_DIR"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/finetune_3d_strict_cot_prompt_onlylora_v15.py" \
  --model_path "$MODEL_PATH" \
  --data_path "$TRAIN_DATA" \
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

# ------------------------------------------------------------------
# Find latest adapter
# ------------------------------------------------------------------
LATEST_ADAPTER=$(find "$SAVE_ROOT" -maxdepth 1 -mindepth 1 -type d -name "*_${MODEL_NAME}" | sort | tail -n 1)
if [ -z "${LATEST_ADAPTER:-}" ]; then
  echo "Failed to find latest adapter under $SAVE_ROOT" >&2
  exit 1
fi

echo "Using latest adapter: $LATEST_ADAPTER"

# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------
python "$ROOT/inference_compare_9_v15.py" \
  --model_path "$MODEL_PATH" \
  --data_path "$TEST_DATA" \
  --trained_weights "$LATEST_ADAPTER" \
  --model_names "$MODEL_NAME" \
  --image_base_dir "$IMAGE_BASE_DIR" \
  --output_path "$EVAL_OUT_BASE" \
  --gpu_ids "$GPU_IDS" \
  --num_slices "$NUM_SLICES" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$EVAL_NUM_WORKERS" \
  --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  $( [ "$USE_FLASH_ATTN" = "1" ] && echo "--use_flash_attn" )

echo "Done."
echo "Train data: $TRAIN_DATA"
echo "Latest adapter: $LATEST_ADAPTER"
echo "Eval outputs dir: $EVAL_OUT_DIR"

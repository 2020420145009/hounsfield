#!/usr/bin/env bash
source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SCRIPT=$ROOT/script/synthesis/synthesize_dataset.py

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/training_data/ctrate_open/v500}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
TP=${TP:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
TEST_LIMIT=${TEST_LIMIT:-8000}
TARGET_COUNT=${TARGET_COUNT:-2000}

python "$SCRIPT" \
  synthesize-ctrate-open \
  --output-dir "$OUTPUT_DIR" \
  --gpu-devices "$GPU_DEVICES" \
  --tensor-parallel-size "$TP" \
  --batch-size "$BATCH_SIZE" \
  --test-limit "$TEST_LIMIT" \
  --target-count "$TARGET_COUNT"

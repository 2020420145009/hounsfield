#!/usr/bin/env bash
source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SCRIPT=$ROOT/synthesize_until_target_local_qwen72b.py

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/training_data/m3d_mcq/v500}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
TP=${TP:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
TARGET_COUNT=${TARGET_COUNT:-2000}

python "$SCRIPT" \
  --mode m3d_mcq \
  --output-dir "$OUTPUT_DIR" \
  --target-count "$TARGET_COUNT" \
  --gpu-devices "$GPU_DEVICES" \
  --tensor-parallel-size "$TP" \
  --batch-size "$BATCH_SIZE"

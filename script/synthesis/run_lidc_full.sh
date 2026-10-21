#!/usr/bin/env bash
source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SCRIPT=$ROOT/synthesize_advanced_cot_all_in_one.py

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/training_data/lidc_mixed/v003}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
TP=${TP:-4}
BATCH_SIZE=${BATCH_SIZE:-6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_TOKENS=${MAX_TOKENS:-4096}
MAX_OPEN_PER_CASE=${MAX_OPEN_PER_CASE:-2}
MAX_MCQ_PER_CASE=${MAX_MCQ_PER_CASE:-2}
MAX_REPORT_PER_CASE=${MAX_REPORT_PER_CASE:-1}
MAX_TOTAL_ITEMS_PER_CASE=${MAX_TOTAL_ITEMS_PER_CASE:-5}

python "$SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-devices "$GPU_DEVICES" \
  --tensor-parallel-size "$TP" \
  --batch-size "$BATCH_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --max-open-per-case "$MAX_OPEN_PER_CASE" \
  --max-mcq-per-case "$MAX_MCQ_PER_CASE" \
  --max-report-per-case "$MAX_REPORT_PER_CASE" \
  --max-total-items-per-case "$MAX_TOTAL_ITEMS_PER_CASE"

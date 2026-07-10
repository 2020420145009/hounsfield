#!/bin/bash
set -euo pipefail

ROOT_DIR="/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d"
MODEL_NAME="${1:-}"

if [ -z "${MODEL_NAME}" ]; then
  echo "用法: bash run_pipeline_lidc_only_v15.sh <model_name>"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/internvl3/bin/python}"
DATA_JSONL="${DATA_JSONL:-${ROOT_DIR}/lidc_only_dataset_v2_strictcot/lidc_only_mcq_open_v2_strictcot.jsonl}"

if [ ! -f "${DATA_JSONL}" ]; then
    echo "未找到数据集: ${DATA_JSONL}"
    echo "先执行: ${PYTHON_BIN} ${ROOT_DIR}/build_lidc_only_dataset_v2_strictcot.py"
    exit 1
fi

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-32}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-8}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-8}"

TRAIN_DATA="${DATA_JSONL}" \
TRAIN_GPU_IDS="${TRAIN_GPU_IDS}" \
TRAIN_NPROC="${TRAIN_NPROC}" \
EVAL_GPU_IDS="${EVAL_GPU_IDS}" \
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS}" \
FULL_BATCH_SIZE="${FULL_BATCH_SIZE}" \
FULL_NUM_WORKERS="${FULL_NUM_WORKERS}" \
bash "${ROOT_DIR}/run_pipeline_m3dtest_like_v2_v15.sh" "${MODEL_NAME}"

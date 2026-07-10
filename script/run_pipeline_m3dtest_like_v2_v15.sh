#!/bin/bash
set -euo pipefail

# 用法:
#   bash run_pipeline_m3dtest_like_v2_v15.sh <model_name>
#
# 默认流程:
#   1. 用 v12 数据集训练 LoRA（沿用 v11 的 prompt、batch 和筛选策略，只替换 cleaned CTRATE）
#   2. 自动检查并 remap LoRA key 前缀
#   3. 先跑 smoke test
#   4. 只有显式 RUN_FULL=1 时才继续跑全量

MODEL_NAME=${1:-}
if [ -z "${MODEL_NAME}" ]; then
    echo "用法: bash run_pipeline_m3dtest_like_v2_v15.sh <model_name>"
    exit 1
fi

ROOT_DIR="/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d"
PYTHON_BIN="${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/internvl3/bin/python}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/mixed_datasets_rebalanced/m3dtest_like_v2/m3dtest_like_mcq_plus_open_v2.jsonl}"
TEST_DATA="${TEST_DATA:-/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-radvqa/3drad_test123456_1.jsonl}"
IMAGE_BASE_DIR="${IMAGE_BASE_DIR:-/volume/med-train/users/shuyan/fzy/3d/datasets}"

WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_dir}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/${MODEL_NAME}}"

TRAIN_SCRIPT="${ROOT_DIR}/finetune_3d_strict_cot_prompt_onlylora_v15.py"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/inference_compare_9_v15.py}"
REMAP_SCRIPT="${ROOT_DIR}/remap_lora_checkpoint_keys.py"

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0}"

NUM_SLICES="${NUM_SLICES:-8}"
EPOCHS="${EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LR="${LR:-1e-5}"
GEM_ALPHA="${GEM_ALPHA:-0.1}"

SMOKE_NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-32}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-4096}"
SMOKE_MCQ_MAX_NEW_TOKENS="${SMOKE_MCQ_MAX_NEW_TOKENS:-8}"

FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"
FULL_MAX_NEW_TOKENS="${FULL_MAX_NEW_TOKENS:-4096}"
FULL_MCQ_MAX_NEW_TOKENS="${FULL_MCQ_MAX_NEW_TOKENS:-8}"

RUN_FULL="${RUN_FULL:-0}"
AUTO_REMAP_LORA="${AUTO_REMAP_LORA:-1}"
EVAL_BASELINE="${EVAL_BASELINE:-0}"
DEBUG_PRINT_EVERY="${DEBUG_PRINT_EVERY:-0}"
SLICE_CACHE_MODE="${SLICE_CACHE_MODE:-auto}"
PRECOMPUTE_SLICE_CACHE="${PRECOMPUTE_SLICE_CACHE:-0}"
PRECOMPUTE_SLICE_CACHE_LIMIT="${PRECOMPUTE_SLICE_CACHE_LIMIT:--1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EXISTING_LORA_DIR="${EXISTING_LORA_DIR:-}"

mkdir -p "${WORK_DIR}" "${RESULT_DIR}"

if [ "${SKIP_TRAIN}" = "1" ]; then
    if [ -z "${EXISTING_LORA_DIR}" ]; then
        echo "SKIP_TRAIN=1 时必须提供 EXISTING_LORA_DIR"
        exit 1
    fi
    LATEST_LORA_DIR="${EXISTING_LORA_DIR}"
    echo "======================================================="
    echo "[1/4] Skip training and reuse existing LoRA"
    echo "======================================================="
    echo "复用 LoRA: ${LATEST_LORA_DIR}"
else
    echo "======================================================="
    echo "[1/4] Training on M3D-test-like v2 with LLM LoRA only + CT slice description prompt (v15)"
    echo "======================================================="
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "TRAIN_DATA=${TRAIN_DATA}"
    echo "TRAIN_GPU_IDS=${TRAIN_GPU_IDS}"

    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" \
    torchrun --nproc_per_node="${TRAIN_NPROC}" "${TRAIN_SCRIPT}" \
        --model_path "${BASE_MODEL}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TRAIN_DATA}" \
        --save_dir "${WORK_DIR}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --num_slices "${NUM_SLICES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
        --lr "${LR}" \
        --gem_alpha "${GEM_ALPHA}" \
        --use_flash_attn \
        --debug_print_every "${DEBUG_PRINT_EVERY}"

    LATEST_LORA_DIR=$(ls -td "${WORK_DIR}"/*_"${MODEL_NAME}" | head -1)
    if [ -z "${LATEST_LORA_DIR}" ]; then
        echo "未找到训练产出的 LoRA 目录"
        exit 1
    fi

    echo "训练完成，LoRA 目录: ${LATEST_LORA_DIR}"
fi

echo "======================================================="
echo "[2/4] Auto-remap LoRA keys if needed"
echo "======================================================="

RUN_LORA_DIR="${LATEST_LORA_DIR}"
RUN_MODEL_NAME="${MODEL_NAME}"

if [ "${AUTO_REMAP_LORA}" = "1" ]; then
    NEED_REMAP=$("${PYTHON_BIN}" - <<'PY' "${LATEST_LORA_DIR}"
import sys
from safetensors.torch import load_file
path = sys.argv[1] + "/adapter_model.safetensors"
state = load_file(path, device="cpu")
need = any(k.startswith("base_model.model.base_model.model.") for k in state.keys())
print("1" if need else "0")
PY
)

    if [ "${NEED_REMAP}" = "1" ]; then
        REMAP_DIR="${LATEST_LORA_DIR}_autoremapped"
        "${PYTHON_BIN}" "${REMAP_SCRIPT}" \
            --src "${LATEST_LORA_DIR}" \
            --dst "${REMAP_DIR}" \
            --force
        RUN_LORA_DIR="${REMAP_DIR}"
        RUN_MODEL_NAME="${MODEL_NAME}_autoremapped"
        echo "LoRA 已 remap 到: ${RUN_LORA_DIR}"
    else
        echo "LoRA key 前缀正常，无需 remap"
    fi
else
    echo "AUTO_REMAP_LORA=0，跳过 remap"
fi

echo "======================================================="
echo "[3/4] Smoke evaluation"
echo "======================================================="

SMOKE_OUTPUT_BASE="${RESULT_DIR}/${MODEL_NAME}_smoke"
BASELINE_FLAG=""
if [ "${EVAL_BASELINE}" = "1" ]; then
    BASELINE_FLAG="--eval_baseline"
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU_IDS}" \
"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --model_path "${BASE_MODEL}" \
    --data_path "${TEST_DATA}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --trained_weights "${RUN_LORA_DIR}" \
    --model_names "${RUN_MODEL_NAME}" \
    --output_path "${SMOKE_OUTPUT_BASE}" \
    --num_slices "${NUM_SLICES}" \
    --num_samples "${SMOKE_NUM_SAMPLES}" \
    --batch_size "${SMOKE_BATCH_SIZE}" \
    --num_workers "${SMOKE_NUM_WORKERS}" \
    --max_new_tokens "${SMOKE_MAX_NEW_TOKENS}" \
    --use_flash_attn \
    --gpu_ids "${EVAL_GPU_IDS}" \
    --debug_print_every 1 \
    ${BASELINE_FLAG}

echo "Smoke 结果前缀: ${SMOKE_OUTPUT_BASE}"

if [ "${RUN_FULL}" != "1" ]; then
    echo "RUN_FULL!=1，已停止在 smoke test。"
    echo "如果 smoke 结果正常，再执行:"
    echo "SKIP_TRAIN=1 RUN_FULL=1 EXISTING_LORA_DIR=${LATEST_LORA_DIR} bash run_pipeline_m3dtest_like_v2_v15.sh ${MODEL_NAME}"
    exit 0
fi

echo "======================================================="
echo "[4/4] Full evaluation"
echo "======================================================="

FULL_OUTPUT_BASE="${RESULT_DIR}/${MODEL_NAME}_full"

CUDA_VISIBLE_DEVICES="${EVAL_GPU_IDS}" \
"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --model_path "${BASE_MODEL}" \
    --data_path "${TEST_DATA}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --trained_weights "${RUN_LORA_DIR}" \
    --model_names "${RUN_MODEL_NAME}" \
    --output_path "${FULL_OUTPUT_BASE}" \
    --num_slices "${NUM_SLICES}" \
    --num_samples -1 \
    --batch_size "${FULL_BATCH_SIZE}" \
    --num_workers "${FULL_NUM_WORKERS}" \
    --max_new_tokens "${FULL_MAX_NEW_TOKENS}" \
    --use_flash_attn \
    --gpu_ids "${EVAL_GPU_IDS}" \
    --debug_print_every 0 \
    ${BASELINE_FLAG}

echo "全量结果前缀: ${FULL_OUTPUT_BASE}"

#!/usr/bin/env bash
# Pipeline: Lingshu-7B LoRA fine-tuning on m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate,
# followed by 3D-RAD task4 benchmark evaluation.

source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

MODEL_NAME=${1:-lingshu_m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate_lora}

ROOT_DIR="/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d"
PYTHON_BIN="${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3/bin/torchrun}"
BASE_MODEL="${BASE_MODEL:-/volume/med-train/users/shuyan/Lingshu-7B}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD}"
BENCH_ENV="${BENCH_ENV:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate.jsonl}"

WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_dir/lingshu_m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate}"
BENCH_RESULT_ROOT="${BENCH_RESULT_ROOT:-${BENCHMARK_ROOT}/runtime/${MODEL_NAME}/results}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${BENCHMARK_ROOT}/runtime/${MODEL_NAME}_comparison}"

TRAIN_SCRIPT="${ROOT_DIR}/finetune_lingshu_lora_v1.py"
BENCH_RUNNER="${BENCHMARK_ROOT}/scripts/run_lingshu_3drad.sh"
EXPORT_TASK4="${BENCHMARK_ROOT}/scripts/export_task4_binary_accuracy.py"
SUMMARIZE_ALL="${BENCHMARK_ROOT}/scripts/summarize_all_results.py"

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
BENCH_GPU_IDS="${BENCH_GPU_IDS:-0,1,2,3}"

NUM_SLICES="${NUM_SLICES:-8}"
EPOCHS="${EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LR="${LR:-1e-5}"
DTYPE="${DTYPE:-bfloat16}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
EXISTING_LORA_DIR="${EXISTING_LORA_DIR:-}"
THREED_RAD_BATCH_SIZE="${THREED_RAD_BATCH_SIZE:-32}"
THREED_RAD_NUM_SLICES="${THREED_RAD_NUM_SLICES:-8}"

mkdir -p "${WORK_DIR}" "${COMPARE_OUTPUT_DIR}"

# ── [1/3] Training ─────────────────────────────────────────────────────────────
if [ "${SKIP_TRAIN}" = "1" ]; then
    if [ -z "${EXISTING_LORA_DIR}" ]; then
        echo "SKIP_TRAIN=1 时必须提供 EXISTING_LORA_DIR" >&2
        exit 1
    fi
    LORA_DIR="${EXISTING_LORA_DIR}"
    echo "======================================================="
    echo "[1/3] 跳过训练，复用 LoRA: ${LORA_DIR}"
    echo "======================================================="
else
    echo "======================================================="
    echo "[1/3] Lingshu-7B LoRA 训练 (${TRAIN_NPROC} GPU)"
    echo "======================================================="
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "TRAIN_DATA=${TRAIN_DATA}"

    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" \
    "${TORCHRUN_BIN}" --nproc_per_node="${TRAIN_NPROC}" "${TRAIN_SCRIPT}" \
        --model_path "${BASE_MODEL}" \
        --data_path "${TRAIN_DATA}" \
        --save_dir "${WORK_DIR}" \
        --model_name "${MODEL_NAME}" \
        --num_slices "${NUM_SLICES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
        --lr "${LR}" \
        --dtype "${DTYPE}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --gpu_ids "${TRAIN_GPU_IDS}" \
        --use_flash_attn \
        --slice_cache_mode auto

    LORA_DIR=$(find "${WORK_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${MODEL_NAME}" | sort | tail -n 1)
    if [ -z "${LORA_DIR}" ]; then
        echo "未找到训练产出的 LoRA 目录" >&2
        exit 1
    fi
    echo "训练完成，LoRA 目录: ${LORA_DIR}"
fi

# ── [2/3] 3D-RAD task4 benchmark ──────────────────────────────────────────────
echo "======================================================="
echo "[2/3] 3D-RAD task4 benchmark"
echo "======================================================="

export CUDA_VISIBLE_DEVICES="${BENCH_GPU_IDS}"
export THREED_RAD_BATCH_SIZE="${THREED_RAD_BATCH_SIZE}"
export THREED_RAD_NUM_SLICES="${THREED_RAD_NUM_SLICES}"

bash "${BENCH_RUNNER}" \
    "${BENCH_ENV}" \
    "${BASE_MODEL}" \
    "${BENCHMARK_ROOT}" \
    task4 \
    "${BENCH_RESULT_ROOT}" \
    "${LORA_DIR}"

# ── [3/3] Export comparison table ─────────────────────────────────────────────
echo "======================================================="
echo "[3/3] 导出 task4 对比表"
echo "======================================================="

"${PYTHON_BIN}" "${EXPORT_TASK4}" \
    --model-result m3d          "${BENCHMARK_ROOT}/runtime/m3d_3drad/results" \
    --model-result fleming_base "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad/results" \
    --model-result fleming_lora "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad_lora/results" \
    --model-result lingshu      "${BENCHMARK_ROOT}/runtime/lingshu7b_3drad/results" \
    --model-result "${MODEL_NAME}" "${BENCH_RESULT_ROOT}" \
    --output "${COMPARE_OUTPUT_DIR}/task4_binary_accuracy_comparison.csv"

echo "======================================================="
echo "Pipeline 完成。"
echo "LoRA 目录:          ${LORA_DIR}"
echo "Benchmark 结果:     ${BENCH_RESULT_ROOT}"
echo "对比表:             ${COMPARE_OUTPUT_DIR}/task4_binary_accuracy_comparison.csv"
echo ""
echo "如需跳过训练直接重跑评测："
echo "  SKIP_TRAIN=1 EXISTING_LORA_DIR=${LORA_DIR} bash $(basename "$0") ${MODEL_NAME}"

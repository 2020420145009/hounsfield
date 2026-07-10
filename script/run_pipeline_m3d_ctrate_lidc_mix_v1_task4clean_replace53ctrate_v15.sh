#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate_lora}

ROOT_DIR="/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d"
PYTHON_BIN="${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3/bin/python}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD}"
BENCH_ENV="${BENCH_ENV:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate.jsonl}"
TEST_DATA="${TEST_DATA:-/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-radvqa/3drad_test123456_1.jsonl}"
IMAGE_BASE_DIR="${IMAGE_BASE_DIR:-/volume/med-train/data/CT-RATE-DATA/dataset}"

WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_dir/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/${MODEL_NAME}}"
BENCH_RESULT_ROOT="${BENCH_RESULT_ROOT:-${BENCHMARK_ROOT}/runtime/${MODEL_NAME}/results}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${BENCHMARK_ROOT}/runtime/${MODEL_NAME}_comparison}"

TRAIN_SCRIPT="${ROOT_DIR}/finetune_3d_strict_cot_prompt_onlylora_v15.py"
EVAL_SCRIPT="${ROOT_DIR}/inference_compare_9_v15.py"
BENCH_RUNNER="${BENCHMARK_ROOT}/scripts/run_flemingvl_3drad.sh"
EXPORT_OPEN="${BENCHMARK_ROOT}/scripts/export_open_task_metrics.py"
EXPORT_TASK4="${BENCHMARK_ROOT}/scripts/export_task4_binary_accuracy.py"
EXPORT_TASK5="${BENCHMARK_ROOT}/scripts/export_task5_accuracy.py"
EXPORT_TASK6="${BENCHMARK_ROOT}/scripts/export_task6_accuracy.py"
SUMMARIZE_ALL="${BENCHMARK_ROOT}/scripts/summarize_all_results.py"
REMAP_SCRIPT="${ROOT_DIR}/remap_lora_checkpoint_keys.py"

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
BENCH_GPU_IDS="${BENCH_GPU_IDS:-0,1,2,3}"

NUM_SLICES="${NUM_SLICES:-8}"
EPOCHS="${EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LR="${LR:-1e-5}"
DTYPE="${DTYPE:-bfloat16}"
GEM_ALPHA="${GEM_ALPHA:-0.1}"
DEBUG_PRINT_EVERY="${DEBUG_PRINT_EVERY:-0}"
SLICE_CACHE_MODE="${SLICE_CACHE_MODE:-auto}"
PRECOMPUTE_SLICE_CACHE="${PRECOMPUTE_SLICE_CACHE:-0}"
PRECOMPUTE_SLICE_CACHE_LIMIT="${PRECOMPUTE_SLICE_CACHE_LIMIT:--1}"

SMOKE_NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-64}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-1024}"

FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"
FULL_MAX_NEW_TOKENS="${FULL_MAX_NEW_TOKENS:-1024}"

RUN_FULL="${RUN_FULL:-1}"
AUTO_REMAP_LORA="${AUTO_REMAP_LORA:-1}"
EVAL_BASELINE="${EVAL_BASELINE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EXISTING_LORA_DIR="${EXISTING_LORA_DIR:-}"
BENCH_TASKS="${BENCH_TASKS:-task1,task2,task3,task4,task5,task6}"
THREED_RAD_BATCH_SIZE="${THREED_RAD_BATCH_SIZE:-32}"
THREED_RAD_NUM_SLICES="${THREED_RAD_NUM_SLICES:-8}"

mkdir -p "${WORK_DIR}" "${RESULT_DIR}" "${COMPARE_OUTPUT_DIR}"

check_need_remap() {
    local lora_dir="$1"
    "${PYTHON_BIN}" - <<'PY' "${lora_dir}"
import sys
from pathlib import Path
from safetensors.torch import load_file

lora_dir = Path(sys.argv[1])
state = load_file(str(lora_dir / "adapter_model.safetensors"), device="cpu")
need = any(k.startswith("base_model.model.base_model.model.") for k in state.keys())
print("1" if need else "0")
PY
}

do_remap() {
    local src_dir="$1"
    local dst_dir="$2"
    "${PYTHON_BIN}" "${REMAP_SCRIPT}" --src "${src_dir}" --dst "${dst_dir}" --force
}

print_metrics_summary() {
    local eval_base="$1"
    "${PYTHON_BIN}" - <<'PY' "${eval_base}"
import glob
import json
import sys
from pathlib import Path

base = sys.argv[1]
metric_paths = sorted(glob.glob(base + "_*_metrics.json"))
if not metric_paths:
    print(f"[Metrics] no metrics json found under prefix: {base}")
    raise SystemExit(0)

print(f"[Metrics] summary for prefix: {base}")
for path_str in metric_paths:
    path = Path(path_str)
    data = json.loads(path.read_text(encoding='utf-8'))
    parts = [path.name]
    if 'mcq_accuracy' in data:
        parts.append(f"mcq_accuracy={data['mcq_accuracy']:.4f}")
    if 'open_bleu' in data:
        parts.append(f"open_bleu={data['open_bleu']:.2f}")
    if 'open_rougeL' in data:
        parts.append(f"open_rougeL={data['open_rougeL']:.2f}")
    parts.append(f"num_rows={data.get('num_rows', 'NA')}")
    print("  " + " | ".join(parts))
PY
}

export_compare_tables() {
    local new_model_name="$1"
    local new_result_root="$2"
    local output_dir="$3"

    common_args=(
      --model-result m3d /volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/m3d_3drad/results
      --model-result fleming_base /volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/flemingvl8b_3drad/results
      --model-result fleming_lora /volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/flemingvl8b_3drad_lora/results
      --model-result fleming_lora_cot /volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/flemingvl8b_3drad_lora_cot/results
      --model-result lingshu /volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/lingshu7b_3drad/results
      --model-result "${new_model_name}" "${new_result_root}"
    )

    mkdir -p "${output_dir}"

    "${PYTHON_BIN}" "${EXPORT_OPEN}" --task task1 "${common_args[@]}" --output "${output_dir}/task1_metrics_comparison.csv"
    "${PYTHON_BIN}" "${EXPORT_OPEN}" --task task2 "${common_args[@]}" --output "${output_dir}/task2_metrics_comparison.csv"
    "${PYTHON_BIN}" "${EXPORT_OPEN}" --task task3 "${common_args[@]}" --output "${output_dir}/task3_metrics_comparison.csv"
    "${PYTHON_BIN}" "${EXPORT_TASK4}" "${common_args[@]}" --output "${output_dir}/task4_binary_accuracy_comparison.csv"
    "${PYTHON_BIN}" "${EXPORT_TASK5}" "${common_args[@]}" --output "${output_dir}/task5_accuracy_comparison.csv"
    "${PYTHON_BIN}" "${EXPORT_TASK6}" "${common_args[@]}" --output "${output_dir}/task6_accuracy_comparison.csv"
    "${PYTHON_BIN}" "${SUMMARIZE_ALL}" "${common_args[@]}" --output "${output_dir}/summary_all_models.json"
}

if [ "${SKIP_TRAIN}" = "1" ]; then
    if [ -z "${EXISTING_LORA_DIR}" ]; then
        echo "SKIP_TRAIN=1 时必须提供 EXISTING_LORA_DIR"
        exit 1
    fi
    LATEST_LORA_DIR="${EXISTING_LORA_DIR}"
    echo "======================================================="
    echo "[1/5] Skip training and reuse existing LoRA"
    echo "======================================================="
    echo "复用 LoRA: ${LATEST_LORA_DIR}"
else
    echo "======================================================="
    echo "[1/5] 4-GPU training on task4clean_replace53ctrate"
    echo "======================================================="
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "TRAIN_DATA=${TRAIN_DATA}"
    echo "TRAIN_GPU_IDS=${TRAIN_GPU_IDS}"
    echo "TRAIN_NPROC=${TRAIN_NPROC}"

    SLICE_CACHE_ARGS=(--slice_cache_mode "${SLICE_CACHE_MODE}")
    if [ "${PRECOMPUTE_SLICE_CACHE}" = "1" ]; then
        SLICE_CACHE_ARGS+=(--precompute_slice_cache --precompute_slice_cache_limit "${PRECOMPUTE_SLICE_CACHE_LIMIT}")
    fi

    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" \
    torchrun --nproc_per_node="${TRAIN_NPROC}" "${TRAIN_SCRIPT}" \
        --model_path "${BASE_MODEL}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TRAIN_DATA}" \
        --save_dir "${WORK_DIR}" \
        --gpu_ids "${TRAIN_GPU_IDS}" \
        --num_slices "${NUM_SLICES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
        --lr "${LR}" \
        --dtype "${DTYPE}" \
        --gem_alpha "${GEM_ALPHA}" \
        --use_llm_lora 1 \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --debug_print_every "${DEBUG_PRINT_EVERY}" \
        --use_flash_attn \
        "${SLICE_CACHE_ARGS[@]}"

    LATEST_LORA_DIR=$(find "${WORK_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${MODEL_NAME}" | sort | tail -n 1)
    if [ -z "${LATEST_LORA_DIR}" ]; then
        echo "未找到训练产出的 LoRA 目录"
        exit 1
    fi
    echo "训练完成，LoRA 目录: ${LATEST_LORA_DIR}"
fi

echo "======================================================="
echo "[2/5] Remap check"
echo "======================================================="

RUN_LORA_DIR="${LATEST_LORA_DIR}"
RUN_MODEL_NAME="${MODEL_NAME}"

if [ "${AUTO_REMAP_LORA}" = "1" ]; then
    NEED_REMAP="$(check_need_remap "${LATEST_LORA_DIR}")"
    if [ "${NEED_REMAP}" = "1" ]; then
        REMAP_DIR="${LATEST_LORA_DIR}_autoremapped"
        do_remap "${LATEST_LORA_DIR}" "${REMAP_DIR}"
        RUN_LORA_DIR="${REMAP_DIR}"
        RUN_MODEL_NAME="${MODEL_NAME}_autoremapped"
        echo "LoRA 已 remap 到: ${RUN_LORA_DIR}"
    else
        echo "LoRA key 前缀正常，无需 remap"
    fi
else
    echo "AUTO_REMAP_LORA=0，跳过 remap"
fi

BASELINE_FLAG=""
if [ "${EVAL_BASELINE}" = "1" ]; then
    BASELINE_FLAG="--eval_baseline"
fi

echo "======================================================="
echo "[3/5] Smoke evaluation (legacy local eval)"
echo "======================================================="

SMOKE_OUTPUT_BASE="${RESULT_DIR}/${MODEL_NAME}_smoke/inference_results"
mkdir -p "$(dirname "${SMOKE_OUTPUT_BASE}")"

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
    --dtype "${DTYPE}" \
    --gpu_ids "${EVAL_GPU_IDS}" \
    --debug_print_every 1 \
    --use_flash_attn \
    ${BASELINE_FLAG}

print_metrics_summary "${SMOKE_OUTPUT_BASE}"

if [ "${RUN_FULL}" != "1" ]; then
    echo "RUN_FULL!=1，已停止在 smoke test。"
    echo "继续全量流程可执行:"
    echo "SKIP_TRAIN=1 RUN_FULL=1 EXISTING_LORA_DIR=${LATEST_LORA_DIR} bash $(basename "$0") ${MODEL_NAME}"
    exit 0
fi

echo "======================================================="
echo "[4/5] Full 3D-RAD benchmark with new pipeline"
echo "======================================================="

export CUDA_VISIBLE_DEVICES="${BENCH_GPU_IDS}"
export THREED_RAD_BATCH_SIZE="${THREED_RAD_BATCH_SIZE}"
export THREED_RAD_NUM_SLICES="${THREED_RAD_NUM_SLICES}"

bash "${BENCH_RUNNER}" \
    "${BENCH_ENV}" \
    "${BASE_MODEL}" \
    "${BENCHMARK_ROOT}" \
    "${BENCH_TASKS}" \
    "${BENCH_RESULT_ROOT}" \
    "${RUN_LORA_DIR}"

echo "======================================================="
echo "[5/5] Export comparison tables"
echo "======================================================="

export_compare_tables "${RUN_MODEL_NAME}" "${BENCH_RESULT_ROOT}" "${COMPARE_OUTPUT_DIR}"

echo "======================================================="
echo "Pipeline done."
echo "Train data: ${TRAIN_DATA}"
echo "LoRA dir: ${RUN_LORA_DIR}"
echo "3D-RAD result root: ${BENCH_RESULT_ROOT}"
echo "Comparison dir: ${COMPARE_OUTPUT_DIR}"

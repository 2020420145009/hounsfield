#!/bin/bash
set -euo pipefail

ROOT_DIR="/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d"
PYTHON_BIN="${PYTHON_BIN:-/volume/med-train/users/shuyan/envs/internvl3/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_SLICES="${NUM_SLICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MAX_PER_ANSWER="${MAX_PER_ANSWER:-256}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
RUN_TAG="${RUN_TAG:-trial}"
EXPERIMENT_DIR="${ROOT_DIR}/regen_experiments/m3d_mcq_shortcot/${RUN_TAG}"
INPUT_JSONL="${EXPERIMENT_DIR}/input_candidates.jsonl"
OUTPUT_BASE="${EXPERIMENT_DIR}/predictions"

mkdir -p "${EXPERIMENT_DIR}"

echo "======================================================="
echo "[1/3] Prepare balanced regeneration input"
echo "======================================================="
"${PYTHON_BIN}" "${ROOT_DIR}/prepare_m3d_mcq_regen_input.py" \
  --input "${ROOT_DIR}/dataset_quality_audit/m3d_mcq_mcq_rejected_cot_too_short.jsonl" \
  --output "${INPUT_JSONL}" \
  --max_per_answer "${MAX_PER_ANSWER}"

echo "======================================================="
echo "[2/3] Run 4-GPU regeneration inference"
echo "======================================================="

INFER_ARGS=(
  "${ROOT_DIR}/inference_compare_9_v15.py"
  --model_path "${MODEL_PATH}"
  --data_path "${INPUT_JSONL}"
  --image_base_dir /volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy1
  --output_path "${OUTPUT_BASE}"
  --num_slices "${NUM_SLICES}"
  --num_samples -1
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --gpu_ids "${GPU_IDS}"
  --debug_print_every 0
  --skip_auto_metric
)

if [ "${USE_FLASH_ATTN}" = "1" ]; then
  INFER_ARGS+=(--use_flash_attn)
fi

if [ -n "${EXISTING_LORA_DIR:-}" ]; then
  INFER_ARGS+=(--trained_weights "${EXISTING_LORA_DIR}" --model_names regen_lora)
else
  INFER_ARGS+=(--eval_baseline)
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${INFER_ARGS[@]}"

if [ ! -f "${OUTPUT_BASE}_mcq_vqa.jsonl" ]; then
  echo "未生成 ${OUTPUT_BASE}_mcq_vqa.jsonl，推理阶段失败。"
  exit 1
fi

echo "======================================================="
echo "[3/3] Validate regenerated outputs"
echo "======================================================="
"${PYTHON_BIN}" "${ROOT_DIR}/validate_m3d_mcq_regen.py" \
  --predictions "${OUTPUT_BASE}_mcq_vqa.jsonl" \
  --accepted_output "${EXPERIMENT_DIR}/accepted_regenerated.jsonl" \
  --rejected_output "${EXPERIMENT_DIR}/rejected_regenerated.jsonl"

echo "Experiment dir: ${EXPERIMENT_DIR}"

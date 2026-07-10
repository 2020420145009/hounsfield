#!/usr/bin/env bash
set -euo pipefail

# Localized from the official Qwen3.6 README vLLM example for a 4xH200 node.
# Official reference:
# vllm serve Qwen/Qwen3.6-35B-A3B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3

ENV_DIR="${ENV_DIR:-/volume/med-train/users/shuyan/envs/qwen-vllm-517}"
MODEL_DIR="${MODEL_DIR:-/volume/med-train/users/shuyan/models/Qwen3.6-35B-A3B}"
SERVICE_DIR="${SERVICE_DIR:-/volume/med-train/users/shuyan/services/qwen36_35b_a3b_vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18036}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-0}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-}"

mkdir -p "${SERVICE_DIR}"

PID_FILE="${SERVICE_DIR}/qwen36.pid"
LOG_FILE="${SERVICE_DIR}/qwen36.log"

if [[ ! -x "${ENV_DIR}/bin/vllm" ]]; then
  echo "vllm binary not found: ${ENV_DIR}/bin/vllm" >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

FOREGROUND="${FOREGROUND:-0}"

if [[ "${FOREGROUND}" != "1" ]] && [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "qwen36 service already running with PID $(cat "${PID_FILE}")"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

ARGS=(
  serve "${MODEL_DIR}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --reasoning-parser qwen3
  --served-model-name "${SERVED_MODEL_NAME}"
)

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
fi

if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  ARGS+=(--language-model-only)
fi

if [[ -n "${MEDIA_IO_KWARGS}" ]]; then
  ARGS+=(--media-io-kwargs "${MEDIA_IO_KWARGS}")
fi

nohup env \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  "${ENV_DIR}/bin/vllm" "${ARGS[@]}" \
  >"${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started qwen36_35b_a3b_vllm on ${HOST}:${PORT} with PID $(cat "${PID_FILE}")"
echo "Model: ${MODEL_DIR}"
echo "Log: ${LOG_FILE}"

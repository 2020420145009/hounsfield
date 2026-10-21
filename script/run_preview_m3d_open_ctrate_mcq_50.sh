#!/usr/bin/env bash
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SCRIPT="$ROOT/synthesize_m3d_ctrate_cot_local_qwen72b.py"
OUT_DIR=${OUT_DIR:-$ROOT/preview_m3d_open_ctrate_mcq_50}
GPU_IDS=${GPU_IDS:-0,1,2,3}
LIMIT=${LIMIT:-50}

mkdir -p "$OUT_DIR"

python "$SCRIPT" \
  --mode m3d_open \
  --output-dir "$OUT_DIR" \
  --gpu-devices "$GPU_IDS" \
  --test-limit "$LIMIT"

python "$SCRIPT" \
  --mode ctrate_mcq \
  --output-dir "$OUT_DIR" \
  --gpu-devices "$GPU_IDS" \
  --test-limit "$LIMIT"

echo "Preview generation done."
echo "Output dir: $OUT_DIR"

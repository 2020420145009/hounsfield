#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/synthesize_m3d_ctrate_cot_local_qwen72b.py
GPU=0,1,2,3

M3D_OUT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/generated_m3d_cot_1000
CTRATE_OUT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/generated_ctrate_cot_1000

python "$SCRIPT" \
  --mode m3d_mcq \
  --output-dir "$M3D_OUT" \
  --gpu-devices "$GPU" \
  --test-limit 1000

python "$SCRIPT" \
  --mode m3d_open \
  --output-dir "$M3D_OUT" \
  --gpu-devices "$GPU" \
  --test-limit 1000

python "$SCRIPT" \
  --mode ctrate_mcq \
  --output-dir "$CTRATE_OUT" \
  --gpu-devices "$GPU" \
  --test-limit 1000

python "$SCRIPT" \
  --mode ctrate_open \
  --output-dir "$CTRATE_OUT" \
  --gpu-devices "$GPU" \
  --test-limit 1000

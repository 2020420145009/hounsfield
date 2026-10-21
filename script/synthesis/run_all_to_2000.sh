#!/usr/bin/env bash
source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SYN_ROOT=$ROOT/script/synthesis

export GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
export TP=${TP:-4}
export BATCH_SIZE=${BATCH_SIZE:-8}
export TEST_LIMIT=${TEST_LIMIT:-8000}
export TARGET_COUNT=${TARGET_COUNT:-2000}

bash "$SYN_ROOT/run_ctrate_open_to_2000.sh"
bash "$SYN_ROOT/run_ctrate_mcq_to_2000.sh"
bash "$SYN_ROOT/run_m3d_open_to_2000.sh"
bash "$SYN_ROOT/run_m3d_mcq_to_2000.sh"

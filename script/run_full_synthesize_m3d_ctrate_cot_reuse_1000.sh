#!/usr/bin/env bash
set -euo pipefail

ROOT=/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d
SCRIPT="$ROOT/synthesize_m3d_ctrate_cot_local_qwen72b.py"
GPU_IDS=${GPU_IDS:-0,1,2,3}
OUT_DIR=${OUT_DIR:-$ROOT/generated_m3d_ctrate_cot_full}

REUSE_M3D_MCQ_FROM=${REUSE_M3D_MCQ_FROM:-}
REUSE_CTRATE_OPEN_FROM=${REUSE_CTRATE_OPEN_FROM:-}
REUSE_CTRATE_MCQ_FROM=${REUSE_CTRATE_MCQ_FROM:-}
LIMIT=${LIMIT:-}

mkdir -p "$OUT_DIR"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -n "$src" ] && [ -f "$src" ]; then
    cp -f "$src" "$dst"
  fi
}

copy_from_dir() {
  local dir="$1"
  local name="$2"
  local dst="$3"
  if [ -n "$dir" ]; then
    copy_if_exists "$dir/$name" "$dst"
  fi
}

# Reuse already synthesized pieces if the caller provides source locations.
copy_from_dir "$REUSE_M3D_MCQ_FROM" "m3d_mcq_cot.jsonl" "$OUT_DIR/m3d_mcq_cot.jsonl"
copy_from_dir "$REUSE_M3D_MCQ_FROM" "m3d_mcq.resume_state.json" "$OUT_DIR/m3d_mcq.resume_state.json"
copy_from_dir "$REUSE_M3D_MCQ_FROM" "m3d_mcq.failed.jsonl" "$OUT_DIR/m3d_mcq.failed.jsonl"
copy_from_dir "$REUSE_M3D_MCQ_FROM" "m3d_mcq.failed_raw.jsonl" "$OUT_DIR/m3d_mcq.failed_raw.jsonl"

copy_from_dir "$REUSE_CTRATE_OPEN_FROM" "ctrate_open_cot.jsonl" "$OUT_DIR/ctrate_open_cot.jsonl"
copy_from_dir "$REUSE_CTRATE_OPEN_FROM" "ctrate_open.resume_state.json" "$OUT_DIR/ctrate_open.resume_state.json"
copy_from_dir "$REUSE_CTRATE_OPEN_FROM" "ctrate_open.failed.jsonl" "$OUT_DIR/ctrate_open.failed.jsonl"
copy_from_dir "$REUSE_CTRATE_OPEN_FROM" "ctrate_open.failed_raw.jsonl" "$OUT_DIR/ctrate_open.failed_raw.jsonl"

copy_from_dir "$REUSE_CTRATE_MCQ_FROM" "ctrate_mcq_cot.jsonl" "$OUT_DIR/ctrate_mcq_cot.jsonl"
copy_from_dir "$REUSE_CTRATE_MCQ_FROM" "ctrate_mcq.resume_state.json" "$OUT_DIR/ctrate_mcq.resume_state.json"
copy_from_dir "$REUSE_CTRATE_MCQ_FROM" "ctrate_mcq.failed.jsonl" "$OUT_DIR/ctrate_mcq.failed.jsonl"
copy_from_dir "$REUSE_CTRATE_MCQ_FROM" "ctrate_mcq.failed_raw.jsonl" "$OUT_DIR/ctrate_mcq.failed_raw.jsonl"

ARGS=(--mode all --output-dir "$OUT_DIR" --gpu-devices "$GPU_IDS")
if [ -n "$LIMIT" ]; then
  ARGS+=(--test-limit "$LIMIT")
fi

python "$SCRIPT" "${ARGS[@]}"

echo "Full synthesis done."
echo "Output dir: $OUT_DIR"

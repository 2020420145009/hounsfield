#!/usr/bin/env bash
#
# LLM-as-a-judge 评分: train_11_2k 模型在 3D-RAD task1 / task2 / task3 上的开放式
# 问答结果(eval_open_vqa.csv) -> 逐条用裁判模型打分(pass/score) ->
# 汇总成与各基线模型的对比表(judge_pass_rate / judge_score_mean)。
#
# 评判方式: vLLM 后端,裁判模型(DeepSeek-R1-Distill-Llama-70B)在进程内加载、
# 直接调用(MODEL-CALL,不另起 HTTP 服务)。传入大的 --max-num-seqs,让 vLLM 用
# 连续批处理把大量评判 prompt 打包进同一批,吃满 GPU。
#
# 复用现有脚本(不重写评判/聚合逻辑):
#   scripts/llm_judge_open_results.py        -> 逐 subtask 产出 eval_open_vqa_judge.csv
#   scripts/export_open_task_judge_metrics.py-> 汇总 judge 指标对比 CSV
#
# 用法(在有 GPU 的机器上跑,无参可直接执行):
#   bash run_train_11_2k_judge.sh [conda_env] [judge_model] [tp_size] [max_num_seqs]
#
# 常用覆盖:
#   TASKS=task3 bash run_train_11_2k_judge.sh        # 只评 task3
#   OVERWRITE=1 bash run_train_11_2k_judge.sh        # 重判(忽略已存在的 judge 文件)
#   MAX_SAMPLES=8 bash run_train_11_2k_judge.sh      # 每个 subtask 只判前 8 条(冒烟)

CONDA_ENV_PATH="${1:-/volume/med-train/users/shuyan/envs/medeval_for_internvl3}"
JUDGE_MODEL="${2:-/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B}"
TP_SIZE="${3:-4}"
MAX_NUM_SEQS="${4:-512}"

TASKS="${TASKS:-task1,task2,task3}"

# 从脚本位置推出 benchmark 根目录(scripts/ 的上一级)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /miniconda3/bin/activate "$CONDA_ENV_PATH"
set -euo pipefail

JUDGE_SCRIPT="${BENCHMARK_ROOT}/scripts/llm_judge_open_results.py"
EXPORT_JUDGE="${BENCHMARK_ROOT}/scripts/export_open_task_judge_metrics.py"

MODEL_NAME="train_11_2k"
RESULTS_ROOT="${BENCHMARK_ROOT}/runtime/${MODEL_NAME}/results"
COMPARE_DIR="${BENCHMARK_ROOT}/runtime/${MODEL_NAME}_judge_comparison"
mkdir -p "${COMPARE_DIR}"

# 可选开关
EXTRA_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then EXTRA_ARGS+=(--overwrite); fi
if [ -n "${MAX_SAMPLES:-}" ]; then EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}"); fi

echo "============================================================"
echo "裁判模型   : ${JUDGE_MODEL}"
echo "后端       : vLLM (in-process model call)"
echo "TP / batch : ${TP_SIZE} / ${MAX_NUM_SEQS}"
echo "评测任务   : ${TASKS}"
echo "结果目录   : ${RESULTS_ROOT}"
echo "对比输出   : ${COMPARE_DIR}"
echo "============================================================"

# ── [1/2] 逐 subtask 评判,写 eval_open_vqa_judge.csv ─────────────────────
python "$JUDGE_SCRIPT" \
  --results-root "$RESULTS_ROOT" \
  --tasks "$TASKS" \
  --judge-model "$JUDGE_MODEL" \
  --judge-backend vllm \
  --tensor-parallel-size "$TP_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${EXTRA_ARGS[@]}"

# ── [2/2] 汇总 judge 指标,导出与各基线模型的对比表 ──────────────────────
COMMON_ARGS=(
  --model-result m3d              "${BENCHMARK_ROOT}/runtime/m3d_3drad/results"
  --model-result fleming_base     "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad/results"
  --model-result fleming_base_lora "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad_cot/results"
  --model-result fleming_lora     "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad_lora/results"
  --model-result fleming_lora_cot "${BENCHMARK_ROOT}/runtime/flemingvl8b_3drad_lora_cot/results"
  --model-result lingshu          "${BENCHMARK_ROOT}/runtime/lingshu7b_3drad/results"
  --model-result "${MODEL_NAME}"  "${RESULTS_ROOT}"
)

IFS=',' read -ra TASK_ARR <<< "$TASKS"
for t in "${TASK_ARR[@]}"; do
  python "$EXPORT_JUDGE" --task "$t" "${COMMON_ARGS[@]}" \
    --output "${COMPARE_DIR}/${t}_judge_comparison.csv"
  echo "[导出] ${COMPARE_DIR}/${t}_judge_comparison.csv"
done

echo "============================================================"
echo "完成。judge 明细写在各 subtask 的 eval_open_vqa_judge.csv"
echo "对比表目录: ${COMPARE_DIR}"
echo "============================================================"

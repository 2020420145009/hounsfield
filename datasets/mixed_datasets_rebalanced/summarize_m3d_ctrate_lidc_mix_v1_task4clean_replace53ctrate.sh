#!/usr/bin/env bash
source /miniconda3/bin/activate /volume/med-train/users/shuyan/envs/medeval_for_internvl3
set -euo pipefail

python /volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/summarize_mixed_dataset.py \
  --input-jsonl /volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate.jsonl \
  --output-summary /volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate/summary.json \
  --label m3d_ctrate_lidc_mix_v1_task4clean_replace53ctrate

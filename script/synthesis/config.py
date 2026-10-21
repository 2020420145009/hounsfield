from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_TRAINING_DATA_ROOT = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/training_data")
DEFAULT_CTRATE_ROOT = Path("/volume/med-train/data/CT-RATE-DATA")
DEFAULT_CTRATE_OPEN_INPUT = Path("/volume/med-train/data/CT-RATE-DATA/Synthesized_CoT_Dataset_ctrate/ct_rate_open_vqa_three_tiers_5000_15000_clean.jsonl")
DEFAULT_CTRATE_MCQ_INPUT = Path("/volume/med-train/data/CT-RATE-DATA/Synthesized_CoT_Dataset_ctrate/ct_rate_mcq_clean.jsonl")
DEFAULT_M3D_MCQ_INPUT = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/generated_m3d_cot_1000/m3d_mcq_cot.jsonl")
DEFAULT_LIDC_MIXED_INPUT = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/generated_lidc_strict_cot_v2/medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl")

OLD_SOURCE_COUNTS = {
    'ctrate_open': 854,
    'm3d_mcq': 861,
    'lidc_mixed': 710,
}

OLD_TASK_TYPE_COUNTS = {
    'open_vqa': 1232,
    'mcq_vqa': 981,
    'report': 212,
}

TASK4_CATEGORY_MIN_TARGET = 30

SOURCE_OUTPUT_FILES = {
    'ctrate_open': 'ctrate_open_cot.jsonl',
    'ctrate_mcq': 'ctrate_mcq_cot.jsonl',
    'm3d_mcq': 'm3d_mcq_cot.jsonl',
    'lidc_mixed': 'medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl',
}

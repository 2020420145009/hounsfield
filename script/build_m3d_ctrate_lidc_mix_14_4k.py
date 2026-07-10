#!/usr/bin/env python3
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'mixed_datasets_rebalanced' / 'm3d_ctrate_lidc_mix_14.4k'
OUT_JSONL = OUT_DIR / 'm3d_ctrate_lidc_mix_14.4k.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

TARGET_TOTAL = 14400
SEED = 42

SOURCE_PATHS = {
    'm3d_mcq': ROOT / 'generated_m3d_ctrate_cot_full' / 'm3d_mcq_cot.jsonl',
    'm3d_open': ROOT / 'generated_m3d_ctrate_cot_full' / 'm3d_open_cot.jsonl',
    'ctrate_mcq': ROOT / 'generated_m3d_ctrate_cot_full' / 'ctrate_mcq_cot.jsonl',
    'ctrate_open': ROOT / 'generated_m3d_ctrate_cot_full' / 'ctrate_open_cot.jsonl',
    'lidc_all': ROOT / 'generated_lidc_strict_cot_v2' / 'medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl',
}


def read_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_question(text: str) -> str:
    text = str(text or '')
    text = text.replace('<image>\n', ' ').replace('<image>', ' ')
    text = re.sub(r'\[Perform .*?\]\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_lidc_answer(text: str) -> str:
    text = str(text or '')

    def _strip_number_prefix(match):
        body = match.group(1)
        cleaned_lines = []
        for line in body.splitlines():
            line = re.sub(r'^\s*\d+\.\s*', '', line).strip()
            if line:
                cleaned_lines.append(line)
        return '<think>\n' + '\n'.join(cleaned_lines) + '\n</think>'

    return re.sub(r'<think>\s*(.*?)\s*</think>', _strip_number_prefix, text, flags=re.S | re.I)


def normalize_lidc_row(row: dict) -> dict:
    out = dict(row)
    convs = list(out.get('conversations', []))
    if len(convs) >= 2:
        convs[0] = {'from': convs[0].get('from', 'human'), 'value': normalize_question(convs[0].get('value', ''))}
        convs[1] = {'from': convs[1].get('from', 'gpt'), 'value': clean_lidc_answer(convs[1].get('value', ''))}
        out['conversations'] = convs
    out['_dataset_source'] = 'lidc_strict_cot_v2_all'
    return out


def proportional_allocate(counts: dict, target_total: int) -> dict:
    total = sum(counts.values())
    base_alloc = {k: int(v / total * target_total) for k, v in counts.items()}
    remaining = target_total - sum(base_alloc.values())
    fractions = sorted(
        ((counts[k] / total * target_total - base_alloc[k], k) for k in counts),
        reverse=True,
    )
    for _, key in fractions[:remaining]:
        base_alloc[key] += 1
    return base_alloc


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_rows = {name: read_jsonl(path) for name, path in SOURCE_PATHS.items()}
    source_counts = {name: len(rows) for name, rows in source_rows.items()}

    lidc_rows = [normalize_lidc_row(row) for row in source_rows['lidc_all']]
    lidc_count = len(lidc_rows)
    if lidc_count > TARGET_TOTAL:
        raise ValueError(f'LIDC rows ({lidc_count}) exceed target total ({TARGET_TOTAL}).')

    non_lidc_counts = {k: v for k, v in source_counts.items() if k != 'lidc_all'}
    target_non_lidc = TARGET_TOTAL - lidc_count
    sample_plan = proportional_allocate(non_lidc_counts, target_non_lidc)

    dataset = []
    selected_source_counts = {}
    for key in ('m3d_mcq', 'm3d_open', 'ctrate_mcq', 'ctrate_open'):
        rows = list(source_rows[key])
        rng.shuffle(rows)
        picked = rows[:sample_plan[key]]
        dataset.extend(picked)
        selected_source_counts[key] = len(picked)

    dataset.extend(lidc_rows)
    selected_source_counts['lidc_all'] = len(lidc_rows)
    rng.shuffle(dataset)

    with OUT_JSONL.open('w', encoding='utf-8') as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'output_jsonl': str(OUT_JSONL),
        'total_rows': len(dataset),
        'target_total_rows': TARGET_TOTAL,
        'seed': SEED,
        'source_paths': {k: str(v) for k, v in SOURCE_PATHS.items()},
        'source_total_counts': source_counts,
        'selected_source_counts': selected_source_counts,
        'task_type_counts': dict(Counter(row.get('task_type', 'unknown') for row in dataset)),
        'dataset_source_counts': dict(Counter(row.get('_dataset_source', 'missing') for row in dataset)),
        'notes': [
            'Included all rows from generated_lidc_strict_cot_v2/medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl.',
            'Sampled M3D and CT-RATE rows proportionally from generated_m3d_ctrate_cot_full to reach exactly 14,400 total rows.',
            'For LIDC rows only, removed [Perform ...] prefixes in the question and stripped numeric prefixes like 1. 2. 3. inside <think>.',
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

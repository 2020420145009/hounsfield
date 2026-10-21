#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'mixed_datasets_rebalanced' / 'm3d_ctrate_lidc_synth_batch_v001'
OUT_JSONL = OUT_DIR / 'm3d_ctrate_lidc_synth_batch_v001.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

SOURCES = {
    'ctrate_open': ROOT / 'training_data/ctrate_open/v500/ctrate_open_cot.jsonl',
    'ctrate_mcq': ROOT / 'training_data/ctrate_mcq/v500/ctrate_mcq_cot.jsonl',
    'm3d_open': ROOT / 'training_data/m3d_open/v500/m3d_open_cot.jsonl',
    'm3d_mcq': ROOT / 'training_data/m3d_mcq/v500/m3d_mcq_cot.jsonl',
    'lidc_mixed': ROOT / 'training_data/lidc_mixed/v003/medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl',
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    source_counts = {}
    task_type_counts = {}
    dimension_counts = {}
    source_task_type_counts = {}
    question_type_counts = {}

    for source_name, path in SOURCES.items():
        count = 0
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row.setdefault('_dataset_source', source_name)
                rows.append(row)
                count += 1
                tt = row.get('task_type', 'unknown')
                dim = row.get('dimension', 'unknown')
                qt = row.get('question_type', 'unknown')
                task_type_counts[tt] = task_type_counts.get(tt, 0) + 1
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
                source_task_type_counts.setdefault(source_name, {})[tt] = source_task_type_counts[source_name].get(tt, 0) + 1
                question_type_counts[qt] = question_type_counts.get(qt, 0) + 1
        source_counts[source_name] = count

    random.Random(20260531).shuffle(rows)

    with OUT_JSONL.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'label': 'm3d_ctrate_lidc_synth_batch_v001',
        'input_sources': {k: str(v) for k, v in SOURCES.items()},
        'total_rows': len(rows),
        'source_counts': source_counts,
        'task_type_counts': task_type_counts,
        'dimension_counts': dict(sorted(dimension_counts.items(), key=lambda x: (-x[1], x[0]))),
        'source_task_type_counts': source_task_type_counts,
        'question_type_counts_top50': dict(sorted(question_type_counts.items(), key=lambda x: (-x[1], x[0]))[:50]),
        'shuffle_seed': 20260531,
        'output_jsonl': str(OUT_JSONL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

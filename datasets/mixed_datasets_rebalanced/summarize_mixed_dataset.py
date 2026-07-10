#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize a mixed training dataset jsonl and write summary.json plus readable stats.')
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, default=None)
    parser.add_argument('--label', type=str, default=None)
    return parser.parse_args()


def normalize_source(value: str) -> str:
    src = str(value or '').strip()
    mapping = {
        'ctrate': 'ctrate_open',
        'm3d_mcq_qa_only': 'm3d_mcq',
        'm3d_mcq': 'm3d_mcq',
        'lidc:strict_cot_v2': 'lidc_mixed',
    }
    return mapping.get(src, src or 'unknown')


def main() -> None:
    args = parse_args()
    rows = []
    with args.input_jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    source_counts = Counter()
    task_type_counts = Counter()
    dimension_counts = Counter()
    source_task_counts = defaultdict(Counter)
    source_dimension_counts = defaultdict(Counter)
    qtype_counts = defaultdict(Counter)

    for row in rows:
        src = normalize_source(row.get('_dataset_source'))
        task_type = str(row.get('task_type', '') or 'unknown')
        dimension = str(row.get('dimension', '') or 'unknown')
        qtype = str(row.get('question_type', '') or 'unknown')

        source_counts[src] += 1
        task_type_counts[task_type] += 1
        dimension_counts[dimension] += 1
        source_task_counts[src][task_type] += 1
        source_dimension_counts[src][dimension] += 1
        qtype_counts[dimension][qtype] += 1

    summary = {
        'label': args.label or args.input_jsonl.stem,
        'output': str(args.input_jsonl),
        'total_rows': len(rows),
        'source_counts': dict(source_counts),
        'task_type_counts': dict(task_type_counts),
        'dimension_counts': dict(dimension_counts),
        'source_task_type_counts': {k: dict(v) for k, v in source_task_counts.items()},
        'source_dimension_counts': {k: dict(v) for k, v in source_dimension_counts.items()},
        'question_type_counts_by_dimension': {k: dict(v) for k, v in qtype_counts.items()},
    }

    output_summary = args.output_summary or args.input_jsonl.with_name('summary.json')
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

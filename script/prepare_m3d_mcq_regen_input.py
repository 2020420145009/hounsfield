#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_m3dtest_like_subset_v2 as builder


def read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare balanced M3D MCQ regeneration input without touching the current clean dataset.')
    parser.add_argument('--input', default='dataset_quality_audit/m3d_mcq_mcq_rejected_cot_too_short.jsonl')
    parser.add_argument('--output', default='regen_experiments/m3d_mcq_shortcot/input_candidates.jsonl')
    parser.add_argument('--max_per_answer', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(args.input)

    by_letter = defaultdict(list)
    skipped = Counter()

    for row in rows:
        question = builder.format_mcq_question(row['conversations'][0]['value'])
        answer = builder.format_mcq_answer(question, row['conversations'][1]['value'])
        letter = builder.extract_answer_letter(answer)
        if letter is None:
            skipped['missing_answer_letter'] += 1
            continue
        row = dict(row)
        row['conversations'] = [
            {'from': 'human', 'value': question, 'type': 'mcq_vqa'},
            {'from': 'gpt', 'value': answer},
        ]
        row['task_type'] = 'mcq_vqa'
        row['_regen_target_letter'] = letter
        row['_regen_source_reason'] = 'cot_too_short'
        by_letter[letter].append(row)

    selected = []
    for letter in ['A', 'B', 'C', 'D']:
        bucket = by_letter[letter]
        random.shuffle(bucket)
        selected.extend(bucket[:args.max_per_answer])

    random.shuffle(selected)
    write_jsonl(args.output, selected)

    summary = {
        'input': str(Path(args.input).resolve()),
        'output': str(Path(args.output).resolve()),
        'total_selected': len(selected),
        'selected_answer_counts': dict(Counter(r['_regen_target_letter'] for r in selected)),
        'available_answer_counts': {k: len(v) for k, v in by_letter.items()},
        'skipped': dict(skipped),
        'seed': args.seed,
        'max_per_answer': args.max_per_answer,
    }
    summary_path = Path(args.output).with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

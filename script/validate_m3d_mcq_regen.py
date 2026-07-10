#!/usr/bin/env python3
import argparse
import json
from collections import Counter
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Validate regenerated M3D MCQ answers and export accepted training-ready samples.')
    parser.add_argument('--predictions', required=True, help='Path to inference output *_mcq_vqa.jsonl')
    parser.add_argument('--accepted_output', default='regen_experiments/m3d_mcq_shortcot/accepted_regenerated.jsonl')
    parser.add_argument('--rejected_output', default='regen_experiments/m3d_mcq_shortcot/rejected_regenerated.jsonl')
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    accepted = []
    rejected = []
    stats = Counter()

    for row in rows:
        question = builder.format_mcq_question(row['question'])
        prediction = builder.sanitize_cot_answer(row['prediction'], question_text=question, force_mcq=True)
        gold = builder.format_mcq_answer(question, row.get('ground_truth', ''))
        pred_letter = builder.extract_answer_letter(prediction)
        gold_letter = builder.extract_answer_letter(gold)
        ok, reason = builder.validate_reasoning_quality(question, prediction, 'mcq_vqa')

        if pred_letter is None:
            reason = 'missing_pred_letter'
            ok = False
        elif gold_letter is None:
            reason = 'missing_gold_letter'
            ok = False
        elif pred_letter != gold_letter:
            reason = 'answer_mismatch'
            ok = False

        out = {
            'id': row['id'],
            'image': row['image'],
            'question': question,
            'ground_truth': gold,
            'prediction': prediction,
            'gold_letter': gold_letter,
            'pred_letter': pred_letter,
            'reason': reason if not ok else None,
        }

        if ok:
            train_row = {
                'id': row['id'],
                'image': row['image'],
                'conversations': [
                    {'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': prediction},
                ],
                'task_type': 'mcq_vqa',
                '_dataset_source': 'm3d:mcq_regenerated',
                '_subset_answer_letter': pred_letter,
            }
            accepted.append(train_row)
            stats['accepted'] += 1
            stats[f'accepted_{pred_letter}'] += 1
        else:
            rejected.append(out)
            stats[reason] += 1

    write_jsonl(args.accepted_output, accepted)
    write_jsonl(args.rejected_output, rejected)

    summary = {
        'predictions': str(Path(args.predictions).resolve()),
        'accepted_output': str(Path(args.accepted_output).resolve()),
        'rejected_output': str(Path(args.rejected_output).resolve()),
        'total_predictions': len(rows),
        'accepted': len(accepted),
        'rejected': len(rejected),
        'stats': dict(stats),
        'accepted_answer_counts': dict(Counter(r['_subset_answer_letter'] for r in accepted)),
    }
    summary_path = Path(args.accepted_output).with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

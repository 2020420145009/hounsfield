#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_m3dtest_like_subset_v2 as builder


ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'dataset_quality_audit'
MAX_EXAMPLES_PER_BUCKET = 5


def classify_record(record, source, mode):
    if mode == 'mcq':
        question = builder.format_mcq_question(record['conversations'][0]['value'])
        answer = builder.format_mcq_answer(question, record['conversations'][1]['value'])
        letter = builder.extract_answer_letter(answer)
        if letter is None:
            return question, answer, False, 'missing_answer_letter'
        ok, reason = builder.validate_reasoning_quality(question, answer, 'mcq_vqa')
        return question, answer, ok, reason

    question = builder.normalize_question(record['conversations'][0]['value'])
    answer = builder.sanitize_cot_answer(str(record['conversations'][1]['value']).strip(), question_text=question, force_mcq=False)
    ok, reason = builder.validate_reasoning_quality(question, answer, 'open_vqa')
    return question, answer, ok, reason


def audit_rows(rows, source, mode):
    stats = Counter()
    examples = defaultdict(list)
    passed = []
    rejected_rows = defaultdict(list)

    for record in rows:
        question, answer, ok, reason = classify_record(record, source, mode)
        if ok:
            stats['passed'] += 1
            passed.append(record)
            continue
        bucket = reason or 'rejected'
        stats[bucket] += 1
        rejected_rows[bucket].append(record)
        if len(examples[bucket]) < MAX_EXAMPLES_PER_BUCKET:
            examples[bucket].append({
                'id': record.get('id'),
                'image': record.get('image'),
                'source': source,
                'mode': mode,
                'question': question,
                'answer': answer,
            })

    stats['total'] = len(rows)
    return stats, examples, passed, rejected_rows


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lidc_mixed = builder.read_jsonl(builder.LIDC_MIXED_PATH)
    lidc_mcq_rows, lidc_open_rows = builder.split_lidc_mixed(lidc_mixed)

    datasets = [
        ('m3d:mcq', 'mcq', builder.read_jsonl(builder.M3D_MCQ_PATH)),
        ('m3d:open', 'open', builder.read_jsonl(builder.M3D_OPEN_PATH)),
        ('ctrate:mcq', 'mcq', builder.read_jsonl(builder.CTRATE_MCQ_PATH)),
        ('ctrate:open', 'open', builder.read_jsonl(builder.CTRATE_OPEN_PATH)),
        ('lidc:mcq', 'mcq', lidc_mcq_rows),
        ('lidc:open', 'open', lidc_open_rows),
    ]

    summary = {
        'paths': {
            'm3d_mcq': str(builder.M3D_MCQ_PATH),
            'm3d_open': str(builder.M3D_OPEN_PATH),
            'ctrate_mcq': str(builder.CTRATE_MCQ_PATH),
            'ctrate_open': str(builder.CTRATE_OPEN_PATH),
            'lidc_mixed': str(builder.LIDC_MIXED_PATH),
        },
        'datasets': {},
    }

    for source, mode, rows in datasets:
        stats, examples, passed_rows, rejected_rows = audit_rows(rows, source, mode)
        source_key = source.replace(':', '_')
        summary['datasets'][source] = dict(stats)
        dump_json(OUT_DIR / f'{source_key}_{mode}_examples.json', examples)
        dump_jsonl(OUT_DIR / f'{source_key}_{mode}_passed.jsonl', passed_rows)
        for reason, reason_rows in rejected_rows.items():
            dump_jsonl(OUT_DIR / f'{source_key}_{mode}_rejected_{reason}.jsonl', reason_rows)

    dump_json(OUT_DIR / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

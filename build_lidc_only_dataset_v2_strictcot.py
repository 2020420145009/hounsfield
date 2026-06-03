#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import build_m3dtest_like_subset_v2 as base


ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'lidc_only_dataset_v2_strictcot'
OUT_JSONL = OUT_DIR / 'lidc_only_mcq_open_v2_strictcot.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

LIDC_MCQ_PATH = ROOT / 'generated_lidc_strict_cot_v2' / 'medical_lidc_mcq_vqa_dataset_strict_cot_v2.jsonl'
LIDC_OPEN_PATH = ROOT / 'generated_lidc_strict_cot_v2' / 'medical_lidc_open_vqa_dataset_strict_cot_v2.jsonl'


def read_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_answer_letter_from_answer_block(text: str):
    text = str(text or '').strip()
    m = base.re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=base.re.I | base.re.S)
    if m:
        text = m.group(1).strip()
    text = base.re.sub(r'\s+', ' ', text)
    m = base.re.search(r'\b([A-D])\b', text, flags=base.re.I)
    if m:
        return m.group(1).upper()
    m = base.re.match(r'\s*([A-Da-d])[\).:\s]', text)
    if m:
        return m.group(1).upper()
    return None


def prepare_mcq(record):
    q = base.format_mcq_question(record['conversations'][0]['value'])
    a = base.format_mcq_answer(q, record['conversations'][1]['value'])
    letter = extract_answer_letter_from_answer_block(a)
    if letter is None:
        return None
    out = dict(record)
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'mcq_vqa'
    out['_dataset_source'] = 'lidc:mcq_only_strictcot'
    out['_subset_answer_letter'] = letter
    return out


def prepare_open(record):
    q = base.normalize_question(record['conversations'][0]['value'])
    a = base.sanitize_cot_answer(record['conversations'][1]['value'], question_text=q, force_mcq=False)
    out = dict(record)
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'open_vqa'
    out['_dataset_source'] = 'lidc:open_only_strictcot'
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mcq_rows = [r for r in (prepare_mcq(x) for x in read_jsonl(LIDC_MCQ_PATH)) if r is not None]
    open_rows = [prepare_open(x) for x in read_jsonl(LIDC_OPEN_PATH)]
    dataset = mcq_rows + open_rows

    with OUT_JSONL.open('w', encoding='utf-8') as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    summary = {
        'output_jsonl': str(OUT_JSONL),
        'total': len(dataset),
        'task_type_counts': dict(Counter(r['task_type'] for r in dataset)),
        'dataset_source_counts': dict(Counter(r['_dataset_source'] for r in dataset)),
        'mcq_answer_counts': dict(Counter(r.get('_subset_answer_letter') for r in dataset if r['task_type'] == 'mcq_vqa')),
        'paths': {
            'lidc_mcq': str(LIDC_MCQ_PATH),
            'lidc_open': str(LIDC_OPEN_PATH),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

import build_m3dtest_like_subset_v2 as base


ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'lidc_only_dataset_v1'
OUT_JSONL = OUT_DIR / 'lidc_only_mcq_open_v1.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

LIDC_MCQ_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/medical_lidc_mcq_vqa_dataset4.json')
LIDC_OPEN_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/medical_lidc_open_vqa_dataset4.json')


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def split_numbered_think_body(body: str):
    body = str(body or '').replace('\r\n', '\n').replace('\r', '\n')
    body = re.sub(r'\s+', ' ', body).strip()
    body = re.sub(r'(?<!^)\s+(\d+\.)', r'\n\1', body)
    parts = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+\.\s*', '', line).strip()
        if line:
            parts.append(line)
    return parts


def light_clean_reasoning_line(line: str):
    line = str(line or '').strip()
    if not line:
        return None
    line = re.sub(r'\b(?:derived from|extracted from|recorded in) the json data\b', 'observed from the lesion attributes', line, flags=re.I)
    line = re.sub(r'\bjson data\b', 'lesion attributes', line, flags=re.I)
    line = re.sub(r'\bjson\b', 'lesion attributes', line, flags=re.I)
    line = re.sub(r'\bmetadata\b', 'lesion attributes', line, flags=re.I)
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def normalize_answer_text(answer_text: str, force_mcq: bool, question_text: str):
    raw = str(answer_text or '').strip()
    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.S | re.I)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw, flags=re.S | re.I)

    cleaned_lines = []
    if think_match:
        lines = split_numbered_think_body(think_match.group(1))
        for idx, line in enumerate(lines):
            line = light_clean_reasoning_line(line)
            if not line:
                continue
            if idx == 0 and not re.match(r'^This is a ', line, flags=re.I):
                line = f'This is a Standard Assessment question. {line}'
            cleaned_lines.append(line)

    final_answer = answer_match.group(1).strip() if answer_match else raw
    final_answer = re.sub(r'\s+', ' ', final_answer).strip()
    if force_mcq:
        letter = base.extract_answer_letter(final_answer)
        choice_map = base.extract_choice_map(question_text)
        if letter and letter in choice_map:
            final_answer = f'{letter}. {base.clean_choice_text(choice_map[letter])}'

    think_block = None
    if cleaned_lines:
        think_block = '<think>\n' + '\n'.join(cleaned_lines) + '\n</think>'
    answer_block = '<answer>\n' + final_answer + '\n</answer>'
    return think_block + '\n' + answer_block if think_block else answer_block


def prepare_lidc_mcq(record):
    q = base.format_mcq_question(record['conversations'][0]['value'])
    a = normalize_answer_text(record['conversations'][1]['value'], force_mcq=True, question_text=q)
    letter = base.extract_answer_letter(a)
    if letter is None:
        return None
    out = dict(record)
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'mcq_vqa'
    out['_dataset_source'] = 'lidc:mcq_only'
    out['_subset_answer_letter'] = letter
    return out


def prepare_lidc_open(record):
    q = base.normalize_question(record['conversations'][0]['value'])
    a = normalize_answer_text(record['conversations'][1]['value'], force_mcq=False, question_text=q)
    out = dict(record)
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'open_vqa'
    out['_dataset_source'] = 'lidc:open_only'
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mcq_rows = [r for r in (prepare_lidc_mcq(x) for x in read_json(LIDC_MCQ_PATH)) if r is not None]
    open_rows = [prepare_lidc_open(x) for x in read_json(LIDC_OPEN_PATH)]
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

from __future__ import annotations

from pathlib import Path
from typing import List
import re

from .category_rules import detect_task4_category
from .dataset_pool import load_source_rows
from .io_utils import read_jsonl
from .schema import SynthesisTask


def extract_think_answer_blocks(text: str) -> tuple[str, str]:
    raw = str(text or '').strip()
    if '<answer>' in raw.lower():
        think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.S | re.I)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw, flags=re.S | re.I)
        think = think_match.group(1).strip() if think_match else ''
        answer = answer_match.group(1).strip() if answer_match else raw
        return think, answer
    return '', raw


def resolve_ctrate_image_path(image_value: str, ctrate_root: Path) -> str:
    image_value = str(image_value or '').strip()
    if image_value.startswith('/'):
        return image_value
    if image_value.endswith('.nii.gz') and image_value.startswith('train_'):
        stem = image_value[:-7]
        parts = stem.split('_')
        if len(parts) >= 3:
            study = '_'.join(parts[:2])
            series = '_'.join(parts[:3])
            return str(ctrate_root / 'dataset' / 'train_fixed' / study / series / image_value)
    if image_value.endswith('.nii.gz') and image_value.startswith('valid_'):
        stem = image_value[:-7]
        parts = stem.split('_')
        if len(parts) >= 3:
            study = '_'.join(parts[:2])
            series = '_'.join(parts[:3])
            return str(ctrate_root / 'dataset' / 'valid_fixed' / study / series / image_value)
    return image_value


def infer_ctrate_reasoning_type(row: dict) -> str:
    row_id = str(row.get('id', ''))
    if '_L1' in row_id:
        return 'Rapid Triage'
    if '_L2' in row_id:
        return 'Standard Assessment'
    if '_L3' in row_id:
        return 'Comprehensive Diagnosis'
    return 'Rapid Triage'


def load_ctrate_tasks(input_path: Path, ctrate_root: Path, task_type: str, limit: int | None = None) -> List[SynthesisTask]:
    tasks: List[SynthesisTask] = []
    for row in read_jsonl(input_path):
        convs = row.get('conversations', [])
        if len(convs) < 2:
            continue
        question = str(convs[0].get('value', '')).strip()
        think, answer = extract_think_answer_blocks(convs[1].get('value', ''))
        if not question or not answer:
            continue
        category, keywords, strategy = detect_task4_category(question, answer)
        task = SynthesisTask(
            task_id=str(row.get('id')),
            source='ctrate_open' if task_type == 'open_vqa' else 'ctrate_mcq',
            task_type=task_type,
            dimension=infer_ctrate_reasoning_type(row),
            image=resolve_ctrate_image_path(row.get('image', ''), ctrate_root),
            question=question,
            answer=answer,
            hint_reasoning=think,
            category=category,
            metadata={'matched_keywords': keywords, 'match_strategy': strategy},
        )
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_ctrate_open_tasks(input_path: Path, ctrate_root: Path, limit: int | None = None) -> List[SynthesisTask]:
    return load_ctrate_tasks(input_path, ctrate_root, 'open_vqa', limit)


def load_ctrate_mcq_tasks(input_path: Path, ctrate_root: Path, limit: int | None = None) -> List[SynthesisTask]:
    return load_ctrate_tasks(input_path, ctrate_root, 'mcq_vqa', limit)


def load_three_source_rows(*, ctrate_open_path: Path, m3d_mcq_path: Path, lidc_mixed_path: Path, limit_per_source: int | None = None):
    ctrate_rows = load_source_rows(ctrate_open_path, 'ctrate_open', limit_per_source)
    m3d_rows = load_source_rows(m3d_mcq_path, 'm3d_mcq', limit_per_source)
    lidc_rows = load_source_rows(lidc_mixed_path, 'lidc_mixed', limit_per_source)
    return ctrate_rows, m3d_rows, lidc_rows

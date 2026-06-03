from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
import re

from .category_rules import detect_task4_category, normalize_yes_no_answer
from .io_utils import read_jsonl


@dataclass
class SourceRow:
    row: dict
    row_id: str
    source_group: str
    task_type: str
    category: Optional[str]
    question: str
    answer: str


def _extract_question_answer(row: dict) -> tuple[str, str]:
    conv = row.get('conversations', [])
    if len(conv) < 2:
        return '', ''
    q = str(conv[0].get('value', '')).strip() if isinstance(conv[0], dict) else ''
    a = str(conv[1].get('value', '')).strip() if isinstance(conv[1], dict) else ''
    return q, a


def _infer_task_type(row: dict) -> str:
    task_type = str(row.get('task_type', '')).strip()
    if task_type:
        return task_type
    q, _ = _extract_question_answer(row)
    if re.search(r'choices?:\s*A', q, flags=re.I) or re.search(r'A[\).]\s*', q):
        return 'mcq_vqa'
    return 'open_vqa'


def load_source_rows(path: Path, source_group: str, limit: int | None = None) -> List[SourceRow]:
    rows: List[SourceRow] = []
    for idx, row in enumerate(read_jsonl(path)):
        q, a = _extract_question_answer(row)
        if not q or not a:
            continue
        category, _, _ = detect_task4_category(q, a)
        rows.append(SourceRow(
            row=row,
            row_id=str(row.get('id') or f'{source_group}_{idx}'),
            source_group=source_group,
            task_type=_infer_task_type(row),
            category=category,
            question=q,
            answer=a,
        ))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def count_task4_categories(rows: Iterable[SourceRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row.category:
            counts[row.category] = counts.get(row.category, 0) + 1
    return counts


def is_valid_close_ended_row(src: SourceRow) -> tuple[bool, str]:
    if src.task_type != 'mcq_vqa':
        return True, ''
    q = src.question
    a = src.answer
    option_count = 0
    for letter in ['A', 'B', 'C', 'D']:
        if re.search(rf'{letter}[\).:]', q):
            option_count += 1
    if option_count < 2:
        return False, 'mcq_without_options'
    if not re.match(r'^\s*[A-D](?:[\.|\)])?\s*', a) and normalize_yes_no_answer(a) not in {'yes', 'no'}:
        return False, 'mcq_answer_not_normalized'
    if src.category == 'Emphysema':
        txt = f"{q}\n{a}".lower()
        if 'subcutaneous emphysema' in txt:
            return False, 'emphysema_off_topic_subcutaneous'
        if 'lung nodule' in txt or 'differential diagnosis' in txt:
            return False, 'emphysema_off_topic_cross_task'
    return True, ''

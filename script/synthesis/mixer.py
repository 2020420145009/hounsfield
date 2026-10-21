from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import OLD_SOURCE_COUNTS, OLD_TASK_TYPE_COUNTS, TASK4_CATEGORY_MIN_TARGET
from .dataset_pool import SourceRow, count_task4_categories, is_valid_close_ended_row
from .io_utils import write_jsonl


def _allocate_by_ratio(total: int, ratio_map: Dict[str, int]) -> Dict[str, int]:
    denom = sum(ratio_map.values())
    base = {k: (total * v) // denom for k, v in ratio_map.items()}
    remainder = total - sum(base.values())
    order = sorted(ratio_map.items(), key=lambda kv: kv[1], reverse=True)
    idx = 0
    while remainder > 0:
        key = order[idx % len(order)][0]
        base[key] += 1
        remainder -= 1
        idx += 1
    return base


def _dedupe_key(src: SourceRow) -> tuple[str, str, str]:
    return (src.source_group, src.question.strip().lower(), src.answer.strip().lower())


def _priority_score(src: SourceRow) -> int:
    score = 0
    if src.category:
        score += 2
    if src.task_type == 'report':
        score += 1
    if src.category == 'Emphysema':
        txt = f"{src.question}\n{src.answer}".lower()
        if any(x in txt for x in ['centrilobular', 'paraseptal', 'panlobular', 'mild', 'moderate', 'severe', 'upper lobe', 'lower lobe', 'bilateral']):
            score += 5
        if normalize_yes_no_answer(src.answer) in {'yes', 'no'}:
            score -= 2
    return score


def _select_task4_minimum(rows: List[SourceRow], min_target: int) -> Tuple[List[SourceRow], List[SourceRow]]:
    by_cat = defaultdict(list)
    for row in rows:
        if row.category:
            by_cat[row.category].append(row)
    selected = []
    selected_keys = set()
    for cat, cat_rows in by_cat.items():
        ordered = sorted(cat_rows, key=_priority_score, reverse=True)
        kept = 0
        for row in ordered:
            key = _dedupe_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            kept += 1
            if kept >= min_target:
                break
    remaining = [row for row in rows if _dedupe_key(row) not in selected_keys]
    return selected, remaining


def build_mixed_dataset(
    *,
    ctrate_rows: List[SourceRow],
    m3d_rows: List[SourceRow],
    lidc_rows: List[SourceRow],
    output_jsonl: Path,
    output_summary: Path,
    total_rows: int = 2425,
    source_ratio: Dict[str, int] | None = None,
    task_type_ratio: Dict[str, int] | None = None,
    task4_min_target: int = TASK4_CATEGORY_MIN_TARGET,
) -> dict:
    source_ratio = source_ratio or OLD_SOURCE_COUNTS
    task_type_ratio = task_type_ratio or OLD_TASK_TYPE_COUNTS
    source_target = _allocate_by_ratio(total_rows, source_ratio)
    task_type_target = _allocate_by_ratio(total_rows, task_type_ratio)

    pools = {
        'ctrate_open': [r for r in ctrate_rows if is_valid_close_ended_row(r)[0]],
        'm3d_mcq': [r for r in m3d_rows if is_valid_close_ended_row(r)[0]],
        'lidc_mixed': [r for r in lidc_rows if is_valid_close_ended_row(r)[0]],
    }

    all_rows = pools['ctrate_open'] + pools['m3d_mcq'] + pools['lidc_mixed']
    preselected, remainder = _select_task4_minimum(all_rows, task4_min_target)

    selected = []
    selected_keys = set()
    source_counts = Counter()
    task_counts = Counter()

    def try_add(row: SourceRow) -> bool:
        key = _dedupe_key(row)
        if key in selected_keys:
            return False
        if source_counts[row.source_group] >= source_target.get(row.source_group, 0):
            return False
        if task_counts[row.task_type] >= task_type_target.get(row.task_type, 0):
            return False
        selected.append(row)
        selected_keys.add(key)
        source_counts[row.source_group] += 1
        task_counts[row.task_type] += 1
        return True

    for row in sorted(preselected, key=_priority_score, reverse=True):
        try_add(row)

    for row in sorted(remainder, key=_priority_score, reverse=True):
        if len(selected) >= total_rows:
            break
        try_add(row)

    # second pass: relax task-type limits first, then source limits if still short
    if len(selected) < total_rows:
        for row in sorted(remainder, key=_priority_score, reverse=True):
            if len(selected) >= total_rows:
                break
            key = _dedupe_key(row)
            if key in selected_keys:
                continue
            if source_counts[row.source_group] >= source_target.get(row.source_group, 0):
                continue
            selected.append(row)
            selected_keys.add(key)
            source_counts[row.source_group] += 1
            task_counts[row.task_type] += 1

    if len(selected) < total_rows:
        for row in sorted(remainder, key=_priority_score, reverse=True):
            if len(selected) >= total_rows:
                break
            key = _dedupe_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            source_counts[row.source_group] += 1
            task_counts[row.task_type] += 1

    write_jsonl(output_jsonl, [r.row for r in selected])
    summary = {
        'output': str(output_jsonl),
        'total_rows': len(selected),
        'source_target': source_target,
        'task_type_target': task_type_target,
        'source_counts': dict(source_counts),
        'task_type_counts': dict(task_counts),
        'task4_category_counts': count_task4_categories(selected),
        'task4_min_target': task4_min_target,
    }
    output_summary.write_text(__import__('json').dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def normalize_yes_no_answer(answer: str) -> str:
    text = str(answer or '').strip().lower()
    text = __import__('re').sub(r'^\s*[ab][\.|\)]\s*', '', text)
    if text.startswith('yes'):
        return 'yes'
    if text.startswith('no'):
        return 'no'
    return text

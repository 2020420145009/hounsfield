from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List

from .category_rules import emphysema_quality_flags, normalize_yes_no_answer
from .schema import SynthesisTask


def normalize_question_signature(question: str) -> str:
    text = str(question or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def apply_emphysema_diversity_policy(tasks: Iterable[SynthesisTask], max_generic_yes_no: int = 2) -> List[SynthesisTask]:
    kept: List[SynthesisTask] = []
    seen_signatures = set()
    generic_yes_no_count = 0
    grouped = defaultdict(list)

    for task in tasks:
        grouped[task.category].append(task)

    for category, category_tasks in grouped.items():
        if category != "Emphysema":
            kept.extend(category_tasks)
            continue

        # Keep detail-rich questions first.
        scored = []
        for task in category_tasks:
            flags = emphysema_quality_flags(task)
            score = 0
            if flags["has_valuable_detail"]:
                score += 10
            if flags["is_yes_no"]:
                score -= 2
            if len(task.answer.strip()) < 40:
                score -= 1
            scored.append((score, task))

        for _, task in sorted(scored, key=lambda x: x[0], reverse=True):
            signature = (normalize_question_signature(task.question), normalize_yes_no_answer(task.answer))
            flags = emphysema_quality_flags(task)

            if signature in seen_signatures:
                continue

            if flags["is_yes_no"] and not flags["has_valuable_detail"]:
                if generic_yes_no_count >= max_generic_yes_no:
                    continue
                generic_yes_no_count += 1

            seen_signatures.add(signature)
            kept.append(task)

    return kept


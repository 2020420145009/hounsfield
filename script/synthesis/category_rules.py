from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .schema import SynthesisTask

TASK4_CATEGORY_HINTS = {
    "Emphysema": ["emphysema", "emphysematous"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged cardiac silhouette"],
    "Medical material": [
        "medical material",
        "catheter",
        "central venous catheter",
        "drainage catheter",
        "chest tube",
        "endotracheal tube",
        "pacemaker",
        "stent",
        "prosthesis",
    ],
    "Arterial wall calcification": ["arterial wall calcification", "vascular calcification"],
    "Hiatal hernia": ["hiatal hernia"],
    "Peribronchial thickening": ["peribronchial thickening", "bronchial wall thickening"],
    "Coronary artery wall calcification": [
        "coronary artery wall calcification",
        "coronary artery calcification",
        "coronary calcification",
    ],
    "Pulmonary fibrotic sequela": ["pulmonary fibrotic sequela", "fibrotic sequela", "pulmonary fibrosis"],
}

EMPHYSEMA_NOISY_PATTERNS = [
    r"\bsubcutaneous emphysema\b",
    r"\bpneumothorax\b",
    r"\blung nodule\b",
    r"\bdifferential diagnosis\b",
]

EMPHYSEMA_VALUABLE_PATTERNS = [
    r"\bcentrilobular emphysema\b",
    r"\bpanlobular emphysema\b",
    r"\bparaseptal emphysema\b",
    r"\bmild emphysema\b",
    r"\bmoderate emphysema\b",
    r"\bsevere emphysema\b",
    r"\bupper lobe\b",
    r"\blower lobe\b",
    r"\bbilateral\b",
    r"\bdistribution\b",
]


def collect_category_hits(text: str) -> Dict[str, List[str]]:
    lowered = str(text or "").lower()
    hits: Dict[str, List[str]] = {}
    for category, hints in TASK4_CATEGORY_HINTS.items():
        matched = [hint for hint in hints if hint.lower() in lowered]
        if matched:
            hits[category] = matched
    return hits


def detect_task4_category(question: str, answer: str) -> Tuple[Optional[str], List[str], str]:
    q_hits = collect_category_hits(question)
    a_hits = collect_category_hits(answer)
    qa_hits = collect_category_hits(f"{question}\n{answer}")

    if len(q_hits) == 1:
        category = next(iter(q_hits))
        return category, q_hits[category], "question_unique"

    qa_without_mm = {k: v for k, v in qa_hits.items() if k != "Medical material"}
    if len(qa_without_mm) == 1:
        category = next(iter(qa_without_mm))
        return category, qa_without_mm[category], "combined_unique_non_medical_material"

    a_without_mm = {k: v for k, v in a_hits.items() if k != "Medical material"}
    if not q_hits and len(a_without_mm) == 1:
        category = next(iter(a_without_mm))
        return category, a_without_mm[category], "answer_unique_non_medical_material"

    return None, [], "unmatched"


def normalize_yes_no_answer(answer: str) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"^\s*[ab][\.\)]\s*", "", text)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def emphysema_quality_flags(task: SynthesisTask) -> dict:
    text = f"{task.question}\n{task.answer}".lower()
    answer_norm = normalize_yes_no_answer(task.answer)
    return {
        "is_yes_no": answer_norm in {"yes", "no"},
        "has_valuable_detail": any(re.search(p, text) for p in EMPHYSEMA_VALUABLE_PATTERNS),
        "has_noise": any(re.search(p, text) for p in EMPHYSEMA_NOISY_PATTERNS),
        "has_wrong_density_wording": "decrease in emphysematous density" in text,
    }


def should_drop_task(task: SynthesisTask) -> tuple[bool, str]:
    category = task.category or ""

    if task.task_type == "mcq_vqa":
        return True, "drop_mcq"
    if category == "Medical material":
        return True, "drop_medical_material"
    if category == "Coronary artery wall calcification":
        return True, "drop_coronary_calcification"
    if category == "Cardiomegaly" and len(task.answer.strip()) < 30:
        return True, "drop_short_cardiomegaly"

    if category == "Emphysema":
        flags = emphysema_quality_flags(task)
        if flags["has_noise"]:
            return True, "drop_noisy_emphysema"
        if flags["has_wrong_density_wording"]:
            return True, "drop_wrong_emphysema_density_statement"

    return False, ""


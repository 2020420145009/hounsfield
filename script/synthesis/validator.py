from __future__ import annotations

import json
from typing import Optional

from .schema import SynthesisTask, ValidationResult

REASONING_TYPES = {
    "Rapid Triage",
    "Standard Assessment",
    "Comprehensive Diagnosis",
}

EXPECTED_LINES = {
    "Rapid Triage": 4,
    "Standard Assessment": 5,
    "Comprehensive Diagnosis": 7,
}

QUESTION_TYPE_PREFIXES = {
    "Rapid Triage": (
        "Existence Judgment",
        "Single-Point Localization",
        "Single-Point Measurement",
        "Binary State",
    ),
    "Standard Assessment": (
        "Multidimensional Comprehensive Description",
        "Feature Description",
        "Anomaly Classification",
        "Regional Comparison",
    ),
    "Comprehensive Diagnosis": (
        "Differential Diagnosis",
        "Primary Diagnosis Integration",
        "Severity/Risk Assessment",
        "Clinical Significance Integration",
        "Next-Step Recommendation",
    ),
}


def extract_json_block(text: str) -> Optional[dict]:
    text = str(text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _normalize_reasoning(reasoning_obj) -> list[str]:
    if isinstance(reasoning_obj, list):
        return [str(x).strip() for x in reasoning_obj if str(x).strip()]
    text = str(reasoning_obj or "").strip()
    return [line.strip() for line in text.splitlines() if line.strip()]


def _valid_question_type(reasoning_type: str, question_type: str) -> bool:
    prefixes = QUESTION_TYPE_PREFIXES.get(reasoning_type, ())
    return any(question_type.startswith(prefix) for prefix in prefixes)


def validate_generated_dict(data: dict, task: SynthesisTask) -> ValidationResult:
    if str(data.get('status', '')).strip().lower() == 'reject':
        reason = str(data.get('reason', '')).strip() or 'model_declared_unfit'
        return ValidationResult(False, f'model_reject:{reason}')

    reasoning_type = str(data.get("type", "")).strip()
    dimension = str(data.get("dimension", "")).strip()
    qtype = str(data.get("question_type", "")).strip()
    question = str(data.get("q", "")).strip()
    answer = str(data.get("a", "")).strip()
    reasoning_lines = _normalize_reasoning(data.get("reasoning", ""))

    if reasoning_type not in REASONING_TYPES:
        return ValidationResult(False, "invalid_type")
    if task.dimension and reasoning_type != task.dimension:
        return ValidationResult(False, 'reasoning_type_mismatch')
    if not dimension:
        return ValidationResult(False, "empty_dimension")
    if not qtype:
        return ValidationResult(False, "empty_question_type")
    if not question:
        return ValidationResult(False, "empty_question")
    if not answer:
        return ValidationResult(False, "empty_answer")
    if not reasoning_lines:
        return ValidationResult(False, "empty_reasoning")
    if not _valid_question_type(reasoning_type, qtype):
        return ValidationResult(False, "invalid_question_type")

    if len(reasoning_lines) != EXPECTED_LINES[reasoning_type]:
        return ValidationResult(False, f"invalid_reasoning_lines type={reasoning_type}")
    expected_first = f"This is a {reasoning_type} question about {qtype}."
    if reasoning_lines[0] != expected_first:
        return ValidationResult(False, "bad_first_line")

    lower = "\n".join(reasoning_lines + [answer]).lower()
    bad_patterns = [
        "decrease in emphysematous density",
        "hidden supervision",
        "json object",
        "output format",
        "prompt rules",
    ]
    if any(p in lower for p in bad_patterns):
        return ValidationResult(False, "instruction_or_fact_leak")

    return ValidationResult(True, "")


def build_output_entry(task: SynthesisTask, data: dict, raw_output: str) -> dict:
    reasoning = data["reasoning"]
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(str(x).strip() for x in reasoning if str(x).strip())
    else:
        reasoning_text = str(reasoning).strip()
    answer = str(data["a"]).strip()
    return {
        "id": task.task_id,
        "image": task.image,
        "conversations": [
            {"from": "human", "value": str(data["q"]).strip()},
            {"from": "gpt", "value": f"<think>\n{reasoning_text}\n</think>\n<answer>\n{answer}\n</answer>"},
        ],
        "task_type": task.task_type,
        "question_type": str(data["question_type"]).strip(),
        "dimension": str(data["dimension"]).strip(),
        "_reasoning_type": str(data["type"]).strip(),
        "_dataset_source": task.source,
        "_category": task.category,
        "_raw_generation": raw_output,
    }

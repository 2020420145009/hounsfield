from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SynthesisTask:
    task_id: str
    source: str
    task_type: str
    image: str
    question: str
    answer: str
    dimension: Optional[str] = None
    question_type: Optional[str] = None
    hint_reasoning: str = ""
    category: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


@dataclass
class GeneratedSample:
    task: SynthesisTask
    dimension: str
    question_type: str
    reasoning: str
    answer: str
    raw_output: str = ""


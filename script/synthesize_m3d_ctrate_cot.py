
#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


PERSONAS = [
    "Persona: Fast-paced ER Doctor. Focus on direct, high-yield triage reasoning.",
    "Persona: Senior Radiology Professor. Focus on precise feature-to-conclusion deduction.",
    "Persona: Meticulous Diagnostician. Emphasize disciplined, evidence-grounded interpretation.",
]

QUESTION_TYPES_BY_DIMENSION = {
    "Rapid Triage": [
        "Existence Judgment (Presence/Absence)",
        "Single-Point Localization (Which Lobe/Segment/Side)",
        "Single-Point Measurement (Size/Thickness/Numerical Inquiry)",
        "Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)",
    ],
    "Standard Assessment": [
        "Multidimensional Comprehensive Description (Location + Morphology)",
        "Feature Description (Morphology, Density, Margins, Internal Structure)",
        "Anomaly Classification (Which subtype/pattern it belongs to)",
        "Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)",
    ],
    "Comprehensive Diagnosis": [
        "Differential Diagnosis (List the 1-2 most probable diagnoses)",
        "Primary Diagnosis Integration",
        "Severity/Risk Assessment",
        "Clinical Significance Integration",
        "Next-Step Recommendation",
    ],
}

EXPECTED_REASONING_LINES = {
    "Rapid Triage": 4,
    "Standard Assessment": 5,
    "Comprehensive Diagnosis": 7,
}

TASK4_FILTER_HINTS = {
    "Emphysema": ["emphysema", "emphysematous"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged cardiac silhouette", "heart size is increased"],
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
    "Pulmonary fibrotic sequela": [
        "pulmonary fibrotic sequela",
        "fibrotic sequela",
        "pulmonary fibrosis",
        "fibrotic changes",
    ],
}

SHORT_ANSWER_CHAR_THRESHOLD = 30

EXTERNAL_INFO_PATTERNS = [
    r"\breport\b",
    r"\bmetadata\b",
    r"\bjson\b",
    r"\bintegrated_meta\b",
    r"\bclinical diagnosis\b",
    r"\bscan context\b",
    r"\bfile name\b",
    r"\bpath\b",
    r"\bprompt\b",
    r"\bquestion states?\b",
    r"\boption [A-D]\b",
    r"\bchoice [A-D]\b",
    r"\bpreliminary diagnosis\b",
]

INSTRUCTION_LEAK_PATTERNS = [
    r"\bchoose exactly one\b",
    r"\breturn only\b",
    r"\boutput format\b",
    r"\bline [1-7]\b",
    r"\bfirst line\b",
    r"\bnon-empty lines?\b",
    r"\bpure json\b",
    r"\bjson object\b",
    r"\bslot_id\b",
    r"\bquestion_type\b",
    r"\bdimension must\b",
    r"\byou must generate\b",
]

REASONING_PREFIX_PATTERNS = [
    r"^Identify and describe the primary anatomical region/structures(?: relevant to the task)?\s*:?\s*",
    r"^Identify and describe the primary anatomical region\s*:?\s*",
    r"^Identify and describe the primary anatomy involved\s*:?\s*",
    r"^Identify and describe the primary anatomical location\s*:?\s*",
    r"^Formally observe and report key abnormal imaging features\s*:?\s*",
    r"^Formally observe and report the key imaging evidence(?: from the JSON)?\s*:?\s*",
    r"^Compare with normal anatomy, formally observe and report key abnormal imaging features\s*:?\s*",
    r"^Compare with normal anatomy to establish the pathological baseline\s*:?\s*",
    r"^Determine the primary abnormality type(?: or structured descriptive summary)?\s*:?\s*",
    r"^Evaluate the severity and clinical risk\s*:?\s*",
    r"^Output the definitive final clinical judgment\s*:?\s*",
    r"^Conclusion\s*:?\s*",
]

GLOBAL_RULES = """
[STRICT GROUNDING REQUIREMENTS]
1. The provided source evidence is only for supervision and sample construction.
2. The final question, reasoning, and answer must read as if they are grounded only in image-visible evidence.
3. Do not mention reports, metadata, JSON, paths, prompt rules, previous annotations, or hidden labels.
4. Do not infer from hidden supervision first and then narrate it as if directly observed.
5. Do not hallucinate anatomy, diagnosis, measurements, morphology, severity, or management details.

[CHAIN-OF-THOUGHT CLEANLINESS]
1. Do not copy synthesis instructions, formatting requirements, or reviewer text into reasoning.
2. Do not leave placeholders, policy text, or slot descriptions in the output.
3. Do not mention how the sample was generated or adjusted.

[REASONING FORMAT]
- Rapid Triage: exactly 4 numbered steps:
  1. This is a Rapid Triage question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Formally observe and report key abnormal imaging features.
  4. Conclusion.
- Standard Assessment: exactly 5 numbered steps:
  1. This is a Standard Assessment question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Compare with normal anatomy, formally observe and report key abnormal imaging features.
  4. Determine the primary abnormality type.
  5. Conclusion.
- Comprehensive Diagnosis: exactly 7 numbered steps:
  1. This is a Comprehensive Diagnosis question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Formally observe and report key abnormal imaging features.
  4. Compare with normal anatomy to establish the pathological baseline.
  5. Determine the primary abnormality type.
  6. Evaluate the severity and clinical risk.
  7. Output the definitive final clinical judgment.

[MCQ INTEGRITY]
1. If the task is MCQ, keep the question and answer exactly unchanged.
2. MCQ reasoning must explicitly traverse A), B), C), and D) before concluding.
3. Do not backsolve from the known answer text.

[STYLE]
1. The final reasoning content in the dataset should be plain sentences only.
2. Do not write step labels such as "Identify and describe the primary anatomical region/structures:".
3. Do not write numbered bullets like "1." or "2." in the final reasoning phrasing.
4. Keep only the substantive medical content of each reasoning step.

[OUTPUT FORMAT]
Return only one JSON object. No markdown. No explanations outside JSON.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize M3D and CT-RATE chain-of-thought datasets with source-specific adapters.")
    parser.add_argument("--mode", choices=["m3d_mcq", "m3d_open", "ctrate_mcq", "ctrate_open", "all"], default="all")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-devices", type=str, default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default="mp")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    parser.add_argument("--gpu-idle-util-threshold", type=float, default=10.0)
    parser.add_argument("--gpu-idle-mem-mb-threshold", type=float, default=2000.0)
    parser.add_argument("--gpu-check-interval-sec", type=int, default=30)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--m3d-mcq-input", type=Path, default=Path("/volume/med-train/users/shuyan/flemingvlcode/zyq/M3Dtrain_all_cleaned.jsonl"))
    parser.add_argument("--m3d-open-base-dir", type=Path, default=Path("/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy1"))
    parser.add_argument("--ctrate-root", type=Path, default=Path("/volume/med-train/data/CT-RATE-DATA"))
    parser.add_argument("--ctrate-mcq-input", type=Path, default=Path("/volume/med-train/data/CT-RATE-DATA/Synthesized_CoT_Dataset_ctrate/ct_rate_mcq_clean.jsonl"))
    parser.add_argument("--ctrate-open-input", type=Path, default=Path("/volume/med-train/data/CT-RATE-DATA/Synthesized_CoT_Dataset_ctrate/ct_rate_open_vqa_three_tiers_5000_15000_clean.jsonl"))
    parser.add_argument("--ctrate-clean-profile", choices=["none", "task4_analysis_v2"], default="none")
    parser.add_argument("--write-filter-audit", action="store_true")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    if args.gpu_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def resolve_tensor_parallel_size(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_count = max(1, len([x for x in visible.split(",") if x.strip()])) if visible else 1
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size > visible_count:
            raise SystemExit(
                f"tensor_parallel_size={args.tensor_parallel_size} exceeds visible GPU count {visible_count} "
                f"under CUDA_VISIBLE_DEVICES='{visible}'."
            )
        return args.tensor_parallel_size
    return visible_count


def get_selected_gpu_indices() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    return [x.strip() for x in visible.split(",") if x.strip()]


def wait_for_idle_gpus(args: argparse.Namespace) -> None:
    if not args.wait_for_idle_gpu:
        return
    gpu_ids = get_selected_gpu_indices()
    if not gpu_ids:
        return

    while True:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            print(f"Skipping GPU idle wait because nvidia-smi failed: {exc}")
            return

        stats = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            idx, util, mem = parts
            try:
                stats[idx] = (float(util), float(mem))
            except ValueError:
                continue

        busy = []
        for gpu_id in gpu_ids:
            util, mem = stats.get(gpu_id, (0.0, 0.0))
            if util > args.gpu_idle_util_threshold or mem > args.gpu_idle_mem_mb_threshold:
                busy.append((gpu_id, util, mem))

        if not busy:
            return

        busy_text = "; ".join(f"gpu {g}: util={u}%, mem={m}MB" for g, u, m in busy)
        print(f"Waiting for selected GPUs to become idle ({busy_text}).")
        time.sleep(args.gpu_check_interval_sec)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_completed_ids(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        item_id = row.get("id")
        if item_id:
            completed.add(item_id)
    return completed


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_block(text: str):
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
    if start == -1:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escaped = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = text[start : pos + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = text.find(open_ch, pos + 1)
                    if start == -1:
                        return None
                    depth = 0
                    in_str = False
                    escaped = False
                    continue
    return None


def normalize_reasoning(reasoning) -> str:
    if reasoning is None:
        return ""
    if isinstance(reasoning, list):
        reasoning = "\n".join(str(x).strip() for x in reasoning if str(x).strip())
    text = str(reasoning).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = "\n".join(str(x).strip() for x in parsed if str(x).strip())
        except Exception:
            pass
    text = re.sub(r"\s+(\d+\.\s+)", r"\n\1", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def reasoning_lines(reasoning: str) -> List[str]:
    return [line.strip() for line in normalize_reasoning(reasoning).splitlines() if line.strip()]


def sanitize_reasoning_line_for_output(line: str) -> str:
    cleaned = re.sub(r"^\d+\.\s*", "", str(line or "").strip())
    for pattern in REASONING_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def reasoning_lines_for_output(reasoning: str) -> List[str]:
    lines = []
    for line in reasoning_lines(reasoning):
        cleaned = sanitize_reasoning_line_for_output(line)
        if cleaned:
            lines.append(cleaned)
    return lines


def numbered_reasoning_lines(reasoning: str) -> List[str]:
    return [line for line in reasoning_lines(reasoning) if re.match(r"^\d+\.\s+", line)]


def parse_mcq_labels(question: str) -> List[str]:
    labels = re.findall(r"\b([A-D])\)", str(question or ""))
    if labels:
        return labels
    return re.findall(r"Choice\s+([A-D])\s*:", str(question or ""), flags=re.I)


def extract_choice_map(question: str) -> Dict[str, str]:
    text = str(question or "").strip()
    choice_matches = list(re.finditer(r"Choice\s+([A-D])\s*:\s*", text, flags=re.I))
    if choice_matches:
        result = {}
        for idx, match in enumerate(choice_matches):
            letter = match.group(1).upper()
            start = match.end()
            end = choice_matches[idx + 1].start() if idx + 1 < len(choice_matches) else len(text)
            result[letter] = text[start:end].strip(" .,;\n\t")
        return result

    abcd_matches = list(re.finditer(r"\b([A-D])\)\s*", text))
    result = {}
    for idx, match in enumerate(abcd_matches):
        letter = match.group(1).upper()
        start = match.end()
        end = abcd_matches[idx + 1].start() if idx + 1 < len(abcd_matches) else len(text)
        result[letter] = text[start:end].strip(" .,;\n\t")
    return result


def normalize_mcq_answer_to_question(answer: str, question: str) -> str:
    answer_text = str(answer or "").strip()
    if not answer_text:
        return answer_text
    choice_map = extract_choice_map(question)
    match = re.match(r"^\s*([A-D])(?:[\.\)])?\s*(.*)$", answer_text, flags=re.I)
    if not match:
        return answer_text
    letter = match.group(1).upper()
    suffix = match.group(2).strip()
    if suffix:
        return f"{letter}. {suffix}"
    if letter in choice_map:
        return f"{letter}. {choice_map[letter]}"
    return letter


def normalize_mcq_answer(answer: str) -> str:
    answer_text = str(answer or "").strip()
    if not answer_text:
        return answer_text
    match = re.match(r"^\s*([A-D])(?:[\.\)])?\s*(.*)$", answer_text, flags=re.I)
    if not match:
        return answer_text
    letter = match.group(1).upper()
    suffix = match.group(2).strip()
    return f"{letter}. {suffix}" if suffix else letter


def parse_traversed_labels(reasoning: str) -> List[str]:
    labels = []
    for line in reasoning_lines(reasoning)[1:]:
        for match in re.finditer(r"(?:\boption\s*|\bchoice\s*)?([A-D])[\):.]", line, flags=re.I):
            labels.append(match.group(1).upper())
    return labels


def contains_external_info(text: str) -> bool:
    s = str(text or "")
    return any(re.search(pattern, s, flags=re.I) for pattern in EXTERNAL_INFO_PATTERNS)


def contains_instruction_leak(text: str) -> bool:
    s = str(text or "")
    return any(re.search(pattern, s, flags=re.I) for pattern in INSTRUCTION_LEAK_PATTERNS)


def extract_think_answer_blocks(text: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.S | re.I)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.S | re.I)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else raw
    return think, answer


def collect_task4_filter_hits(text: str) -> Dict[str, List[str]]:
    lowered = str(text or "").lower()
    hits: Dict[str, List[str]] = {}
    for category, hints in TASK4_FILTER_HINTS.items():
        matched = [hint for hint in hints if hint.lower() in lowered]
        if matched:
            hits[category] = matched
    return hits


def detect_task4_filter_category(question: str, answer: str) -> Tuple[Optional[str], List[str], str]:
    question_hits = collect_task4_filter_hits(question)
    answer_hits = collect_task4_filter_hits(answer)
    combined_hits = collect_task4_filter_hits(f"{question}\n{answer}")

    if len(question_hits) == 1:
        category = next(iter(question_hits))
        return category, question_hits[category], "question_unique"

    combined_without_mm = {k: v for k, v in combined_hits.items() if k != "Medical material"}
    if len(combined_without_mm) == 1:
        category = next(iter(combined_without_mm))
        return category, combined_without_mm[category], "combined_unique_non_medical_material"

    answer_without_mm = {k: v for k, v in answer_hits.items() if k != "Medical material"}
    if len(answer_without_mm) == 1 and not question_hits:
        category = next(iter(answer_without_mm))
        return category, answer_without_mm[category], "answer_unique_non_medical_material"

    if len(question_hits) == 1 and "Medical material" in question_hits:
        return "Medical material", question_hits["Medical material"], "question_unique_medical_material"

    return None, [], "unmatched"


def is_short_answer(answer: str) -> bool:
    return len(str(answer or "").strip()) < SHORT_ANSWER_CHAR_THRESHOLD


def should_filter_ctrate_task(task: Dict, args: argparse.Namespace) -> Optional[dict]:
    if args.ctrate_clean_profile != "task4_analysis_v2":
        return None

    question = str(task.get("question", ""))
    answer = str(task.get("answer", ""))
    category, matched_hints, match_strategy = detect_task4_filter_category(question, answer)
    inferred_qtype = fallback_question_type(question, task.get("task_type", "open_vqa"), task.get("dimension", "Rapid Triage"))
    is_mcq = task.get("task_type") == "mcq_vqa" or len(parse_mcq_labels(question)) >= 2
    short_answer = is_short_answer(answer)

    if is_mcq:
        return {
            "reason": "drop_mcq_under_task4_analysis_v2",
            "category": category,
            "matched_keywords": matched_hints,
            "match_strategy": match_strategy,
            "question_type": inferred_qtype,
        }

    if category == "Medical material":
        return {
            "reason": "drop_medical_material_after_task4_analysis_v2",
            "category": category,
            "matched_keywords": matched_hints,
            "match_strategy": match_strategy,
            "question_type": inferred_qtype,
        }

    if category == "Coronary artery wall calcification":
        return {
            "reason": "drop_coronary_calcification_after_task4_analysis_v2",
            "category": category,
            "matched_keywords": matched_hints,
            "match_strategy": match_strategy,
            "question_type": inferred_qtype,
        }

    if category == "Cardiomegaly" and short_answer:
        return {
            "reason": "drop_short_cardiomegaly_answer",
            "category": category,
            "matched_keywords": matched_hints,
            "match_strategy": match_strategy,
            "question_type": inferred_qtype,
        }

    if category == "Emphysema" and inferred_qtype in {
        "Differential Diagnosis (List the 1-2 most probable diagnoses)",
        "Primary Diagnosis Integration",
        "Clinical Significance Integration",
    }:
        return {
            "reason": "drop_complex_emphysema_question_type",
            "category": category,
            "matched_keywords": matched_hints,
            "match_strategy": match_strategy,
            "question_type": inferred_qtype,
        }

    return None


def normalize_question_type(question_type: str, dimension: str) -> str:
    qt = (question_type or "").strip()
    if qt in QUESTION_TYPES_BY_DIMENSION[dimension]:
        return qt
    aliases = {
        "Existence Judgment": "Existence Judgment (Presence/Absence)",
        "Single-Point Localization": "Single-Point Localization (Which Lobe/Segment/Side)",
        "Single-Point Measurement": "Single-Point Measurement (Size/Thickness/Numerical Inquiry)",
        "Binary State": "Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)",
        "Multidimensional Comprehensive Description": "Multidimensional Comprehensive Description (Location + Morphology)",
        "Feature Description": "Feature Description (Morphology, Density, Margins, Internal Structure)",
        "Anomaly Classification": "Anomaly Classification (Which subtype/pattern it belongs to)",
        "Regional Comparison": "Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)",
        "Differential Diagnosis": "Differential Diagnosis (List the 1-2 most probable diagnoses)",
    }
    return aliases.get(qt, qt)


def fallback_question_type(question: str, task_type: str, dimension: str) -> str:
    ql = (question or "").lower()
    option_count = len(parse_mcq_labels(question))
    if dimension == "Rapid Triage":
        if re.search(r"\b(mm|cm|diameter|size|measure|volume|largest|smallest|thick|thickness)\b", ql):
            return "Single-Point Measurement (Size/Thickness/Numerical Inquiry)"
        if re.search(r"\bwhere|which\s+(lobe|segment|side|organ)|left|right|upper|lower|location|located\b", ql):
            return "Single-Point Localization (Which Lobe/Segment/Side)"
        if option_count == 2 or re.search(r"\b(is|are|does|do|whether|normal|abnormal|yes|no)\b", ql):
            return "Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)"
        return "Existence Judgment (Presence/Absence)"
    if dimension == "Standard Assessment":
        if re.search(r"\bcompare|ratio|relative|versus|than\b", ql):
            return "Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)"
        if re.search(r"\bclassif|pattern|type|subtype\b", ql):
            return "Anomaly Classification (Which subtype/pattern it belongs to)"
        if re.search(r"\bshape|margin|texture|spiculation|feature|appearance|describe|morphology\b", ql):
            return "Feature Description (Morphology, Density, Margins, Internal Structure)"
        return "Multidimensional Comprehensive Description (Location + Morphology)"
    if task_type == "report" or re.search(r"\brecommend|next step|follow-up|biopsy|management\b", ql):
        return "Next-Step Recommendation"
    if re.search(r"\brisk|severity|high-risk|concern\b", ql):
        return "Severity/Risk Assessment"
    if re.search(r"\bsignificance|implication\b", ql):
        return "Clinical Significance Integration"
    if re.search(r"\bdifferential|diagnosis|likely\b", ql):
        return "Differential Diagnosis (List the 1-2 most probable diagnoses)"
    return "Primary Diagnosis Integration"


def repair_reasoning_first_line(reasoning: str, dimension: str, question_type: str) -> str:
    lines = numbered_reasoning_lines(reasoning)
    if not lines:
        lines = reasoning_lines(reasoning)
    if not lines:
        return ""
    lines[0] = f"1. This is a {dimension} question about {question_type}."
    return "\n".join(lines)


def is_valid_reasoning(reasoning: str, dimension: str, question_type: str) -> bool:
    lines = numbered_reasoning_lines(reasoning)
    if len(lines) < EXPECTED_REASONING_LINES[dimension]:
        return False
    if lines[0] != f"1. This is a {dimension} question about {question_type}.":
        return False
    for idx, line in enumerate(lines[: EXPECTED_REASONING_LINES[dimension]], start=1):
        if not re.match(rf"^{idx}\.\s+", line):
            return False
    return True


def get_persona(sample_id: str) -> str:
    idx = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest(), 16) % len(PERSONAS)
    return PERSONAS[idx]


def resolve_ctrate_image_path(image_value: str, ctrate_root: Path) -> str:
    image_value = str(image_value or "").strip()
    if image_value.startswith("/"):
        return image_value
    if image_value.endswith(".nii.gz") and image_value.startswith("train_"):
        stem = image_value[:-7]
        parts = stem.split("_")
        if len(parts) >= 3:
            study = "_".join(parts[:2])
            series = "_".join(parts[:3])
            return str(ctrate_root / "dataset" / "train_fixed" / study / series / image_value)
    if image_value.endswith(".nii.gz") and image_value.startswith("valid_"):
        stem = image_value[:-7]
        parts = stem.split("_")
        if len(parts) >= 3:
            study = "_".join(parts[:2])
            series = "_".join(parts[:3])
            return str(ctrate_root / "dataset" / "valid_fixed" / study / series / image_value)
    return image_value


def infer_m3d_report_path(image_path: str) -> Optional[Path]:
    image = Path(str(image_path))
    if not image.parts:
        return None
    return image.parent / "text.txt"


def pick_m3d_open_image(case_dir: Path) -> Optional[str]:
    preferred = [
        "Axial_C__portal_venous_phase.npy",
        "Axial_non_contrast.npy",
        "Axial_C__arterial_phase.npy",
        "Axial_oral_contrast.npy",
    ]
    for name in preferred:
        candidate = case_dir / name
        if candidate.exists():
            return str(candidate)
    npys = sorted(case_dir.glob("*.npy"))
    return str(npys[0]) if npys else None


def infer_ctrate_dimension(row: dict) -> str:
    row_id = str(row.get("id", ""))
    think, _ = extract_think_answer_blocks(row.get("conversations", [{}, {}])[1].get("value", ""))
    first = reasoning_lines(think)
    first_line = first[0] if first else ""
    if "_L1" in row_id or "Rapid Triage" in first_line:
        return "Rapid Triage"
    if "_L2" in row_id or "Standard Assessment" in first_line:
        return "Standard Assessment"
    if "_L3" in row_id or "Comprehensive Diagnosis" in first_line:
        return "Comprehensive Diagnosis"
    return "Rapid Triage"


def build_m3d_mcq_tasks(args: argparse.Namespace) -> List[Dict]:
    tasks = []
    for row in read_jsonl(args.m3d_mcq_input):
        image = str(row.get("image", ""))
        report_path = infer_m3d_report_path(image)
        report_text = None
        source = "m3d_mcq_qa_only"
        if report_path and report_path.exists():
            candidate = report_path.read_text(encoding="utf-8").strip()
            if candidate:
                report_text = candidate
                source = "m3d_mcq"

        convs = row.get("conversations", [])
        qa_index = 0
        for idx in range(0, len(convs) - 1, 2):
            human = convs[idx]
            gpt = convs[idx + 1]
            if human.get("from") != "human" or gpt.get("from") != "gpt":
                continue
            question = str(human.get("value", "")).strip()
            answer = str(gpt.get("value", "")).strip()
            if not question or not answer:
                continue
            qa_index += 1
            tasks.append(
                {
                    "id": f"M3D-{row.get('id')}_mcq_vqa_{qa_index}",
                    "source": source,
                    "task_type": "mcq_vqa",
                    "dimension": "Rapid Triage",
                    "image": image,
                    "report_text": report_text,
                    "question": question,
                    "answer": answer,
                }
            )
            if args.test_limit is not None and len(tasks) >= args.test_limit:
                return tasks
    return tasks


def build_m3d_open_tasks(args: argparse.Namespace) -> List[Dict]:
    tasks = []
    if not args.m3d_open_base_dir.exists():
        return tasks
    for case_dir in sorted(args.m3d_open_base_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        report_path = case_dir / "text.txt"
        if not report_path.exists():
            continue
        image = pick_m3d_open_image(case_dir)
        if not image:
            continue
        report_text = report_path.read_text(encoding="utf-8").strip()
        if not report_text:
            continue
        tasks.append(
            {
                "id": f"M3D-{case_dir.name}_open_vqa",
                "source": "m3d_open",
                "task_type": "open_vqa",
                "image": image,
                "report_text": report_text,
            }
        )
        if args.test_limit is not None and len(tasks) >= args.test_limit:
            return tasks
    return tasks


def build_ctrate_tasks(input_path: Path, task_type: str, args: argparse.Namespace) -> List[Dict]:
    tasks = []
    filtered = []
    for row in read_jsonl(input_path):
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue
        question = str(convs[0].get("value", "")).strip()
        think, answer = extract_think_answer_blocks(convs[1].get("value", ""))
        if not question or not answer:
            continue
        task = {
            "id": str(row.get("id")),
            "source": "ctrate",
            "task_type": task_type,
            "dimension": infer_ctrate_dimension(row),
            "image": resolve_ctrate_image_path(row.get("image", ""), args.ctrate_root),
            "question": question,
            "answer": answer,
            "hint_reasoning": think,
        }
        filter_detail = should_filter_ctrate_task(task, args)
        if filter_detail is not None:
            filter_record = {
                "id": task["id"],
                "task_type": task_type,
                "question": question,
                "answer": answer,
                **filter_detail,
            }
            filtered.append(filter_record)
            continue
        tasks.append(task)
        if args.test_limit is not None and len(tasks) >= args.test_limit:
            break

    if filtered:
        reason_counts: Dict[str, int] = {}
        for row in filtered:
            reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
        print(f"Filtered {len(filtered)} CT-RATE {task_type} samples under profile {args.ctrate_clean_profile}: {reason_counts}")
        if args.write_filter_audit:
            audit_path = args.output_dir / f"ctrate_{task_type}_filtered.audit.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("w", encoding="utf-8") as f:
                for row in filtered:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tasks


def build_prompt_for_task(task: Dict) -> str:
    persona = get_persona(task["id"])
    source = task["source"]
    task_type = task["task_type"]
    prompt = [GLOBAL_RULES, f"[PERSONA] {persona}"]

    if source == "m3d_mcq":
        allowed_qtypes = "\n".join(f"- {x}" for x in QUESTION_TYPES_BY_DIMENSION["Rapid Triage"])
        prompt.append(
            f"""
[SOURCE TYPE]
M3D raw MCQ source. This source provides a radiology report plus a fixed question-answer pair.

[TASK]
Generate one corrected MCQ CoT item.
1. Keep q exactly unchanged.
2. Keep a exactly unchanged.
3. dimension must be exactly Rapid Triage.
4. Choose one question_type only from:
{allowed_qtypes}
5. reasoning must be exactly 4 numbered steps.
6. reasoning must be image-grounded in wording, even though the report is provided as hidden supervision.
7. Explicitly compare A), B), C), and D) before the conclusion.

[OUTPUT JSON]
{{
  "type": "mcq_vqa",
  "dimension": "Rapid Triage",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}}
""".strip()
        )
    elif source == "m3d_mcq_qa_only":
        allowed_qtypes = "\n".join(f"- {x}" for x in QUESTION_TYPES_BY_DIMENSION["Rapid Triage"])
        prompt.append(
            f"""
[SOURCE TYPE]
M3D raw MCQ source without report-side supervision available in the current environment.

[TASK]
Generate one corrected MCQ CoT item.
1. Keep q exactly unchanged.
2. Keep a exactly unchanged.
3. dimension must be exactly Rapid Triage.
4. Choose one question_type only from:
{allowed_qtypes}
5. reasoning must be exactly 4 numbered steps.
6. reasoning must read as image-grounded, concise, and medically coherent.
7. Explicitly compare A), B), C), and D) before the conclusion.
8. Do not mention that hidden supervision is unavailable.

[OUTPUT JSON]
{{
  "type": "mcq_vqa",
  "dimension": "Rapid Triage",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}}
""".strip()
        )
    elif source == "m3d_open":
        prompt.append(
            """
[SOURCE TYPE]
M3D report-driven open question synthesis. This source provides a radiology report but no fixed question.

[TASK]
Generate one open_vqa item.
1. Choose the most suitable dimension from Rapid Triage, Standard Assessment, or Comprehensive Diagnosis.
2. Choose exactly one question_type allowed under that dimension.
3. q must be a concrete image-resolvable clinical question, not a generic report request.
4. a must directly answer q and remain strictly supported by the provided findings.
5. reasoning must follow the exact numbered-step template for the selected dimension.

[OUTPUT JSON]
{
  "type": "open_vqa",
  "dimension": "...",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}
""".strip()
        )
    else:
        dimension = task["dimension"]
        allowed_qtypes = "\n".join(f"- {x}" for x in QUESTION_TYPES_BY_DIMENSION[dimension])
        task_text = "MCQ" if task_type == "mcq_vqa" else "open_vqa"
        extra = ""
        if task_type == "mcq_vqa":
            extra = "\n5. Explicitly compare A), B), C), and D) before the conclusion.\n6. Keep q and a exactly unchanged."
        else:
            extra = "\n5. Keep q and a exactly unchanged."
        prompt.append(
            f"""
[SOURCE TYPE]
CT-RATE QA-driven source. This source provides an existing question, a canonical answer, and possibly a short prior rationale, but not a report file.

[TASK]
Generate one corrected {task_text} CoT item.
1. dimension must be exactly {dimension}.
2. Choose exactly one question_type allowed under this dimension:
{allowed_qtypes}
3. Rewrite reasoning to be cleaner, more specific, and image-grounded in wording.
4. Do not mention prior annotations or hidden supervision.{extra}

[OUTPUT JSON]
{{
  "type": "{task_type}",
  "dimension": "{dimension}",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}}
""".strip()
        )
    return "\n\n".join(prompt)


def build_user_content(task: Dict) -> str:
    if task["source"] == "m3d_mcq":
        return f"Hidden supervision report:\n{task['report_text']}\n\nFixed question:\n{task['question']}\n\nFixed answer:\n{task['answer']}"
    if task["source"] == "m3d_mcq_qa_only":
        return f"Fixed question:\n{task['question']}\n\nFixed answer:\n{task['answer']}"
    if task["source"] == "m3d_open":
        return f"Hidden supervision report:\n{task['report_text']}"
    return (
        f"Fixed question:\n{task['question']}\n\n"
        f"Fixed answer:\n{task['answer']}\n\n"
        f"Prior short rationale hint:\n{task.get('hint_reasoning', '')}"
    )


def build_reflection_prompt(task: Dict) -> str:
    rules = [
        "1. Output must be one JSON object only.",
        f"2. type must be exactly {task['task_type']}.",
        "3. reasoning must not mention report text, metadata, JSON, prompt rules, source annotations, or hidden supervision.",
        "4. reasoning must not contain template instructions or reviewer text.",
        "5. reasoning must use consecutive numbered steps starting from 1.",
    ]
    if task["source"] == "m3d_open":
        rules.extend(
            [
                "6. dimension must be one of Rapid Triage, Standard Assessment, Comprehensive Diagnosis.",
                "7. question_type must belong to the chosen dimension.",
                "8. q must be an open clinical question, not a generic report request.",
            ]
        )
    else:
        rules.extend(
            [
                f"6. dimension must be exactly {task['dimension']}.",
                f"7. q must remain exactly unchanged: {task['question']}",
                f"8. a must remain exactly unchanged: {task['answer']}",
            ]
        )
        if task["task_type"] == "mcq_vqa":
            rules.append("9. reasoning must explicitly traverse A), B), C), and D) before the conclusion.")
    return "Review your JSON against these rules:\n" + "\n".join(rules) + "\nIf violated, rewrite. Output only pure JSON."


def validate_item(item: dict, task: Dict) -> Optional[str]:
    if item.get("type") != task["task_type"]:
        return f"type_mismatch expected={task['task_type']} got={item.get('type')}"

    dimension = item.get("dimension", "").strip()
    if task["source"] == "m3d_open":
        if dimension not in QUESTION_TYPES_BY_DIMENSION:
            return f"invalid_dimension {dimension}"
    else:
        if dimension != task["dimension"]:
            return f"dimension_mismatch expected={task['dimension']} got={dimension}"

    question_type = normalize_question_type(item.get("question_type", "").strip(), dimension)
    if question_type not in QUESTION_TYPES_BY_DIMENSION[dimension]:
        question_type = fallback_question_type(item.get("q", ""), task["task_type"], dimension)
    item["question_type"] = question_type

    if task["task_type"] == "mcq_vqa":
        item["a"] = normalize_mcq_answer_to_question(item.get("a", ""), task["question"])
        expected_answer = normalize_mcq_answer_to_question(task["answer"], task["question"])

    if task["source"] != "m3d_open":
        if item.get("q", "").strip() != task["question"]:
            return "question_changed"
        if task["task_type"] == "mcq_vqa":
            if normalize_mcq_answer(item.get("a", "")).strip() != normalize_mcq_answer(expected_answer).strip():
                return "answer_changed"
        elif item.get("a", "").strip() != task["answer"]:
            return "answer_changed"

    item["reasoning"] = repair_reasoning_first_line(item.get("reasoning", ""), dimension, question_type)
    if not item["reasoning"].strip():
        return "empty_reasoning"
    if not is_valid_reasoning(item["reasoning"], dimension, question_type):
        return f"invalid_reasoning_lines dimension={dimension}"

    reasoning_and_answer = "\n".join([str(item.get("reasoning", "")), str(item.get("a", ""))])
    if contains_external_info(reasoning_and_answer):
        return "external_info_leak"
    if contains_instruction_leak(reasoning_and_answer):
        return "instruction_leak"

    if task["task_type"] == "mcq_vqa":
        labels = parse_mcq_labels(item.get("q", ""))
        if labels and labels != ["A", "B", "C", "D"]:
            return f"invalid_mcq_options labels={labels}"
    return None


def build_output_entry(task: Dict, item: dict) -> dict:
    question_type = normalize_question_type(item["question_type"], item["dimension"])
    reasoning = repair_reasoning_first_line(item["reasoning"], item["dimension"], question_type)
    reasoning_output = "\n".join(reasoning_lines_for_output(reasoning))
    return {
        "id": task["id"],
        "image": task["image"],
        "conversations": [
            {"from": "human", "value": item["q"].strip()},
            {"from": "gpt", "value": f"<think>\n{reasoning_output}\n</think>\n<answer>\n{item['a'].strip()}\n</answer>"},
        ],
        "task_type": task["task_type"],
        "question_type": question_type,
        "dimension": item["dimension"],
        "_dataset_source": task["source"],
    }


def output_path_for_mode(mode: str, output_dir: Path) -> Path:
    names = {
        "m3d_mcq": "m3d_mcq_cot.jsonl",
        "m3d_open": "m3d_open_cot.jsonl",
        "ctrate_mcq": "ctrate_mcq_cot.jsonl",
        "ctrate_open": "ctrate_open_cot.jsonl",
    }
    return output_dir / names[mode]


def state_path_for_mode(mode: str, output_dir: Path) -> Path:
    return output_dir / f"{mode}.resume_state.json"


def failure_path_for_mode(mode: str, output_dir: Path) -> Path:
    return output_dir / f"{mode}.failed.jsonl"


def raw_failure_path_for_mode(mode: str, output_dir: Path) -> Path:
    return output_dir / f"{mode}.failed_raw.jsonl"


def build_tasks_for_mode(mode: str, args: argparse.Namespace) -> List[Dict]:
    if mode == "m3d_mcq":
        return build_m3d_mcq_tasks(args)
    if mode == "m3d_open":
        return build_m3d_open_tasks(args)
    if mode == "ctrate_mcq":
        return build_ctrate_tasks(args.ctrate_mcq_input, "mcq_vqa", args)
    if mode == "ctrate_open":
        return build_ctrate_tasks(args.ctrate_open_input, "open_vqa", args)
    raise ValueError(mode)


def process_mode(
    mode: str,
    llm: LLM,
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    output_path = output_path_for_mode(mode, args.output_dir)
    state_path = state_path_for_mode(mode, args.output_dir)
    failed_path = failure_path_for_mode(mode, args.output_dir)
    raw_failed_path = raw_failure_path_for_mode(mode, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path) if args.resume else set()
    prior_state = load_state(state_path) if args.resume else {}

    tasks = build_tasks_for_mode(mode, args)
    total_discovered = len(tasks)
    if completed:
        tasks = [task for task in tasks if task["id"] not in completed]

    if not tasks:
        state = {
            "mode": mode,
            "resume_enabled": args.resume,
            "status": "complete",
            "total_discovered": total_discovered,
            "already_completed": len(completed),
            "pending": 0,
            "saved_this_run": 0,
            "failed_this_run": 0,
            "output_path": str(output_path),
            "failed_path": str(failed_path),
            "last_completed_id": sorted(completed)[-1] if completed else None,
        }
        save_state(state_path, state)
        print(f"{mode}: no pending tasks")
        return 0, 0

    file_mode = "a" if args.resume and output_path.exists() else "w"
    saved = 0
    failed = 0
    failed_details = []
    last_completed_id = prior_state.get("last_completed_id")

    state = {
        "mode": mode,
        "resume_enabled": args.resume,
        "status": "running",
        "total_discovered": total_discovered,
        "already_completed": len(completed),
        "pending": len(tasks),
        "saved_this_run": 0,
        "failed_this_run": 0,
        "output_path": str(output_path),
        "failed_path": str(failed_path),
        "last_completed_id": last_completed_id,
    }
    save_state(state_path, state)

    with output_path.open(file_mode, encoding="utf-8") as fout:
        failed_file_mode = "a" if args.resume and failed_path.exists() else "w"
        ffail = failed_path.open(failed_file_mode, encoding="utf-8")
        raw_failed_file_mode = "a" if args.resume and raw_failed_path.exists() else "w"
        fraw = raw_failed_path.open(raw_failed_file_mode, encoding="utf-8")
        pbar = tqdm(total=len(tasks), desc=f"Synthesize {mode}", unit="item")
        try:
            for batch_start in range(0, len(tasks), args.batch_size):
                batch = tasks[batch_start : batch_start + args.batch_size]
                batch_messages = [
                    [
                        {"role": "system", "content": build_prompt_for_task(task)},
                        {"role": "user", "content": build_user_content(task)},
                    ]
                    for task in batch
                ]

                pass1_outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)
                reflection_messages = []
                for task, output, original_messages in zip(batch, pass1_outputs, batch_messages):
                    history = list(original_messages)
                    history.append({"role": "assistant", "content": output.outputs[0].text})
                    history.append({"role": "user", "content": build_reflection_prompt(task)})
                    reflection_messages.append(history)

                pass2_outputs = llm.chat(messages=reflection_messages, sampling_params=sampling_params, use_tqdm=False)
                for task, output in zip(batch, pass2_outputs):
                    data = extract_json_block(output.outputs[0].text)
                    if not isinstance(data, dict):
                        failed += 1
                        detail = {"id": task["id"], "reason": "json_parse_failed"}
                        failed_details.append(detail)
                        ffail.write(json.dumps(detail, ensure_ascii=False) + "\n")
                        fraw.write(
                            json.dumps(
                                {
                                    "id": task["id"],
                                    "reason": "json_parse_failed",
                                    "raw_output": output.outputs[0].text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    fail_reason = validate_item(data, task)
                    if fail_reason:
                        failed += 1
                        detail = {"id": task["id"], "reason": fail_reason}
                        failed_details.append(detail)
                        ffail.write(json.dumps(detail, ensure_ascii=False) + "\n")
                        fraw.write(
                            json.dumps(
                                {
                                    "id": task["id"],
                                    "reason": fail_reason,
                                    "raw_output": output.outputs[0].text,
                                    "parsed_output": data,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    fout.write(json.dumps(build_output_entry(task, data), ensure_ascii=False) + "\n")
                    saved += 1
                    last_completed_id = task["id"]

                fout.flush()
                ffail.flush()
                fraw.flush()
                pbar.update(len(batch))

                state.update(
                    {
                        "status": "running",
                        "saved_this_run": saved,
                        "failed_this_run": failed,
                        "last_completed_id": last_completed_id,
                        "remaining_estimate": max(0, len(tasks) - (batch_start + len(batch))),
                    }
                )
                save_state(state_path, state)
        except KeyboardInterrupt:
            state.update(
                {
                    "status": "interrupted",
                    "saved_this_run": saved,
                    "failed_this_run": failed,
                    "last_completed_id": last_completed_id,
                }
            )
            save_state(state_path, state)
            pbar.close()
            ffail.close()
            fraw.close()
            print(f"{mode}: interrupted, progress saved, rerun with the same command to continue")
            raise
        pbar.close()
        ffail.close()
        fraw.close()

    state.update(
        {
            "status": "complete",
            "saved_this_run": saved,
            "failed_this_run": failed,
            "last_completed_id": last_completed_id,
            "remaining_estimate": 0,
        }
    )
    save_state(state_path, state)

    print(f"{mode}: saved={saved}, failed={failed}, output={output_path}, state={state_path}")
    return saved, failed


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    wait_for_idle_gpus(args)
    tensor_parallel_size = resolve_tensor_parallel_size(args)

    print(
        f"Runtime config -> mode={args.mode}, output_dir={args.output_dir}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}, "
        f"tensor_parallel_size={tensor_parallel_size}, backend={args.distributed_executor_backend}, test_limit={args.test_limit}"
    )

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    modes = [args.mode] if args.mode != "all" else ["m3d_mcq", "m3d_open", "ctrate_mcq", "ctrate_open"]
    total_saved = 0
    total_failed = 0
    for mode in modes:
        saved, failed = process_mode(mode, llm, sampling_params, args)
        total_saved += saved
        total_failed += failed

    print(f"All done -> saved={total_saved}, failed={total_failed}")


if __name__ == "__main__":
    main()

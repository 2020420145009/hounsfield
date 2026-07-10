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
from typing import Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams


DEFAULT_BASE_DIR = Path("/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset")
DEFAULT_MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_OUTPUT_DIR = Path(
    "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/generated_lidc_strict_cot_v2"
)
DEFAULT_OUTPUT_OPEN_NAME = "medical_lidc_open_vqa_dataset_strict_cot_v2.jsonl"
DEFAULT_OUTPUT_MCQ_NAME = "medical_lidc_mcq_vqa_dataset_strict_cot_v2.jsonl"
DEFAULT_OUTPUT_REPORT_NAME = "medical_lidc_report_dataset_strict_cot_v2.jsonl"
DEFAULT_OUTPUT_ALL_NAME = "medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl"


PERSONAS = [
    "Persona: Fast-paced ER Doctor. Focus on direct, high-yield triage reasoning.",
    "Persona: Senior Radiology Professor. Focus on precise feature-to-conclusion deduction.",
    "Persona: Meticulous Diagnostician. Emphasize disciplined, evidence-grounded interpretation.",
]

DIMENSION_TRIGGER = {
    "Rapid Triage": "[Perform Rapid Triage]",
    "Standard Assessment": "[Perform Standard Assessment]",
    "Comprehensive Diagnosis": "[Perform Comprehensive Diagnosis]",
}

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

GLOBAL_RULES = """
[STRICT GROUND-TRUTH REQUIREMENTS]
1. The integrated_meta JSON is the only structured source available to design the sample.
2. The final question, reasoning, and answer must be written as if they are grounded only in image-visible evidence.
3. Do not mention metadata, report text, labels, file names, scan context, prompt instructions, or any non-image source in q, reasoning, or a.
4. Do not infer from metadata first and then claim the conclusion was obtained from the image.
5. Do not hallucinate lobe location, diagnosis, measurements, or morphology absent from the structured case data.

[MCQ REASONING INTEGRITY]
1. For mcq_vqa, reasoning must examine the options explicitly before concluding.
2. Reasoning must traverse A), B), C), and D) rather than jumping directly to the answer.
3. Do not backsolve from the known correct answer text.
4. Distractors must remain plausible and medically coherent.

[CHAIN-OF-THOUGHT CLEANLINESS]
1. Do not copy synthesis instructions, formatting requirements, or reviewer text into the reasoning.
2. Do not leave template placeholders, slot descriptions, or policy text inside reasoning.
3. Do not mention JSON fields, prompt rules, or how the sample was synthesized.

[TASK TYPES]
- open_vqa: an open-ended imaging question with one direct answer.
- mcq_vqa: a multiple-choice question with exactly 4 options labeled A), B), C), D).
- report: a broad analysis request answered with a concise diagnostic report paragraph.

[COMPLEXITY ROUTING]
Each slot already has a fixed dimension. You must follow it exactly.

[REASONING FORMAT]
- Rapid Triage: reasoning must be exactly 4 numbered steps:
  1. This is a Rapid Triage question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Formally observe and report key abnormal imaging features.
  4. Conclusion.
  For MCQ, steps 3-4 must still reflect explicit option traversal before the final conclusion.
- Standard Assessment: reasoning must be exactly 5 numbered steps:
  1. This is a Standard Assessment question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Compare with normal anatomy, formally observe and report key abnormal imaging features.
  4. Determine the primary abnormality type.
  5. Conclusion.
  For MCQ, the option traversal must be integrated into steps 3-5 without breaking the numbered template.
- Comprehensive Diagnosis: reasoning must be exactly 7 numbered steps:
  1. This is a Comprehensive Diagnosis question about [question type].
  2. Identify and describe the primary anatomical region/structures.
  3. Formally observe and report key abnormal imaging features.
  4. Compare with normal anatomy to establish the pathological baseline.
  5. Determine the primary abnormality type.
  6. Evaluate the severity and clinical risk.
  7. Output the definitive final clinical judgment.
  For MCQ, explicit option traversal must be integrated into steps 3-7 without breaking the numbered template.

[QUESTION QUALITY]
1. Maximize coverage of informative metadata: largest lesion, most suspicious lesion, multiplicity, size hierarchy, and case-level significance when clinically relevant.
2. Keep questions diverse across slots. Avoid near-duplicates.
3. For MCQ, exactly one option must be correct.
4. For report, write a coherent integrated summary rather than a short answer fragment.

[OUTPUT FORMAT]
Return only JSON. No markdown. No explanations outside JSON.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize advanced 3D CoT VQA data from integrated case metadata.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-open-name", type=str, default=DEFAULT_OUTPUT_OPEN_NAME)
    parser.add_argument("--output-mcq-name", type=str, default=DEFAULT_OUTPUT_MCQ_NAME)
    parser.add_argument("--output-report-name", type=str, default=DEFAULT_OUTPUT_REPORT_NAME)
    parser.add_argument("--output-all-name", type=str, default=DEFAULT_OUTPUT_ALL_NAME)
    parser.add_argument("--gpu-devices", type=str, default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--test-case-limit", type=int, default=None)
    parser.add_argument("--max-open-per-case", type=int, default=2)
    parser.add_argument("--max-mcq-per-case", type=int, default=2)
    parser.add_argument("--max-report-per-case", type=int, default=1)
    parser.add_argument("--max-total-items-per-case", type=int, default=5)
    parser.add_argument("--image-name-template", type=str, default="{case_id}_full_volume.nii.gz")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    parser.add_argument("--gpu-idle-util-threshold", type=float, default=10.0)
    parser.add_argument("--gpu-idle-mem-mb-threshold", type=float, default=2000.0)
    parser.add_argument("--gpu-check-interval-sec", type=int, default=30)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    if args.gpu_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def resolve_tensor_parallel_size(args: argparse.Namespace) -> int:
    if args.tensor_parallel_size is not None:
        return args.tensor_parallel_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 1
    return max(1, len([x for x in visible.split(",") if x.strip()]))


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


def load_valid_cases(args: argparse.Namespace) -> List[Dict]:
    cases = []
    for case_dir in sorted(args.base_dir.iterdir()):
        if args.test_case_limit is not None and len(cases) >= args.test_case_limit:
            break
        if not case_dir.is_dir():
            continue
        json_path = case_dir / "integrated_meta.json"
        image_path = case_dir / args.image_name_template.format(case_id=case_dir.name)
        if not json_path.exists() or not image_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cases.append({"dir": case_dir, "image": str(image_path), "data": data})
    return cases


def has_informative_diagnosis(case_data: dict) -> bool:
    diag = case_data.get("clinical_diagnosis", {})
    for key, value in diag.items():
        key_l = str(key).lower()
        value_l = str(value).strip().lower()
        if "patient id" in key_l:
            continue
        if value_l in {"", "unknown", "none", "nan", "0", "0.0"}:
            continue
        return True
    return False


def build_case_insights(case_data: dict) -> Dict:
    lesions = case_data.get("nodules", []) or case_data.get("lesions", [])
    scan_info = case_data.get("scan_info", {})
    lesion_count = len(lesions)
    if not lesions:
        return {
            "patient_id": case_data.get("patient_id", "unknown"),
            "lesion_count": 0,
            "has_diag": has_informative_diagnosis(case_data),
            "scan_info": scan_info,
            "largest_diameter": None,
            "largest_volume": None,
            "highest_malignancy": None,
            "highest_spiculation": None,
            "smallest_diameter": None,
        }

    def feat(item: dict, name: str) -> float:
        return float(item.get("features", {}).get(name, -1))

    by_diameter_desc = sorted(lesions, key=lambda n: feat(n, "diameter_mm"), reverse=True)
    by_volume_desc = sorted(lesions, key=lambda n: feat(n, "volume_mm3"), reverse=True)
    by_malignancy_desc = sorted(lesions, key=lambda n: feat(n, "malignancy_score"), reverse=True)
    by_spiculation_desc = sorted(lesions, key=lambda n: feat(n, "spiculation_score"), reverse=True)
    by_diameter_asc = sorted(lesions, key=lambda n: feat(n, "diameter_mm"))

    return {
        "patient_id": case_data.get("patient_id", "unknown"),
        "lesion_count": lesion_count,
        "has_diag": has_informative_diagnosis(case_data),
        "scan_info": scan_info,
        "largest_diameter": by_diameter_desc[0],
        "second_largest_diameter": by_diameter_desc[1] if lesion_count > 1 else None,
        "smallest_diameter": by_diameter_asc[0],
        "largest_volume": by_volume_desc[0],
        "highest_malignancy": by_malignancy_desc[0],
        "highest_spiculation": by_spiculation_desc[0],
    }


def lesion_focus_text(label: str, lesion: Optional[dict]) -> str:
    if not lesion:
        return label
    features = lesion.get("features", {})
    return (
        f"{label}: {lesion.get('nodule_id', lesion.get('lesion_id', 'unknown'))} "
        f"(diameter={features.get('diameter_mm')} mm, volume={features.get('volume_mm3')} mm3, "
        f"malignancy={features.get('malignancy_score')}, spiculation={features.get('spiculation_score')})"
    )


def build_focus_lists(insights: Dict) -> Dict[str, List[str]]:
    lesion_count = insights["lesion_count"]
    mcq_focuses = [
        lesion_focus_text("Largest diameter lesion", insights.get("largest_diameter")),
        lesion_focus_text("Highest malignancy lesion", insights.get("highest_malignancy")),
        lesion_focus_text("Highest spiculation lesion", insights.get("highest_spiculation")),
        lesion_focus_text("Largest volume lesion", insights.get("largest_volume")),
    ]
    open_focuses = [
        lesion_focus_text("Largest diameter lesion", insights.get("largest_diameter")),
        lesion_focus_text("Highest malignancy lesion", insights.get("highest_malignancy")),
        "Overall lesion burden and multiplicity",
    ]
    report_focuses = [
        "Integrated whole-case report covering the dominant lesion and relevant secondary lesions",
        "Risk-focused integrated report emphasizing the most suspicious lesion and case-level significance",
    ]

    if lesion_count > 1:
        mcq_focuses.extend(
            [
                lesion_focus_text("Smallest diameter lesion", insights.get("smallest_diameter")),
                "Comparison between the largest and smallest lesions",
                f"Total lesion count = {lesion_count}",
            ]
        )
        open_focuses.extend(
            [
                "Comparison between the largest and smallest lesions",
                f"Multiplicity of lesions with total count {lesion_count}",
            ]
        )

    return {"mcq": mcq_focuses, "open": open_focuses, "report": report_focuses}


def balanced_dimensions(count: int) -> List[str]:
    base = ["Rapid Triage", "Standard Assessment", "Comprehensive Diagnosis"]
    return [base[i % len(base)] for i in range(count)]


def report_dimensions(count: int) -> List[str]:
    if count <= 1:
        return ["Comprehensive Diagnosis"] if count else []
    return ["Standard Assessment", "Comprehensive Diagnosis"][:count]


def build_slot_plan(case_data: dict, args: argparse.Namespace) -> List[Dict]:
    insights = build_case_insights(case_data)
    if insights["lesion_count"] == 0:
        return []

    open_count = min(args.max_open_per_case, 1 if insights["lesion_count"] == 1 else 2)
    mcq_count = min(args.max_mcq_per_case, 1 if insights["lesion_count"] == 1 else 2)
    report_count = 1 if args.max_report_per_case > 0 else 0
    focuses = build_focus_lists(insights)
    slots = []

    while open_count + mcq_count + report_count > args.max_total_items_per_case:
        if open_count >= mcq_count and open_count > 0:
            open_count -= 1
        elif mcq_count > 0:
            mcq_count -= 1
        elif report_count > 0:
            report_count -= 1
        else:
            break

    for idx, dim in enumerate(balanced_dimensions(open_count), start=1):
        slots.append(
            {
                "slot_id": f"open_{idx}",
                "type": "open_vqa",
                "dimension": dim,
                "focus": focuses["open"][(idx - 1) % len(focuses["open"])],
            }
        )
    for idx, dim in enumerate(balanced_dimensions(mcq_count), start=1):
        slots.append(
            {
                "slot_id": f"mcq_{idx}",
                "type": "mcq_vqa",
                "dimension": dim,
                "focus": focuses["mcq"][(idx - 1) % len(focuses["mcq"])],
            }
        )
    for idx, dim in enumerate(report_dimensions(report_count), start=1):
        slots.append(
            {
                "slot_id": f"report_{idx}",
                "type": "report",
                "dimension": dim,
                "focus": focuses["report"][(idx - 1) % len(focuses["report"])],
            }
        )
    return slots


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

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
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
                    try:
                        parsed = ast.literal_eval(candidate)
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except Exception:
                        return None
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def reasoning_lines(reasoning: str) -> List[str]:
    return [line.strip() for line in normalize_reasoning(reasoning).splitlines() if line.strip()]


def parse_mcq_labels(question: str) -> List[str]:
    return re.findall(r"\b([A-D])\)", str(question or ""))


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


def fallback_question_type(q: str, task_type: str, dimension: str) -> str:
    ql = (q or "").lower()
    if dimension == "Rapid Triage":
        if re.search(r"\b(mm|cm|diameter|size|measure|volume|largest|smallest)\b", ql):
            return "Single-Point Measurement (Size/Thickness/Numerical Inquiry)"
        if re.search(r"\bwhere|which\s+(lobe|segment|side)|left|right|upper|lower|location|located)\b", ql):
            return "Single-Point Localization (Which Lobe/Segment/Side)"
        if re.search(r"\b(is|are|does|do|whether|normal|abnormal|yes|no)\b", ql):
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


def repair_reasoning_first_line(reasoning: str, dimension: str, question_type: str) -> str:
    lines = reasoning_lines(reasoning)
    if not lines:
        return ""
    lines[0] = f"1. This is a {dimension} question about {question_type}."
    return "\n".join(lines)


def is_valid_reasoning(reasoning: str, dimension: str, question_type: str) -> bool:
    lines = reasoning_lines(reasoning)
    if len(lines) != EXPECTED_REASONING_LINES[dimension]:
        return False
    if lines[0] != f"1. This is a {dimension} question about {question_type}.":
        return False
    for idx, line in enumerate(lines, start=1):
        if not re.match(rf"^{idx}\.\s+", line):
            return False
    return True


def get_persona(case_id: str) -> str:
    idx = int(hashlib.md5(case_id.encode("utf-8")).hexdigest(), 16) % len(PERSONAS)
    return PERSONAS[idx]


def build_entry(case_id: str, image_path: str, item: dict) -> dict:
    question_type = normalize_question_type(item.get("question_type", "").strip(), item["dimension"])
    if not question_type:
        question_type = fallback_question_type(item.get("q", ""), item.get("type", "unknown"), item["dimension"])
    reasoning = repair_reasoning_first_line(item.get("reasoning", ""), item["dimension"], question_type)
    answer = item.get("a", "").strip()
    return {
        "id": f"{case_id}_{item.get('slot_id', item.get('type', 'item'))}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"{DIMENSION_TRIGGER[item['dimension']]} {item['q']}".strip()},
            {"from": "gpt", "value": f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"},
        ],
        "task_type": item["type"],
        "question_type": question_type,
        "dimension": item["dimension"],
    }


def build_slot_prompt(slot: Dict, persona: str) -> str:
    allowed_qtypes = "\n".join(f"- {x}" for x in QUESTION_TYPES_BY_DIMENSION[slot["dimension"]])
    mcq_extra = ""
    if slot["type"] == "mcq_vqa":
        mcq_extra = (
            "8. In reasoning, explicitly traverse options A), B), C), and D) before the conclusion.\n"
            "9. Do not state or imply that the answer is known first and justified afterward.\n"
        )
    return f"""You are an expert medical annotator creating one dataset item from structured case metadata.
{GLOBAL_RULES}
[PERSONA] {persona}

[FIXED SLOT]
- slot_id: {slot['slot_id']}
- type: {slot['type']}
- dimension: {slot['dimension']}
- focus: {slot['focus']}

[QUESTION TYPE POLICY FOR THIS SLOT]
Choose exactly one question_type from:
{allowed_qtypes}

[TASK CONSTRAINTS]
1. q must align with the fixed slot type, fixed dimension, and fixed focus.
2. q, reasoning, and a must be written as image-grounded content only.
3. Do not mention metadata, reports, JSON, prompts, labels, or synthesis instructions.
4. If type is mcq_vqa, q must include exactly four options labeled A), B), C), D).
5. a must directly answer q and stay consistent with the evidence.
6. reasoning must follow the exact numbered-step template of the fixed dimension.
7. reasoning step 1 must be exactly: 1. This is a {slot['dimension']} question about [question type].
{mcq_extra}10. Return only one pure JSON object. No markdown. No commentary.

[OUTPUT FORMAT]
{{
  "slot_id": "{slot['slot_id']}",
  "type": "{slot['type']}",
  "dimension": "{slot['dimension']}",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}}"""


def build_reflection_prompt(slot: Dict) -> str:
    rules = [
        "1. Output must be one JSON object only.",
        f"2. slot_id must be exactly {slot['slot_id']}.",
        f"3. type must be exactly {slot['type']}.",
        f"4. dimension must be exactly {slot['dimension']}.",
        "5. reasoning must not mention metadata, report text, JSON, prompt rules, or synthesis instructions.",
        "6. reasoning must not contain reviewer or template text such as line labels, output format notes, or question_type instructions.",
        "7. reasoning must use consecutive numbered steps starting from 1.",
        "8. reasoning step 1 must exactly follow the required template.",
    ]
    if slot["type"] == "mcq_vqa":
        rules.extend(
            [
                "9. q must contain exactly four explicit options A), B), C), and D).",
                "10. reasoning must explicitly traverse options A), B), C), and D) before the conclusion.",
                "11. do not backsolve from the final answer.",
            ]
        )
    return "Review your JSON against these rules:\n" + "\n".join(rules) + "\nIf violated, rewrite. Output only pure JSON."


def build_json_repair_prompt(slot: Dict, broken_output: str) -> str:
    mcq_extra = ""
    if slot["type"] == "mcq_vqa":
        mcq_extra = (
            "\n9. q must contain exactly four options labeled A), B), C), D)."
            "\n10. reasoning must explicitly traverse A), B), C), and D) before the conclusion."
        )
    return f"""Your previous response was not machine-parseable JSON.

Rewrite it as exactly one valid JSON object and preserve the intended medical content.

Requirements:
1. slot_id must be exactly {slot['slot_id']}.
2. type must be exactly {slot['type']}.
3. dimension must be exactly {slot['dimension']}.
4. reasoning must follow the exact numbered template for the dimension.
5. Do not mention metadata, JSON, reports, prompts, or synthesis instructions.
6. Return only pure JSON with double-quoted keys and strings.
7. No markdown fences. No explanatory text before or after the JSON.
8. The JSON schema is:
{{
  "slot_id": "{slot['slot_id']}",
  "type": "{slot['type']}",
  "dimension": "{slot['dimension']}",
  "question_type": "...",
  "q": "...",
  "reasoning": "...\\n...",
  "a": "..."
}}{mcq_extra}

Broken previous response:
{broken_output}
"""


def validate_slot_item(item: dict, slot: Dict) -> Optional[str]:
    if item.get("slot_id") != slot["slot_id"]:
        return f"slot_id_mismatch expected={slot['slot_id']} got={item.get('slot_id')}"
    if item.get("type") != slot["type"] or item.get("dimension") != slot["dimension"]:
        return (
            f"type_or_dimension_mismatch expected=({slot['type']},{slot['dimension']}) "
            f"got=({item.get('type')},{item.get('dimension')})"
        )
    question_type = normalize_question_type(item.get("question_type", "").strip(), slot["dimension"])
    if question_type not in QUESTION_TYPES_BY_DIMENSION[slot["dimension"]]:
        return f"invalid_question_type dimension={slot['dimension']} question_type={question_type}"
    item["question_type"] = question_type
    item["reasoning"] = repair_reasoning_first_line(item.get("reasoning", ""), slot["dimension"], question_type)
    if not is_valid_reasoning(item["reasoning"], slot["dimension"], question_type):
        return f"invalid_reasoning_lines dimension={slot['dimension']}"
    combined_text = "\n".join([str(item.get("q", "")), str(item.get("reasoning", "")), str(item.get("a", ""))])
    if contains_external_info(combined_text):
        return "external_info_leak"
    if contains_instruction_leak(combined_text):
        return "instruction_leak"
    if slot["type"] == "mcq_vqa":
        labels = parse_mcq_labels(item.get("q", ""))
        if labels != ["A", "B", "C", "D"]:
            return f"invalid_mcq_options labels={labels}"
        traversed = sorted(set(parse_traversed_labels(item["reasoning"])))
        if traversed != ["A", "B", "C", "D"]:
            return f"mcq_no_full_option_traversal traversed={traversed}"
    return None


def build_output_paths(args: argparse.Namespace, output_dir: Path) -> Dict[str, Path]:
    return {
        "open_vqa": output_dir / args.output_open_name,
        "mcq_vqa": output_dir / args.output_mcq_name,
        "report": output_dir / args.output_report_name,
        "all": output_dir / args.output_all_name,
    }


def initialize_output_files(args: argparse.Namespace, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(args, output_dir)
    for path in output_paths.values():
        path.write_text("", encoding="utf-8")
    return output_paths


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_entry(output_paths: Dict[str, Path], entry: dict) -> None:
    append_jsonl(output_paths[entry["task_type"]], entry)
    append_jsonl(output_paths["all"], entry)


def process_task_group(
    llm: LLM,
    sampling_params: SamplingParams,
    task_type: str,
    tasks: List[Dict],
    output_paths: Dict[str, Path],
    dataset_open: List[dict],
    dataset_mcq: List[dict],
    dataset_report: List[dict],
    failed_details: List[dict],
    args: argparse.Namespace,
) -> int:
    if not tasks:
        return 0
    failed = 0
    pbar = tqdm(total=len(tasks), desc=f"Generate {task_type}", unit="slot")
    for batch_start in range(0, len(tasks), args.batch_size):
        batch = tasks[batch_start : batch_start + args.batch_size]
        batch_messages = [task["messages"] for task in batch]
        pass1_outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)

        reflection_messages = []
        for task, output in zip(batch, pass1_outputs):
            history = list(task["messages"])
            history.append({"role": "assistant", "content": output.outputs[0].text})
            history.append({"role": "user", "content": build_reflection_prompt(task["slot"])})
            reflection_messages.append(history)

        pass2_outputs = llm.chat(messages=reflection_messages, sampling_params=sampling_params, use_tqdm=False)
        repaired_indices = []
        final_texts = [output.outputs[0].text for output in pass2_outputs]
        for idx, (task, raw_text) in enumerate(zip(batch, final_texts)):
            data = extract_json_block(raw_text)
            if not isinstance(data, dict):
                repaired_indices.append(idx)

        if repaired_indices:
            repair_messages = []
            for idx in repaired_indices:
                task = batch[idx]
                repair_messages.append(
                    [
                        {"role": "system", "content": build_slot_prompt(task["slot"], get_persona(task["case_id"]))},
                        {"role": "user", "content": f"Patient metadata JSON:\n{json.dumps(task['case_data'], ensure_ascii=False)}"},
                        {"role": "assistant", "content": final_texts[idx]},
                        {"role": "user", "content": build_json_repair_prompt(task["slot"], final_texts[idx])},
                    ]
                )
            repair_outputs = llm.chat(messages=repair_messages, sampling_params=sampling_params, use_tqdm=False)
            for idx, repair_output in zip(repaired_indices, repair_outputs):
                final_texts[idx] = repair_output.outputs[0].text

        for task, raw_text in zip(batch, final_texts):
            data = extract_json_block(raw_text)
            obj = data if isinstance(data, dict) else None
            if not obj:
                failed += 1
                failed_details.append(
                    {
                        "case_id": task["case_id"],
                        "slot_id": task["slot"]["slot_id"],
                        "reason": "json_object_parse_failed",
                        "preview": raw_text[:600].replace("\n", " "),
                    }
                )
                continue

            fail_reason = validate_slot_item(obj, task["slot"])
            if fail_reason:
                failed += 1
                failed_details.append(
                    {
                        "case_id": task["case_id"],
                        "slot_id": task["slot"]["slot_id"],
                        "reason": fail_reason,
                        "preview": output.outputs[0].text[:600].replace("\n", " "),
                    }
                )
                continue

            entry = build_entry(task["case_id"], task["image"], obj)
            if entry["task_type"] == "open_vqa":
                dataset_open.append(entry)
            elif entry["task_type"] == "mcq_vqa":
                dataset_mcq.append(entry)
            else:
                dataset_report.append(entry)
            append_entry(output_paths, entry)
        pbar.update(len(batch))
    pbar.close()
    return failed


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    cases = load_valid_cases(args)
    if not cases:
        print("No valid cases found.")
        return

    tensor_parallel_size = resolve_tensor_parallel_size(args)
    print(
        f"Runtime config -> base_dir={args.base_dir}, output_dir={args.output_dir}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}, "
        f"tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"test_case_limit={args.test_case_limit}"
    )
    wait_for_idle_gpus(args)

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    slot_tasks = {"open_vqa": [], "mcq_vqa": [], "report": []}
    dataset_open = []
    dataset_mcq = []
    dataset_report = []
    failed_details = []
    output_paths = initialize_output_files(args, args.output_dir)

    for case in tqdm(cases, desc="Build slot plans", unit="case"):
        case_id = case["data"].get("patient_id", case["dir"].name)
        slots = build_slot_plan(case["data"], args)
        if not slots:
            continue
        persona = get_persona(case_id)
        for slot in slots:
            slot_tasks[slot["type"]].append(
                {
                    "case_id": case_id,
                    "image": case["image"],
                    "slot": slot,
                    "case_data": case["data"],
                    "messages": [
                        {"role": "system", "content": build_slot_prompt(slot, persona)},
                        {"role": "user", "content": f"Patient metadata JSON:\n{json.dumps(case['data'], ensure_ascii=False)}"},
                    ],
                }
            )

    failed = 0
    failed += process_task_group(
        llm,
        sampling_params,
        "open_vqa",
        slot_tasks["open_vqa"],
        output_paths,
        dataset_open,
        dataset_mcq,
        dataset_report,
        failed_details,
        args,
    )
    failed += process_task_group(
        llm,
        sampling_params,
        "mcq_vqa",
        slot_tasks["mcq_vqa"],
        output_paths,
        dataset_open,
        dataset_mcq,
        dataset_report,
        failed_details,
        args,
    )
    failed += process_task_group(
        llm,
        sampling_params,
        "report",
        slot_tasks["report"],
        output_paths,
        dataset_open,
        dataset_mcq,
        dataset_report,
        failed_details,
        args,
    )

    print(f"Open VQA: {len(dataset_open)}")
    print(f"MCQ VQA: {len(dataset_mcq)}")
    print(f"Report: {len(dataset_report)}")
    print(f"Failed: {failed}")
    print(f"Output dir: {args.output_dir}")
    print(f"Open file: {args.output_open_name}")
    print(f"MCQ file: {args.output_mcq_name}")
    print(f"Report file: {args.output_report_name}")
    print(f"Mixed file: {args.output_all_name}")
    if failed_details:
        print("Failure details:")
        for item in failed_details[:10]:
            print(f"- {item['case_id']} / {item['slot_id']}: {item['reason']}")


if __name__ == "__main__":
    main()

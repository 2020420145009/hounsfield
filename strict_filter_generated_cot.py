#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

import build_m3dtest_like_subset_v2 as builder


ROOT = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d")

DATASETS = [
    {
        "name": "generated_m3d_cot_1000/m3d_mcq_cot.jsonl",
        "task_type": "mcq_vqa",
        "question_mode": "mcq",
    },
    {
        "name": "generated_m3d_cot_1000/m3d_open_cot.jsonl",
        "task_type": "open_vqa",
        "question_mode": "open",
    },
    {
        "name": "generated_ctrate_cot_1000/ctrate_mcq_cot.jsonl",
        "task_type": "mcq_vqa",
        "question_mode": "mcq",
    },
    {
        "name": "generated_ctrate_cot_1000/ctrate_open_cot.jsonl",
        "task_type": "open_vqa",
        "question_mode": "open",
    },
]

EXPECTED_REASONING_LINES = {
    "Rapid Triage": 4,
    "Standard Assessment": 5,
    "Comprehensive Diagnosis": 7,
}

TEMPLATE_LINE_PATTERNS = [
    r"^Identify and describe the primary anatomy involved\.?$",
    r"^Identify and describe the primary anatomical region/structures(?: relevant to the task)?\.?$",
    r"^Formally observe and report key abnormal imaging features\.?$",
    r"^Formally observe and report the key imaging evidence(?: from the JSON)?\.?$",
    r"^Compare with normal anatomy, formally observe and report key abnormal imaging features\.?$",
    r"^Compare with normal anatomy to establish the pathological baseline\.?$",
    r"^Determine the primary abnormality type(?: or structured descriptive summary)?\.?$",
    r"^Evaluate the severity and clinical risk\.?$",
    r"^Output the definitive final clinical judgment\.?$",
    r"^Conclusion\.?$",
]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_blocks(text: str):
    raw = str(text or "").strip()
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.S | re.I)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.S | re.I)
    think_lines = []
    if think_match:
        think_lines = [ln.strip() for ln in think_match.group(1).splitlines() if ln.strip()]
    answer_text = answer_match.group(1).strip() if answer_match else ""
    return raw, think_match is not None, answer_match is not None, think_lines, answer_text


def has_numbered_reasoning(raw: str) -> bool:
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.S | re.I)
    if not think_match:
        return False
    return bool(re.search(r"^\s*\d+[\.\)]\s+", think_match.group(1), flags=re.M))


def has_template_line(line: str) -> bool:
    for pat in TEMPLATE_LINE_PATTERNS:
        if re.match(pat, str(line or "").strip(), flags=re.I):
            return True
    return False


def validate_record(record: dict, task_type: str, question_mode: str):
    reasons = []
    question_raw = str(record.get("conversations", [{}])[0].get("value", ""))
    answer_raw = str(record.get("conversations", [{}, {}])[1].get("value", ""))
    question = (
        builder.format_mcq_question(question_raw)
        if question_mode == "mcq"
        else builder.normalize_question(question_raw)
    )
    normalized_answer = (
        builder.format_mcq_answer(question, answer_raw)
        if question_mode == "mcq"
        else builder.sanitize_cot_answer(answer_raw, question_text=question, force_mcq=False)
    )

    raw, has_think, has_answer, think_lines, final_answer = split_blocks(answer_raw)
    dimension = str(record.get("dimension", "")).strip()
    question_type = str(record.get("question_type", "")).strip()
    expected_lines = EXPECTED_REASONING_LINES.get(dimension)
    expected_first_line = f"This is a {dimension} question about {question_type}."

    if not has_think:
        reasons.append("missing_think_block")
    if not has_answer:
        reasons.append("missing_answer_block")
    if "```" in raw:
        reasons.append("markdown_fence")
    if builder.contains_report_leak(raw) or builder.contains_report_leak(final_answer):
        reasons.append("report_leak")
    if builder.contains_instruction_leak(raw):
        reasons.append("instruction_leak")
    if has_numbered_reasoning(raw):
        reasons.append("numbered_reasoning")

    if has_think:
        if not think_lines:
            reasons.append("empty_think_block")
        else:
            if think_lines[0] != expected_first_line:
                reasons.append("first_line_mismatch")
            if expected_lines is None:
                reasons.append("unknown_dimension")
            elif len(think_lines) != expected_lines:
                reasons.append("reasoning_line_count_mismatch")
            for line in think_lines[1:]:
                if not line.strip():
                    reasons.append("empty_reasoning_line")
                    break
            for line in think_lines[1:]:
                if has_template_line(line):
                    reasons.append("template_line_residue")
                    break

    if task_type == "mcq_vqa":
        choice_map = builder.extract_choice_map(question)
        if set(choice_map.keys()) != {"A", "B", "C", "D"}:
            reasons.append("mcq_missing_four_options")
        answer_letter = builder.extract_answer_letter(normalized_answer)
        if answer_letter is None:
            reasons.append("missing_answer_letter")
        traversal_line_idx = builder.find_option_traversal_line_index(think_lines)
        if set(choice_map.keys()) == {"A", "B", "C", "D"} and traversal_line_idx is None:
            reasons.append("mcq_no_option_traversal")
        chosen_text = choice_map.get(answer_letter) if answer_letter else None
        if chosen_text and traversal_line_idx is not None:
            for line in think_lines[1:traversal_line_idx]:
                if builder.mentions_choice_text(line, chosen_text):
                    reasons.append("answer_first_cheating")
                    break

    return sorted(set(reasons)), question, answer_raw


def filter_dataset(path: Path, task_type: str, question_mode: str):
    if not path.exists():
        return {
            "path": str(path),
            "status": "missing",
        }

    rows = read_jsonl(path)
    kept = []
    rejected = []
    reason_counter = Counter()

    for idx, row in enumerate(rows):
        reasons, normalized_question, normalized_answer = validate_record(row, task_type, question_mode)
        if reasons:
            reason_counter.update(reasons)
            rejected.append(
                {
                    "index": idx,
                    "id": row.get("id"),
                    "image": row.get("image"),
                    "task_type": row.get("task_type"),
                    "dimension": row.get("dimension"),
                    "question_type": row.get("question_type"),
                    "reasons": reasons,
                    "question": normalized_question,
                    "answer": normalized_answer,
                    "record": row,
                }
            )
        else:
            kept.append(row)

    kept_path = path.with_name(path.stem + ".strict_filtered.jsonl")
    rejected_path = path.with_name(path.stem + ".strict_rejected.jsonl")
    summary_path = path.with_name(path.stem + ".strict_summary.json")

    write_jsonl(kept_path, kept)
    write_jsonl(rejected_path, rejected)
    write_json(
        summary_path,
        {
            "source_file": str(path),
            "strict_filtered_file": str(kept_path),
            "strict_rejected_file": str(rejected_path),
            "total": len(rows),
            "kept": len(kept),
            "rejected": len(rejected),
            "reason_counts": dict(reason_counter),
        },
    )
    return {
        "path": str(path),
        "status": "ok",
        "total": len(rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "reason_counts": dict(reason_counter),
        "strict_filtered_file": str(kept_path),
        "strict_rejected_file": str(rejected_path),
        "strict_summary_file": str(summary_path),
    }


def main():
    all_summary = {}
    for dataset in DATASETS:
        path = ROOT / dataset["name"]
        all_summary[dataset["name"]] = filter_dataset(
            path=path,
            task_type=dataset["task_type"],
            question_mode=dataset["question_mode"],
        )

    out_path = ROOT / "strict_filter_generated_cot.summary.json"
    write_json(out_path, all_summary)
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


TAG_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
TAG_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
ALLOWED_ANSWERS = {"yes", "no"}


def validate_sample(obj):
    reasons = []

    if not isinstance(obj, dict):
        return ["not_object"]

    for key in ("id", "image", "conversations"):
        if key not in obj:
            reasons.append(f"missing_{key}")

    conv = obj.get("conversations")
    if not isinstance(conv, list):
        reasons.append("conversations_not_list")
        return reasons
    if len(conv) != 2:
        reasons.append("conversations_len_not_2")
        return reasons

    first, second = conv
    if not isinstance(first, dict) or not isinstance(second, dict):
        reasons.append("conversation_turn_not_object")
        return reasons

    if first.get("from") != "human":
        reasons.append("first_turn_not_human")
    if second.get("from") != "gpt":
        reasons.append("second_turn_not_gpt")

    question = first.get("value")
    answer_text = second.get("value")
    if not isinstance(question, str) or not question.strip():
        reasons.append("empty_question")
    if not isinstance(answer_text, str) or not answer_text.strip():
        reasons.append("empty_answer_text")
        return reasons

    if "ChatCompletion(" in answer_text:
        reasons.append("chatcompletion_artifact")

    think_match = TAG_THINK_RE.search(answer_text)
    answer_match = TAG_ANSWER_RE.search(answer_text)
    if not think_match:
        reasons.append("missing_think_tag")
    if not answer_match:
        reasons.append("missing_answer_tag")

    if think_match:
        think_body = think_match.group(1).strip()
        if not think_body:
            reasons.append("empty_think_body")
    if answer_match:
        answer_body = answer_match.group(1).strip()
        if not answer_body:
            reasons.append("empty_answer_body")
        elif answer_body.lower() not in ALLOWED_ANSWERS:
            reasons.append("answer_not_yes_no")

    return reasons


def main():
    parser = argparse.ArgumentParser(description="Filter task4 JSONL for finetune-safe samples.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Filtered output JSONL path")
    parser.add_argument(
        "--rejected-output",
        default=None,
        help="Rejected samples with reasons as JSONL",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = Path(args.rejected_output) if args.rejected_output else output_path.with_suffix(".rejected.jsonl")
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    total = 0
    reason_counter = Counter()

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout, \
         rejected_path.open("w", encoding="utf-8") as frej:
        for lineno, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as exc:
                rejected += 1
                reason_counter["bad_json"] += 1
                frej.write(json.dumps({
                    "line": lineno,
                    "id": None,
                    "reasons": ["bad_json"],
                    "error": str(exc),
                    "raw": line,
                }, ensure_ascii=False) + "\n")
                continue

            reasons = validate_sample(obj)
            if reasons:
                rejected += 1
                reason_counter.update(reasons)
                frej.write(json.dumps({
                    "line": lineno,
                    "id": obj.get("id"),
                    "image": obj.get("image"),
                    "reasons": reasons,
                    "sample": obj,
                }, ensure_ascii=False) + "\n")
                continue

            kept += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "total_samples": total,
        "kept_samples": kept,
        "rejected_samples": rejected,
        "rejection_reasons": dict(reason_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

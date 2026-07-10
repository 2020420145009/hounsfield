#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams

DEFAULT_MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_INPUT_JSONL = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl"
DEFAULT_GAP_CSV = "/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/benchmarks/3D-RAD/runtime/task4_fleming_lora_cot_gap_ranking.csv"
DEFAULT_OUTPUT_DIR = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/task4_top8_split_by_gap"

TOP8 = [
    "Emphysema",
    "Cardiomegaly",
    "Medical material",
    "Arterial wall calcification",
    "Hiatal hernia",
    "Peribronchial thickening",
    "Coronary artery wall calcification",
    "Pulmonary fibrotic sequela",
]

CATEGORY_HINTS = {
    "Emphysema": ["emphysema", "emphysematous"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged cardiac silhouette"],
    "Medical material": [
        "medical material",
        "catheter",
        "tube",
        "stent",
        "pacemaker",
        "prosthesis",
        "central venous catheter",
        "drainage catheter",
        "chest tube",
        "endotracheal tube",
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

SYSTEM_PROMPT = """You classify chest CT training samples by whether their medical content is most related to one of a fixed set of task4 benchmark categories.
Return one JSON object only.
Choose exactly one `matched_category` from the provided list, or `NONE` if the sample is not mainly about one of them.
Prefer semantic medical matching, not shallow string overlap.
If multiple categories appear, choose the most central one for the question and answer pair.
"""

USER_TEMPLATE = """Top-8 benchmark categories:
{categories}

Question:
{question}

Answer:
{answer}

Return exactly one JSON object on a single line with this schema:
{{"matched_category": "<one category or NONE>", "confidence": "high|medium|low", "reason": "short explanation"}}
Do not output any text before or after the JSON."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split training samples by task4 top-8 categories and whether Fleming LoRA CoT improved over Fleming base."
    )
    parser.add_argument("--input-jsonl", type=Path, default=Path(DEFAULT_INPUT_JSONL))
    parser.add_argument("--gap-csv", type=Path, default=Path(DEFAULT_GAP_CSV))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-devices", type=str, default="0,1,2,3")
    parser.add_argument("--mode", choices=["hybrid", "llm_only", "keyword_only"], default="hybrid")
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def load_gap_groups(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    top8_rows = [row for row in rows if row.get("pathology") in TOP8]
    order_map = {row["pathology"]: int(row["rank"]) for row in top8_rows}
    top8_rows.sort(key=lambda row: order_map[row["pathology"]])

    no_improve = []
    improved = []
    for row in top8_rows:
        gap = float(row["gap_vs_fleming_base"])
        if gap >= 0:
            no_improve.append(row["pathology"])
        else:
            improved.append(row["pathology"])
    return no_improve, improved


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect_hits(text: str) -> Dict[str, List[str]]:
    lowered = text.lower()
    results: Dict[str, List[str]] = {}
    for category, hints in CATEGORY_HINTS.items():
        hits = [hint for hint in hints if hint.lower() in lowered]
        if hits:
            results[category] = hits
    return results


def keyword_match(question: str, answer: str):
    question_hits = _collect_hits(question)
    answer_hits = _collect_hits(answer)
    combined_hits = _collect_hits(f"{question}\n{answer}")

    # First preference: the question itself explicitly targets one category.
    if len(question_hits) == 1:
        category = next(iter(question_hits))
        return category, question_hits[category], 'question_unique'

    # Medical material is noisy when only the answer background mentions devices.
    # Do not keyword-match it unless the question itself targets device/material content.
    combined_without_mm = {k: v for k, v in combined_hits.items() if k != 'Medical material'}
    if len(combined_without_mm) == 1:
        category = next(iter(combined_without_mm))
        return category, combined_without_mm[category], 'combined_unique_non_medical_material'

    # Allow answer-only keyword matching for precise categories, but never for Medical material.
    answer_without_mm = {k: v for k, v in answer_hits.items() if k != 'Medical material'}
    if len(answer_without_mm) == 1 and not question_hits:
        category = next(iter(answer_without_mm))
        return category, answer_without_mm[category], 'answer_unique_non_medical_material'

    return None, [], 'llm_needed'


def build_prompt(question: str, answer: str) -> str:
    categories = "\n".join(f"- {name}" for name in TOP8)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{USER_TEMPLATE.format(categories=categories, question=question.strip(), answer=answer.strip())}\n"
        f"<|assistant|>\n"
    )


def parse_json_response(text: str) -> Dict[str, str]:
    text = str(text or "").strip()
    if not text:
        return {"matched_category": "NONE", "confidence": "low", "reason": "Empty model output."}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"\{.*?\}", text, flags=re.S)
    for candidate in reversed(matches):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    lowered = text.lower()
    for category in TOP8:
        if category.lower() in lowered:
            return {"matched_category": category, "confidence": "low", "reason": text[:200]}
    return {"matched_category": "NONE", "confidence": "low", "reason": text[:200]}


def normalize_category(value: str) -> str:
    value = str(value or "").strip()
    if value == "NONE":
        return "NONE"
    lowered = value.lower()
    for category in TOP8:
        if lowered == category.lower() or category.lower() in lowered:
            return category
    return "NONE"


def init_llm(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    no_improve_categories, improved_categories = load_gap_groups(args.gap_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(read_jsonl(args.input_jsonl))
    if args.test_limit:
        samples = samples[: args.test_limit]

    llm = None
    sampling_params = None
    if args.mode in {"hybrid", "llm_only"}:
        llm = init_llm(args)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n\n"],
        )

    all_results: List[dict] = []
    batched_prompts: List[str] = []
    batched_meta: List[dict] = []

    def flush_batch():
        nonlocal batched_prompts, batched_meta, all_results
        if not batched_prompts:
            return
        outputs = llm.generate(batched_prompts, sampling_params, use_tqdm=False)
        for meta, output in zip(batched_meta, outputs):
            text = output.outputs[0].text if output.outputs else ""
            parsed = parse_json_response(text)
            matched_category = normalize_category(parsed.get("matched_category", "NONE"))
            record = dict(meta)
            record["matched_category"] = matched_category
            record["classification_backend"] = "llm"
            record["classification_confidence"] = str(parsed.get("confidence", "low"))
            record["classification_reason"] = str(parsed.get("reason", ""))
            record["matched_keywords"] = []
            record["classification_raw"] = text
            all_results.append(record)
        batched_prompts = []
        batched_meta = []

    for obj in tqdm(samples, desc="classify training samples"):
        conv = obj.get("conversations", [])
        if len(conv) < 2:
            continue
        question = str(conv[0].get("value", ""))
        answer = str(conv[1].get("value", ""))
        merged_text = f"{question}\n{answer}"
        record = {
            "id": obj.get("id", ""),
            "image": obj.get("image", ""),
            "_dataset_source": obj.get("_dataset_source", ""),
            "task_type": obj.get("task_type", ""),
            "question_type": obj.get("question_type", ""),
            "dimension": obj.get("dimension", ""),
            "question": question,
            "answer": answer,
        }

        if args.mode in {"hybrid", "keyword_only"}:
            matched_category, matched_hints, match_strategy = keyword_match(question, answer)
            if matched_category is not None:
                record["matched_category"] = matched_category
                record["classification_backend"] = "keyword"
                record["classification_confidence"] = "high"
                record["classification_reason"] = f"Keyword match via {match_strategy}."
                record["matched_keywords"] = matched_hints
                record["classification_raw"] = ""
                all_results.append(record)
                continue
            if args.mode == "keyword_only":
                record["matched_category"] = "NONE"
                record["classification_backend"] = "keyword"
                record["classification_confidence"] = "low"
                record["classification_reason"] = f"No reliable keyword match ({match_strategy})."
                record["matched_keywords"] = []
                record["classification_raw"] = ""
                all_results.append(record)
                continue

        batched_prompts.append(build_prompt(question, answer))
        batched_meta.append(record)
        if len(batched_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()

    no_improve_rows: List[dict] = []
    improved_rows: List[dict] = []
    other_rows: List[dict] = []
    counter = Counter()
    for row in all_results:
        category = row["matched_category"]
        if category in no_improve_categories:
            row["group_label"] = "no_improve_or_worse"
            no_improve_rows.append(row)
            counter["no_improve_or_worse"] += 1
        elif category in improved_categories:
            row["group_label"] = "improved"
            improved_rows.append(row)
            counter["improved"] += 1
        else:
            row["group_label"] = "other_or_unmatched"
            other_rows.append(row)
            counter["other_or_unmatched"] += 1
        counter[f"category::{category}"] += 1

    write_jsonl(args.output_dir / "all_classified.jsonl", all_results)
    write_jsonl(args.output_dir / "no_improve_or_worse.jsonl", no_improve_rows)
    write_jsonl(args.output_dir / "improved.jsonl", improved_rows)
    write_jsonl(args.output_dir / "other_or_unmatched.jsonl", other_rows)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "gap_csv": str(args.gap_csv),
        "mode": args.mode,
        "model_dir": args.model_dir if args.mode != "keyword_only" else None,
        "top8_categories": TOP8,
        "no_improve_or_worse_categories": no_improve_categories,
        "improved_categories": improved_categories,
        "counts": dict(counter),
        "total_rows": len(all_results),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

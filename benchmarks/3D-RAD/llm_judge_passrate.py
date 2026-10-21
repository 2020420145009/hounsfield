#!/usr/bin/env python3
"""LLM-as-a-judge for 3D-RAD open tasks (task1/2/3) with LOCAL model, PASSRATE only.

Modeled on scripts/llm_judge_open_results.py. It loads a local judge model via
vLLM (default) or talks to an OpenAI-compatible endpoint, judges each prediction
in eval_open_vqa.csv, and finally reports ONLY the pass rate (per subtask, per
task macro, and overall). A per-subtask judged CSV is still written so the judge
does not need to re-run, but the headline output is pass rate.

Run on a GPU machine, e.g.:
  python scripts/llm_judge_passrate.py \
      --results-root runtime/m3d_3drad/results \
      --judge-model /volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B \
      --tensor-parallel-size 4
"""
import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as _exc:  # record the real reason so we don't fail with a vague message
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = _exc

TASK_SUBTASKS = {
    "task1": ["Anatomical_observation", "Pathological_observation"],
    "task2": ["Abnormality_feature", "Abnormality_position", "Abnormality_type", "Diagnosis"],
    "task3": ["Diameter", "Size", "Thickness"],
}

DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
DEFAULT_JUDGE_MODEL = os.environ.get("THREED_RAD_JUDGE_MODEL", "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B")

SYSTEM_PROMPT = """You are a strict medical imaging VQA judge.

Evaluate whether the candidate answer is clinically correct for the question, using the reference answer as gold.

Scoring rubric:
- 5: clinically equivalent to the reference; correct and complete enough for this benchmark
- 4: correct with only minor wording or granularity differences
- 3: partially correct or incomplete
- 2: mostly incorrect
- 1: clearly wrong
- 0: empty, irrelevant, or nonsensical

Set "pass" to 1 only when the candidate should count as correct for short-answer medical VQA evaluation. Allow standard synonyms and abbreviations. Return JSON only."""

USER_PROMPT_TEMPLATE = """Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}

Judge whether the candidate answer is clinically correct relative to the reference answer.
Return exactly one JSON object on a single line with this schema:
{{"pass": 0 or 1, "score": integer 0-5, "reason": "short explanation"}}
Do not output analysis, chain-of-thought, markdown, or any text before or after the JSON."""

TASK3_SYSTEM_PROMPT = """You are a strict medical imaging measurement judge.

The question asks for a measurement (a diameter, a size, or a thickness). You are given a reference (gold) measurement and a candidate measurement. Decide whether the candidate is numerically close enough to the reference to count as correct.

Follow these rules exactly:
1. UNIT CONVERSION: Convert every value to millimetres before comparing. 1 cm = 10 mm. Treat "1.1 cm" and "11 mm" as identical. A pure unit difference is NEVER an error.
2. PICK THE COMPARISON VALUE:
   - If a value is a range (e.g. "2-3 mm" or "2-3 cm"), use its midpoint.
   - If the reference is multi-dimensional (e.g. "30x13 mm", "16x10 mm"), use its LARGEST dimension as the gold value (the question asks for one length). If the candidate is also multi-dimensional, use its largest dimension too.
   - If the candidate reports several values, choose the one that best matches the reference's largest dimension.
3. TOLERANCE (this defines correctness): let R be the reference value in mm and C the candidate value in mm. The candidate is CORRECT if |C - R| <= max(0.25 * R, 3 mm). That is, within 25% of the reference, or within 3 mm, whichever is larger.
4. Apply the tolerance consistently to every case. Do not invent stricter or looser thresholds.

Scoring rubric (after converting to mm):
- 5: essentially exact, within ~10% of the reference
- 4: within tolerance (|C - R| <= max(0.25*R, 3mm)) but not near-exact
- 3: outside tolerance but same order of magnitude (off by < 2x)
- 2: off by roughly 2x to 10x
- 1: off by an order of magnitude or more, or clearly the wrong structure
- 0: empty, no number given, irrelevant, or nonsensical

Set "pass" to 1 if and only if the candidate is within tolerance (rule 3), i.e. score >= 4. Otherwise set "pass" to 0. Return JSON only."""

TASK3_USER_PROMPT_TEMPLATE = """Question:
{question}

Reference measurement:
{reference}

Candidate measurement:
{candidate}

Convert both to millimetres, pick the comparison value per the rules, then apply the tolerance |C - R| <= max(0.25*R, 3mm).
Return exactly one JSON object on a single line with this schema:
{{"pass": 0 or 1, "score": integer 0-5, "reason": "short explanation; state R, C in mm and the tolerance"}}
Do not output analysis, chain-of-thought, markdown, or any text before or after the JSON."""


def get_task_prompts(task):
    if task == "task3":
        return TASK3_SYSTEM_PROMPT, TASK3_USER_PROMPT_TEMPLATE
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-judge open tasks (task1/2/3) with a local model, report PASSRATE only.")
    parser.add_argument("--results-root", type=Path, action="append", required=True, dest="results_roots",
                        help="Model results root containing task1/task2/task3 (e.g. runtime/m3d_3drad/results). Repeat for several roots.")
    parser.add_argument("--tasks", default="task1,task2,task3", help="Comma-separated tasks. Default: task1,task2,task3")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help=f"Judge model path or served name. Default: {DEFAULT_JUDGE_MODEL}")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"Judge API base URL (openai backend). Default: {DEFAULT_API_BASE}")
    parser.add_argument("--judge-backend", choices=["auto", "openai", "vllm"], default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=512, help="vLLM max concurrent sequences (effective batch size).")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional cap on vLLM context length.")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", default=False,
                        help="Disable vLLM custom all-reduce kernel (fixes cuda invalid argument on some setups).")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Judge API key (openai backend). Default: EMPTY")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per subtask for quick testing.")
    parser.add_argument("--concurrency", type=int, default=1, help="OpenAI backend only: judge requests in flight at once.")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between judge calls (openai backend).")
    parser.add_argument("--overwrite", action="store_true", help="Rejudge even if the output file already exists.")
    parser.add_argument("--output-suffix", default="", help="Suffix appended to judge filename before .csv.")
    parser.add_argument("--passrate-output", type=Path, default=None, help="Optional CSV path to write the pass-rate summary.")
    return parser.parse_args()


def parse_tasks(task_string: str):
    tasks = []
    for item in task_string.split(","):
        task = item.strip()
        if not task:
            continue
        if task not in TASK_SUBTASKS:
            raise ValueError(f"Unsupported task: {task}")
        tasks.append(task)
    if not tasks:
        raise ValueError("No valid tasks requested.")
    return tasks


def _dedup_sentences(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if not text:
        return text
    chunks = re.split(r"(?<=[.!?;\n])\s+", text)
    seen = set()
    kept = []
    consecutive_dups = 0
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        norm = re.sub(r"\s+", " ", c.lower())
        norm = re.sub(r"\b(\w+)( \1\b)+", r"\1", norm)
        if norm in seen:
            consecutive_dups += 1
            if consecutive_dups >= 2:
                break
            continue
        consecutive_dups = 0
        seen.add(norm)
        kept.append(c)
    out = " ".join(kept).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].strip()
    return out


def clean_candidate(pred: str) -> str:
    text = str(pred or "").strip()
    if not text:
        return ""
    lower = text.lower()
    idx = lower.rfind("<answer>")
    if idx != -1:
        text = text[idx + len("<answer>"):]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("</answer>", "").replace("<answer>", "")
    text = text.strip()
    return _dedup_sentences(text)


def parse_json_response(content: str):
    text = str(content or "").strip()
    if not text:
        return {"pass": 0, "score": 0, "reason": "Empty judge output."}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = text.replace("<think>", "").strip()
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
    lower = text.lower()
    score_match = re.search(r"score[^0-9]{0,20}([0-5])", lower)
    pass_match = re.search(r"pass[^0-9]{0,20}([01])", lower)
    if score_match or pass_match:
        return {
            "pass": int(pass_match.group(1)) if pass_match else 0,
            "score": int(score_match.group(1)) if score_match else 0,
            "reason": text[:200].replace("\n", " ").strip(),
        }
    raise ValueError(f"Could not parse judge JSON from: {text[:500]}")


def normalize_judge_fields(data: dict):
    judge_pass = data.get("pass", 0)
    judge_score = data.get("score", 0)
    reason = str(data.get("reason", "")).strip()
    try:
        judge_pass = int(judge_pass)
    except (TypeError, ValueError):
        judge_pass = 0
    judge_pass = 1 if judge_pass else 0
    try:
        judge_score = int(judge_score)
    except (TypeError, ValueError):
        judge_score = 0
    judge_score = max(0, min(5, judge_score))
    return judge_pass, judge_score, reason


def build_prompt(question: str, reference: str, candidate: str, task: str = "task1") -> str:
    system_prompt, user_template = get_task_prompts(task)
    return (
        f"{system_prompt}\n\n"
        f"{user_template.format(question=question.strip(), reference=reference.strip(), candidate=candidate.strip())}\n\n"
        f"JSON:"
    )


def judge_one_openai(client, model, question, reference, candidate, timeout, task="task1"):
    system_prompt, user_template = get_task_prompts(task)
    request_kwargs = dict(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(
                question=question.strip(), reference=reference.strip(), candidate=candidate.strip())},
        ],
        timeout=timeout,
    )
    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **request_kwargs)
    except Exception:
        response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or ""
    data = parse_json_response(content)
    judge_pass, judge_score, reason = normalize_judge_fields(data)
    return {"judge_pass": str(judge_pass), "judge_score": str(judge_score), "judge_reason": reason, "judge_raw": content.strip()}


def judge_batch_vllm(llm, sampling_params, rows, task="task1"):
    judged = [None] * len(rows)
    prompts = []
    positions = []
    for i, row in enumerate(rows):
        pred = clean_candidate(row.get("Pred", ""))
        if pred:
            prompts.append(build_prompt(
                question=str(row.get("Question", "")),
                reference=str(row.get("Answer", "")),
                candidate=pred,
                task=task,
            ))
            positions.append(i)
        else:
            judged[i] = {"judge_pass": "0", "judge_score": "0", "judge_reason": "Empty prediction.", "judge_raw": ""}
    if prompts:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for pos, output in zip(positions, outputs):
            content = output.outputs[0].text if output.outputs else ""
            try:
                data = parse_json_response(content)
                judge_pass, judge_score, reason = normalize_judge_fields(data)
                judged[pos] = {"judge_pass": str(judge_pass), "judge_score": str(judge_score),
                               "judge_reason": reason, "judge_raw": content.strip()}
            except Exception as exc:
                judged[pos] = {"judge_pass": "0", "judge_score": "0",
                               "judge_reason": f"Judge error: {type(exc).__name__}: {exc}"[:200],
                               "judge_raw": content.strip()}
    return judged


def run_subtask(judge_runtime, args, results_root, task, subtask):
    """Judge one subtask. Returns (pass_sum, judged, total) or None if missing input."""
    input_csv = results_root / task / subtask / "eval_open_vqa.csv"
    suffix = args.output_suffix if args.output_suffix else ""
    output_csv = results_root / task / subtask / f"eval_open_vqa_judge{suffix}.csv"
    if not input_csv.exists() or input_csv.stat().st_size == 0:
        print(f"Missing input, skip: {input_csv}")
        return None

    rows = list(csv.DictReader(open(input_csv, newline="")))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        return None

    fieldnames = list(rows[0].keys()) + ["judge_pass", "judge_score", "judge_reason", "judge_raw"]

    # Reuse a completed judge file unless overwrite is set.
    if output_csv.exists() and not args.overwrite:
        judged_rows = list(csv.DictReader(open(output_csv, newline="")))
        if len(judged_rows) >= len(rows):
            print(f"Reuse completed judge file: {output_csv} ({len(judged_rows)}/{len(rows)})")
            return _passrate_from_rows(judged_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.judge_backend != "openai":
        print(f"[{task}/{subtask}] batching {len(rows)} prompts to vLLM ...")
        judged_list = judge_batch_vllm(judge_runtime["llm"], judge_runtime["sampling_params"], rows, task=task)
        out_rows = []
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, judged in zip(rows, judged_list):
                out_row = dict(row)
                out_row.update(judged)
                writer.writerow(out_row)
                out_rows.append(out_row)
        print(f"[{task}/{subtask}] done {len(rows)}/{len(rows)}")
        return _passrate_from_rows(out_rows)

    # OpenAI backend.
    def judge_row(row):
        pred = clean_candidate(row.get("Pred", ""))
        if not pred:
            return {"judge_pass": "0", "judge_score": "0", "judge_reason": "Empty prediction.", "judge_raw": ""}
        try:
            return judge_one_openai(judge_runtime, args.judge_model, str(row.get("Question", "")),
                                    str(row.get("Answer", "")), pred, args.request_timeout, task=task)
        except Exception as exc:
            return {"judge_pass": "0", "judge_score": "0",
                    "judge_reason": f"Judge error: {type(exc).__name__}: {exc}"[:200], "judge_raw": ""}

    concurrency = max(1, int(getattr(args, "concurrency", 1)))
    out_rows = []
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if concurrency == 1:
            for idx, row in enumerate(rows):
                judged = judge_row(row)
                out_row = dict(row)
                out_row.update(judged)
                writer.writerow(out_row)
                f.flush()
                out_rows.append(out_row)
                print(f"[{task}/{subtask}] judged {idx + 1}/{len(rows)}")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for chunk_start in range(0, len(rows), concurrency):
                    chunk = rows[chunk_start:chunk_start + concurrency]
                    for row, judged in zip(chunk, pool.map(judge_row, chunk)):
                        out_row = dict(row)
                        out_row.update(judged)
                        writer.writerow(out_row)
                        out_rows.append(out_row)
                    f.flush()
                    print(f"[{task}/{subtask}] judged {min(chunk_start + len(chunk), len(rows))}/{len(rows)}")
    return _passrate_from_rows(out_rows)


def _passrate_from_rows(rows):
    judged = 0
    pass_sum = 0
    for row in rows:
        try:
            judge_pass = int(str(row.get("judge_pass", "")).strip())
        except ValueError:
            continue
        judged += 1
        pass_sum += judge_pass
    return pass_sum, judged, len(rows)


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)

    if args.judge_backend == "auto":
        args.judge_backend = "vllm" if Path(args.judge_model).exists() else "openai"

    if args.judge_backend == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package is not available in the current environment.")
        judge_runtime = OpenAI(api_key=args.api_key, base_url=args.api_base)
    else:
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM is not available in the current environment. "
                "Activate the env first (e.g. `source /volume/med-train/users/shuyan/flemingvlcode/fzy/env.sh`) "
                "and run with that env's python. "
                f"Original import error: {type(_VLLM_IMPORT_ERROR).__name__}: {_VLLM_IMPORT_ERROR}"
            )
        judge_runtime = {
            "llm": LLM(
                model=args.judge_model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype="bfloat16",
                trust_remote_code=True,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
            ),
            "sampling_params": SamplingParams(temperature=0.0, max_tokens=2048),
        }

    for results_root in args.results_roots:
        if not results_root.exists():
            print(f"Missing results root, skip: {results_root}")
            continue
        print(f"=== Judging results root: {results_root} ===")
        summary = [["scope", "pass_rate", "judged", "total"]]
        overall_pass = 0
        overall_judged = 0
        for task in tasks:
            task_pass = 0
            task_judged = 0
            for subtask in TASK_SUBTASKS[task]:
                result = run_subtask(judge_runtime, args, results_root, task, subtask)
                if result is None:
                    summary.append([f"{task}/{subtask}", "missing", "0", "0"])
                    continue
                pass_sum, judged, total = result
                task_pass += pass_sum
                task_judged += judged
                rate = pass_sum / judged if judged else 0.0
                summary.append([f"{task}/{subtask}", f"{rate:.6f}", str(judged), str(total)])
            task_rate = task_pass / task_judged if task_judged else 0.0
            summary.append([f"{task.upper()}_PASSRATE", f"{task_rate:.6f}", str(task_judged), str(task_judged)])
            overall_pass += task_pass
            overall_judged += task_judged

        overall_rate = overall_pass / overall_judged if overall_judged else 0.0
        summary.append(["OVERALL_PASSRATE", f"{overall_rate:.6f}", str(overall_judged), str(overall_judged)])

        print(f"\n===== PASSRATE SUMMARY: {results_root} =====")
        for row in summary:
            print("\t".join(row))

        if args.passrate_output is not None:
            args.passrate_output.parent.mkdir(parents=True, exist_ok=True)
            with args.passrate_output.open("w", newline="") as f:
                csv.writer(f).writerows(summary)
            print(f"\nWrote pass-rate summary: {args.passrate_output}")


if __name__ == "__main__":
    main()

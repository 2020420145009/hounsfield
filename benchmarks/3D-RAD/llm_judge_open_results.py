#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

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

# ---------------------------------------------------------------------------
# task3 is a NUMERIC MEASUREMENT benchmark (diameter / size / thickness). The
# generic clinical rubric above has no notion of numeric tolerance, so the judge
# was deciding "how close counts" ad hoc — producing inconsistent verdicts
# (e.g. 11mm passed vs an 18mm reference but failed vs 22mm) and penalising
# unit-only differences (1.1 cm vs 11 mm) or 2-D references (30x13 mm) it could
# not align. The prompts below pin down: convert everything to millimetres
# first, compare the largest/primary dimension, and apply an explicit tolerance.
# ---------------------------------------------------------------------------
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
    """Return (system_prompt, user_template) for the given task.

    task3 (numeric measurement) uses a tolerance-based rubric; everything else
    uses the generic clinical-equivalence rubric.
    """
    if task == "task3":
        return TASK3_SYSTEM_PROMPT, TASK3_USER_PROMPT_TEMPLATE
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge over task1/2/3 benchmark result CSVs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        required=True,
        dest="results_roots",
        help="Model results root containing task1/task2/task3. Repeat to judge several "
             "result roots (e.g. ablation fractions) with a single judge-model load.",
    )
    parser.add_argument(
        "--tasks",
        default="task1,task2,task3",
        help="Comma-separated tasks to judge. Default: task1,task2,task3",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help=f"Judge model name or served model name. Default: {DEFAULT_JUDGE_MODEL}")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"Judge API base URL. Default: {DEFAULT_API_BASE}")
    parser.add_argument('--judge-backend', choices=['auto', 'openai', 'vllm'], default='auto')
    parser.add_argument('--tensor-parallel-size', type=int, default=4)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-num-seqs', type=int, default=512,
                        help='vLLM max concurrent sequences (effective batch size). Larger keeps the GPU busier; raise/lower to fit memory.')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Optional cap on vLLM context length (prompt + generation).')
    parser.add_argument('--disable-custom-all-reduce', action='store_true', default=False,
                        help='Disable vLLM custom all-reduce kernel (fixes cuda invalid argument on some setups)')
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Judge API key. Default: EMPTY")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick testing.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="OpenAI backend only: number of judge requests in flight at once. >1 submits rows in "
                             "ordered chunks via a thread pool, preserving output row order and incremental writes.")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between judge calls.")
    parser.add_argument("--overwrite", action="store_true", help="Rejudge even if the output file already exists.")
    parser.add_argument("--output-suffix", default="", help="Suffix appended to output filename before .csv, e.g. '_deepseekr1' -> eval_open_vqa_judge_deepseekr1.csv")
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
    """Collapse runaway repetition then hard-cap length.

    Truncated reasoning from the candidate model frequently degenerates into the
    same sentence (or near-identical sentences) repeated dozens of times, e.g.
    "the pleural cavities are open. ... the pleural cavities are open. ...".
    Feeding that verbatim to the judge wastes context and biases scoring. We:
      1) split into sentence-ish chunks,
      2) drop a chunk if a normalized form was already seen (order preserved),
      3) also break out once N consecutive chunks are duplicates (catches loops
         that alternate between a couple of templates),
      4) hard-cap the final string at max_chars.
    """
    text = text.strip()
    if not text:
        return text
    # Split on sentence boundaries but keep it simple/robust.
    chunks = re.split(r"(?<=[.!?;\n])\s+", text)
    seen = set()
    kept = []
    consecutive_dups = 0
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        # Normalize for dedup: lowercase, collapse whitespace, strip word-level
        # immediate stutter ("are are" -> "are", "but but" -> "but").
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
    """Extract the model's actual answer from a raw prediction.

    Handles the response shapes seen in 3D-RAD task1/2/3 outputs:
      - plain short answer  -> returned as-is (after dedup/cap)
      - "<answer> ..."       -> text after <answer> (closing tag often missing,
                                but the answer is complete and should be kept)
      - "<think> ... <answer> ..." -> only the post-<answer> text
      - "<think> ..." with no answer (truncated reasoning, frequently repetitive)
        -> strip the tag and salvage the de-duplicated reasoning text
    Stray/unbalanced <think> / </think> / </answer> tags are removed throughout.
    """
    text = str(pred or "").strip()
    if not text:
        return ""

    # If an <answer> marker exists, the answer is everything after the LAST one,
    # even when </answer> is missing (complete answers must still be extracted).
    lower = text.lower()
    idx = lower.rfind("<answer>")
    if idx != -1:
        text = text[idx + len("<answer>"):]

    # Remove well-formed <think>...</think> blocks, then any stray tags.
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
    # Strip <think>...</think> reasoning blocks emitted by DeepSeek-R1.
    # R1 can emit malformed/unbalanced tags (e.g. a leading "</think>" with no
    # opening tag, or "</think><think>" reversed). Handle all cases:
    #   1) well-formed paired blocks
    #   2) any leftover stray <think> / </think> tags
    # After stripping the tags we still try to find JSON in whatever remains.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # If a closing tag remains (reasoning that never opened, or reversed tags),
    # drop everything up to and including the last </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    # Remove any remaining stray opening tags.
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
    score_match = re.search(r"score[^0-9]{0,20}([0-5])", lower)
    pass_match = re.search(r"pass[^0-9]{0,20}([01])", lower)
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


def judge_one_openai(client: OpenAI, model: str, question: str, reference: str, candidate: str, timeout: float, task: str = "task1"):
    system_prompt, user_template = get_task_prompts(task)
    request_kwargs = dict(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_template.format(
                    question=question.strip(),
                    reference=reference.strip(),
                    candidate=candidate.strip(),
                ),
            },
        ],
        timeout=timeout,
    )
    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or ""
    data = parse_json_response(content)
    judge_pass, judge_score, reason = normalize_judge_fields(data)
    return {
        "judge_pass": str(judge_pass),
        "judge_score": str(judge_score),
        "judge_reason": reason,
        "judge_raw": content.strip(),
    }


def judge_one_vllm(llm, sampling_params, question: str, reference: str, candidate: str):
    prompt = build_prompt(question, reference, candidate)
    output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    content = output.outputs[0].text if output.outputs else ""
    data = parse_json_response(content)
    judge_pass, judge_score, reason = normalize_judge_fields(data)
    return {
        "judge_pass": str(judge_pass),
        "judge_score": str(judge_score),
        "judge_reason": reason,
        "judge_raw": content.strip(),
    }


def judge_batch_vllm(llm, sampling_params, rows, task: str = "task1"):
    """Judge a list of rows in a SINGLE batched llm.generate() call.

    Submitting one prompt at a time (the old judge_one_vllm path) leaves the GPU
    almost idle: vLLM's throughput comes from continuous batching, which only
    kicks in when many requests are in flight at once. Here we build prompts for
    every non-empty-prediction row and hand them to vLLM together, so it packs
    them into running batches and keeps the 70B busy.

    Returns a list of judged dicts aligned 1:1 with `rows`. Rows with an empty
    prediction get a default failing score without hitting the model. A single
    un-parseable output is recorded as a default score (does not affect the rest
    of the batch).
    """
    judged = [None] * len(rows)
    prompts = []
    positions = []
    for i, row in enumerate(rows):
        pred = clean_candidate(row.get("Pred", ""))
        if pred:
            prompts.append(
                build_prompt(
                    question=str(row.get("Question", "")),
                    reference=str(row.get("Answer", "")),
                    candidate=pred,
                    task=task,
                )
            )
            positions.append(i)
        else:
            judged[i] = {
                "judge_pass": "0",
                "judge_score": "0",
                "judge_reason": "Empty prediction.",
                "judge_raw": "",
            }

    if prompts:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for pos, output in zip(positions, outputs):
            content = output.outputs[0].text if output.outputs else ""
            try:
                data = parse_json_response(content)
                judge_pass, judge_score, reason = normalize_judge_fields(data)
                judged[pos] = {
                    "judge_pass": str(judge_pass),
                    "judge_score": str(judge_score),
                    "judge_reason": reason,
                    "judge_raw": content.strip(),
                }
            except Exception as exc:
                judged[pos] = {
                    "judge_pass": "0",
                    "judge_score": "0",
                    "judge_reason": f"Judge error: {type(exc).__name__}: {exc}"[:200],
                    "judge_raw": content.strip(),
                }

    return judged


def run_subtask(judge_runtime, args: argparse.Namespace, results_root: Path, task: str, subtask: str):
    input_csv = results_root / task / subtask / "eval_open_vqa.csv"
    suffix = args.output_suffix if args.output_suffix else ""
    output_csv = results_root / task / subtask / f"eval_open_vqa_judge{suffix}.csv"
    if not input_csv.exists() or input_csv.stat().st_size == 0:
        print(f"Missing input, skip: {input_csv}")
        return

    rows = list(csv.DictReader(open(input_csv, newline="")))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    start_idx = 0
    fieldnames = list(rows[0].keys()) + ["judge_pass", "judge_score", "judge_reason", "judge_raw"]
    if output_csv.exists() and not args.overwrite:
        judged_rows = list(csv.DictReader(open(output_csv, newline="")))
        if len(judged_rows) >= len(rows):
            print(f"Skip completed judge file: {output_csv} ({len(judged_rows)}/{len(rows)})")
            return
        start_idx = len(judged_rows)
        print(f"Resume judge file: {output_csv} ({start_idx}/{len(rows)})")
    else:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    pending_rows = rows[start_idx:]

    if args.judge_backend != "openai":
        # vLLM path: judge the whole remaining slice in ONE batched call so vLLM
        # can continuous-batch across all prompts and keep the GPU saturated.
        # Results are written only after the batch completes. (Resume still works
        # at subtask granularity via start_idx; mid-subtask resume is sacrificed
        # for the large throughput win.)
        print(f"[{task}/{subtask}] batching {len(pending_rows)} prompts to vLLM ...")
        judged_list = judge_batch_vllm(
            llm=judge_runtime["llm"],
            sampling_params=judge_runtime["sampling_params"],
            rows=pending_rows,
            task=task,
        )
        with output_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row, judged in zip(pending_rows, judged_list):
                out_row = dict(row)
                out_row.update(judged)
                writer.writerow(out_row)
            f.flush()
        print(f"[{task}/{subtask}] done {len(rows)}/{len(rows)}")
        return

    # OpenAI path: per-request network calls. With --concurrency 1 we stream one
    # row at a time (original behaviour). With --concurrency N we submit rows in
    # ordered chunks of N through a thread pool, then write each chunk in input
    # order — so output row order and incremental/resume semantics are preserved
    # while N requests stay in flight against the online API.
    def judge_row(row):
        pred = clean_candidate(row.get("Pred", ""))
        if not pred:
            return {
                "judge_pass": "0",
                "judge_score": "0",
                "judge_reason": "Empty prediction.",
                "judge_raw": "",
            }
        try:
            return judge_one_openai(
                client=judge_runtime,
                model=args.judge_model,
                question=str(row.get("Question", "")),
                reference=str(row.get("Answer", "")),
                candidate=pred,
                timeout=args.request_timeout,
                task=task,
            )
        except Exception as exc:
            return {
                "judge_pass": "0",
                "judge_score": "0",
                "judge_reason": f"Judge error: {type(exc).__name__}: {exc}"[:200],
                "judge_raw": "",
            }

    concurrency = max(1, int(getattr(args, "concurrency", 1)))
    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if concurrency == 1:
            for idx in range(start_idx, len(rows)):
                row = rows[idx]
                judged = judge_row(row)
                out_row = dict(row)
                out_row.update(judged)
                writer.writerow(out_row)
                f.flush()
                print(f"[{task}/{subtask}] judged {idx + 1}/{len(rows)}")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for chunk_start in range(start_idx, len(rows), concurrency):
                    chunk = rows[chunk_start:chunk_start + concurrency]
                    for row, judged in zip(chunk, pool.map(judge_row, chunk)):
                        out_row = dict(row)
                        out_row.update(judged)
                        writer.writerow(out_row)
                    f.flush()
                    print(f"[{task}/{subtask}] judged {min(chunk_start + len(chunk), len(rows))}/{len(rows)}")


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)

    if args.judge_backend == "auto":
        args.judge_backend = "vllm" if Path(args.judge_model).exists() else "openai"

    if args.judge_backend == "openai":
        judge_runtime = OpenAI(api_key=args.api_key, base_url=args.api_base)
    else:
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM is not available in the current environment.")
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
            # DeepSeek-R1 is a reasoning model: it emits a <think>...</think> block
            # BEFORE the JSON. max_tokens must be large enough to cover the reasoning
            # chain *plus* the final JSON, otherwise the output gets cut off before the
            # JSON is ever produced. Do NOT use stop=["\n\n"] — the reasoning trace is
            # full of blank lines and would be truncated at the first one.
            "sampling_params": SamplingParams(temperature=0.0, max_tokens=2048),
        }

    for results_root in args.results_roots:
        if not results_root.exists():
            print(f"Missing results root, skip: {results_root}")
            continue
        print(f"=== Judging results root: {results_root} ===")
        for task in tasks:
            for subtask in TASK_SUBTASKS[task]:
                run_subtask(judge_runtime, args, results_root, task, subtask)


if __name__ == "__main__":
    main()

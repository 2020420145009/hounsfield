from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm import tqdm

from .generator import GeneratorConfig, LocalGenerator
from .io_utils import append_jsonl, load_completed_ids, load_state, save_state
from .schema import SynthesisTask
from .task_loader import load_ctrate_mcq_tasks, load_ctrate_open_tasks, load_three_source_rows
from .validator import build_output_entry, extract_json_block, validate_generated_dict
from .mixer import build_mixed_dataset


def run_ctrate_mode(
    *,
    mode: str,
    model_dir: str,
    output_dir: Path,
    ctrate_root: Path,
    ctrate_input: Path,
    batch_size: int,
    tensor_parallel_size: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    distributed_executor_backend: str,
    resume: bool,
    test_limit: int | None,
    target_count: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'ctrate_open' if mode == 'open_vqa' else 'ctrate_mcq'
    out_path = output_dir / f'{prefix}_cot.jsonl'
    failed_path = output_dir / f'{prefix}.failed.jsonl'
    failed_raw_path = output_dir / f'{prefix}.failed_raw.jsonl'
    state_path = output_dir / f'{prefix}.resume_state.json'

    if mode == 'open_vqa':
        tasks = load_ctrate_open_tasks(ctrate_input, ctrate_root, limit=test_limit)
    else:
        tasks = load_ctrate_mcq_tasks(ctrate_input, ctrate_root, limit=test_limit)

    done = load_completed_ids(out_path) if resume else set()
    pending = [task for task in tasks if task.task_id not in done]
    state = load_state(state_path) if resume else {}
    state.update(
        {
            'mode': prefix,
            'status': 'running',
            'total_loaded': len(tasks),
            'already_completed': len(done),
            'pending': len(pending),
            'target_count': target_count,
        }
    )
    save_state(state_path, state)

    if target_count is not None and len(done) >= target_count:
        state.update({'status': 'complete', 'saved_this_run': 0, 'failed_this_run': 0, 'remaining_estimate': 0})
        save_state(state_path, state)
        return state

    generator = LocalGenerator(
        GeneratorConfig(
            model_dir=model_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            distributed_executor_backend=distributed_executor_backend,
        )
    )

    saved = 0
    failed = 0
    for start in tqdm(range(0, len(pending), batch_size), desc=f'synthesize {prefix}', unit='batch'):
        batch = pending[start : start + batch_size]
        outputs = generator.generate_batch(batch)
        for task, raw in zip(batch, outputs):
            data = extract_json_block(raw)
            if not isinstance(data, dict):
                failed += 1
                append_jsonl(failed_path, {'id': task.task_id, 'reason': 'json_parse_failed'})
                append_jsonl(failed_raw_path, {'id': task.task_id, 'reason': 'json_parse_failed', 'raw_output': raw})
                continue
            result = validate_generated_dict(data, task)
            if not result.ok:
                failed += 1
                append_jsonl(failed_path, {'id': task.task_id, 'reason': result.reason})
                append_jsonl(
                    failed_raw_path,
                    {'id': task.task_id, 'reason': result.reason, 'raw_output': raw, 'parsed_output': data},
                )
                continue
            append_jsonl(out_path, build_output_entry(task, data, raw))
            saved += 1
            if target_count is not None and (len(done) + saved) >= target_count:
                state.update({'saved_this_run': saved, 'failed_this_run': failed, 'remaining_estimate': 0, 'status': 'complete'})
                save_state(state_path, state)
                return state

        state.update({'saved_this_run': saved, 'failed_this_run': failed, 'remaining_estimate': max(0, len(pending) - (start + len(batch)))})
        save_state(state_path, state)

    state.update({'status': 'complete', 'saved_this_run': saved, 'failed_this_run': failed, 'remaining_estimate': 0})
    save_state(state_path, state)
    return state


def run_ctrate_open(**kwargs) -> dict:
    return run_ctrate_mode(mode='open_vqa', **kwargs)


def run_ctrate_mcq(**kwargs) -> dict:
    return run_ctrate_mode(mode='mcq_vqa', **kwargs)


def run_build_mix(
    *,
    ctrate_open_path: Path,
    m3d_mcq_path: Path,
    lidc_mixed_path: Path,
    output_dir: Path,
    total_rows: int,
    task4_min_target: int,
    limit_per_source: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ctrate_rows, m3d_rows, lidc_rows = load_three_source_rows(
        ctrate_open_path=ctrate_open_path,
        m3d_mcq_path=m3d_mcq_path,
        lidc_mixed_path=lidc_mixed_path,
        limit_per_source=limit_per_source,
    )
    return build_mixed_dataset(
        ctrate_rows=ctrate_rows,
        m3d_rows=m3d_rows,
        lidc_rows=lidc_rows,
        output_jsonl=output_dir / 'mixed_dataset.jsonl',
        output_summary=output_dir / 'summary.json',
        total_rows=total_rows,
        task4_min_target=task4_min_target,
    )

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Dict

from .config import (
    DEFAULT_CTRATE_OPEN_INPUT,
    DEFAULT_CTRATE_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRAINING_DATA_ROOT,
    OLD_SOURCE_COUNTS,
    SOURCE_OUTPUT_FILES,
    TASK4_CATEGORY_MIN_TARGET,
)
from .io_utils import load_state, save_state
from .mixer import _allocate_by_ratio
from .synthesizer import run_build_mix, run_ctrate_open

PROJECT_ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
M3D_WRAPPER = PROJECT_ROOT / 'synthesize_m3d_ctrate_cot_local_qwen72b.py'
LIDC_SCRIPT = PROJECT_ROOT / 'synthesize_advanced_cot_all_in_one.py'


def ensure_version(training_data_root: Path, version: str | None) -> str:
    if version:
        return version
    mixed_root = training_data_root / 'mixed'
    existing = []
    if mixed_root.exists():
        for p in mixed_root.iterdir():
            if p.is_dir() and p.name.startswith('v') and p.name[1:].isdigit():
                existing.append(int(p.name[1:]))
    next_idx = (max(existing) + 1) if existing else 1
    return f'v{next_idx:03d}'


def source_dir(training_data_root: Path, source: str, version: str) -> Path:
    return training_data_root / source / version


def mixed_dir(training_data_root: Path, version: str) -> Path:
    return training_data_root / 'mixed' / version


def output_path_for_source(training_data_root: Path, source: str, version: str) -> Path:
    return source_dir(training_data_root, source, version) / SOURCE_OUTPUT_FILES[source]


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def write_source_manifest(path: Path, payload: dict) -> None:
    save_state(path / 'manifest.json', payload)


def build_source_targets(total_rows: int, source_targets: Dict[str, int] | None = None) -> Dict[str, int]:
    return dict(source_targets or _allocate_by_ratio(total_rows, OLD_SOURCE_COUNTS))


def _visible_gpu_count(gpu_devices: str) -> int:
    return max(1, len([x for x in str(gpu_devices).split(',') if x.strip()]))


def run_ctrate_source(*, training_data_root: Path, version: str, target_count: int, model_dir: str, gpu_devices: str, tensor_parallel_size: int | None, batch_size: int, max_model_len: int, temperature: float, top_p: float, max_tokens: int, distributed_executor_backend: str, test_limit: int | None = None) -> dict:
    out_dir = source_dir(training_data_root, 'ctrate_open', version)
    out_dir.mkdir(parents=True, exist_ok=True)
    tp = tensor_parallel_size or _visible_gpu_count(gpu_devices)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    state = run_ctrate_open(
        model_dir=model_dir,
        output_dir=out_dir,
        ctrate_root=DEFAULT_CTRATE_ROOT,
        ctrate_open_input=DEFAULT_CTRATE_OPEN_INPUT,
        batch_size=batch_size,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        distributed_executor_backend=distributed_executor_backend,
        resume=True,
        test_limit=test_limit,
        target_count=target_count,
    )
    produced = count_rows(out_dir / SOURCE_OUTPUT_FILES['ctrate_open'])
    manifest = {
        'source': 'ctrate_open',
        'version': version,
        'target_count': target_count,
        'produced_count': produced,
        'model_dir': model_dir,
        'gpu_devices': gpu_devices,
        'tensor_parallel_size': tp,
        'backend': distributed_executor_backend,
        'status': state.get('status', 'unknown'),
    }
    write_source_manifest(out_dir, manifest)
    return manifest


def run_m3d_source(*, training_data_root: Path, version: str, target_count: int, model_dir: str, gpu_devices: str, tensor_parallel_size: int | None, batch_size: int, max_model_len: int, temperature: float, top_p: float, max_tokens: int, distributed_executor_backend: str) -> dict:
    out_dir = source_dir(training_data_root, 'm3d_mcq', version)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = count_rows(out_dir / SOURCE_OUTPUT_FILES['m3d_mcq'])
    if current < target_count:
        candidate_limit = max(target_count + 64, math.ceil(target_count * 1.25))
        cmd = [
            sys.executable, str(M3D_WRAPPER),
            '--mode', 'm3d_mcq',
            '--output-dir', str(out_dir),
            '--model-dir', model_dir,
            '--gpu-devices', gpu_devices,
            '--batch-size', str(batch_size),
            '--max-model-len', str(max_model_len),
            '--temperature', str(temperature),
            '--top-p', str(top_p),
            '--max-tokens', str(max_tokens),
            '--distributed-executor-backend', distributed_executor_backend,
            '--test-limit', str(candidate_limit),
            '--write-filter-audit',
        ]
        if tensor_parallel_size is not None:
            cmd.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
        subprocess.run(cmd, check=True)
    produced = count_rows(out_dir / SOURCE_OUTPUT_FILES['m3d_mcq'])
    manifest = {
        'source': 'm3d_mcq',
        'version': version,
        'target_count': target_count,
        'produced_count': produced,
        'model_dir': model_dir,
        'gpu_devices': gpu_devices,
        'tensor_parallel_size': tensor_parallel_size or _visible_gpu_count(gpu_devices),
        'backend': distributed_executor_backend,
        'status': 'complete' if produced >= target_count else 'partial',
    }
    write_source_manifest(out_dir, manifest)
    return manifest


def run_lidc_source(*, training_data_root: Path, version: str, target_count: int, model_dir: str, gpu_devices: str, tensor_parallel_size: int | None, batch_size: int, max_model_len: int, temperature: float, top_p: float, max_tokens: int) -> dict:
    out_dir = source_dir(training_data_root, 'lidc_mixed', version)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = count_rows(out_dir / SOURCE_OUTPUT_FILES['lidc_mixed'])
    if current < target_count:
        # One deployment per run: pick a case limit that should slightly exceed target.
        avg_items_per_case = 4.0
        case_limit = max(8, math.ceil((target_count * 1.15) / avg_items_per_case))
        cmd = [
            sys.executable, str(LIDC_SCRIPT),
            '--model-dir', model_dir,
            '--output-dir', str(out_dir),
            '--gpu-devices', gpu_devices,
            '--batch-size', str(max(1, min(batch_size, 6))),
            '--max-model-len', str(max_model_len),
            '--temperature', str(temperature),
            '--top-p', str(top_p),
            '--max-tokens', str(max_tokens),
            '--test-case-limit', str(case_limit),
        ]
        if tensor_parallel_size is not None:
            cmd.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
        subprocess.run(cmd, check=True)
    produced = count_rows(out_dir / SOURCE_OUTPUT_FILES['lidc_mixed'])
    manifest = {
        'source': 'lidc_mixed',
        'version': version,
        'target_count': target_count,
        'produced_count': produced,
        'model_dir': model_dir,
        'gpu_devices': gpu_devices,
        'tensor_parallel_size': tensor_parallel_size or _visible_gpu_count(gpu_devices),
        'status': 'complete' if produced >= target_count else 'partial',
    }
    write_source_manifest(out_dir, manifest)
    return manifest


def run_versioned_synthesis(*, training_data_root: Path = DEFAULT_TRAINING_DATA_ROOT, version: str | None = None, total_rows: int = 2425, source_targets: Dict[str, int] | None = None, model_dir: str = DEFAULT_MODEL_DIR, gpu_devices: str = '0,1,2,3', tensor_parallel_size: int | None = None, batch_size: int = 8, max_model_len: int = 8192, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 3072, distributed_executor_backend: str = 'mp', task4_min_target: int = TASK4_CATEGORY_MIN_TARGET, build_mix_after: bool = True) -> dict:
    training_data_root.mkdir(parents=True, exist_ok=True)
    resolved_version = ensure_version(training_data_root, version)
    targets = build_source_targets(total_rows, source_targets)

    manifests = {
        'ctrate_open': run_ctrate_source(
            training_data_root=training_data_root,
            version=resolved_version,
            target_count=targets['ctrate_open'],
            model_dir=model_dir,
            gpu_devices=gpu_devices,
            tensor_parallel_size=tensor_parallel_size,
            batch_size=batch_size,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            distributed_executor_backend=distributed_executor_backend,
        ),
        'm3d_mcq': run_m3d_source(
            training_data_root=training_data_root,
            version=resolved_version,
            target_count=targets['m3d_mcq'],
            model_dir=model_dir,
            gpu_devices=gpu_devices,
            tensor_parallel_size=tensor_parallel_size,
            batch_size=batch_size,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            distributed_executor_backend=distributed_executor_backend,
        ),
        'lidc_mixed': run_lidc_source(
            training_data_root=training_data_root,
            version=resolved_version,
            target_count=targets['lidc_mixed'],
            model_dir=model_dir,
            gpu_devices=gpu_devices,
            tensor_parallel_size=tensor_parallel_size,
            batch_size=batch_size,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    }

    mix_summary = None
    incomplete_sources = {
        name: {
            'target_count': targets[name],
            'produced_count': manifests[name].get('produced_count', 0),
        }
        for name in manifests
        if manifests[name].get('produced_count', 0) < targets[name]
    }
    if build_mix_after and not incomplete_sources:
        mix_out_dir = mixed_dir(training_data_root, resolved_version)
        mix_summary = run_build_mix(
            ctrate_open_path=output_path_for_source(training_data_root, 'ctrate_open', resolved_version),
            m3d_mcq_path=output_path_for_source(training_data_root, 'm3d_mcq', resolved_version),
            lidc_mixed_path=output_path_for_source(training_data_root, 'lidc_mixed', resolved_version),
            output_dir=mix_out_dir,
            total_rows=total_rows,
            task4_min_target=task4_min_target,
            limit_per_source=None,
        )
        save_state(mix_out_dir / 'manifest.json', {
            'version': resolved_version,
            'total_rows': total_rows,
            'source_targets': targets,
            'task4_min_target': task4_min_target,
            'source_manifests': manifests,
            'mix_summary': mix_summary,
        })

    return {
        'version': resolved_version,
        'training_data_root': str(training_data_root),
        'source_targets': targets,
        'source_manifests': manifests,
        'incomplete_sources': incomplete_sources,
        'mix_summary': mix_summary,
    }

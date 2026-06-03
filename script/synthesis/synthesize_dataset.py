#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = THIS_DIR.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from synthesis.config import (
    DEFAULT_CTRATE_MCQ_INPUT,
    DEFAULT_CTRATE_OPEN_INPUT,
    DEFAULT_CTRATE_ROOT,
    DEFAULT_LIDC_MIXED_INPUT,
    DEFAULT_M3D_MCQ_INPUT,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRAINING_DATA_ROOT,
    OLD_SOURCE_COUNTS,
    TASK4_CATEGORY_MIN_TARGET,
)
from synthesis.orchestrator import build_source_targets, run_versioned_synthesis
from synthesis.synthesizer import run_build_mix, run_ctrate_mcq, run_ctrate_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Refactored synthesis entrypoint.')
    sub = parser.add_subparsers(dest='command', required=True)

    s1 = sub.add_parser('synthesize-ctrate-open')
    s1.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
    s1.add_argument('--output-dir', type=Path, required=True)
    s1.add_argument('--ctrate-root', type=Path, default=DEFAULT_CTRATE_ROOT)
    s1.add_argument('--ctrate-open-input', type=Path, default=DEFAULT_CTRATE_OPEN_INPUT)
    s1.add_argument('--gpu-devices', type=str, default='0')
    s1.add_argument('--tensor-parallel-size', type=int, default=None)
    s1.add_argument('--batch-size', type=int, default=8)
    s1.add_argument('--max-model-len', type=int, default=8192)
    s1.add_argument('--temperature', type=float, default=0.7)
    s1.add_argument('--top-p', type=float, default=0.9)
    s1.add_argument('--max-tokens', type=int, default=3072)
    s1.add_argument('--distributed-executor-backend', choices=['mp', 'ray'], default='mp')
    s1.add_argument('--resume', action='store_true', default=True)
    s1.add_argument('--test-limit', type=int, default=None)
    s1.add_argument('--target-count', type=int, default=None)

    s1b = sub.add_parser('synthesize-ctrate-mcq')
    s1b.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
    s1b.add_argument('--output-dir', type=Path, required=True)
    s1b.add_argument('--ctrate-root', type=Path, default=DEFAULT_CTRATE_ROOT)
    s1b.add_argument('--ctrate-mcq-input', type=Path, default=DEFAULT_CTRATE_MCQ_INPUT)
    s1b.add_argument('--gpu-devices', type=str, default='0')
    s1b.add_argument('--tensor-parallel-size', type=int, default=None)
    s1b.add_argument('--batch-size', type=int, default=8)
    s1b.add_argument('--max-model-len', type=int, default=8192)
    s1b.add_argument('--temperature', type=float, default=0.7)
    s1b.add_argument('--top-p', type=float, default=0.9)
    s1b.add_argument('--max-tokens', type=int, default=3072)
    s1b.add_argument('--distributed-executor-backend', choices=['mp', 'ray'], default='mp')
    s1b.add_argument('--resume', action='store_true', default=True)
    s1b.add_argument('--test-limit', type=int, default=None)
    s1b.add_argument('--target-count', type=int, default=None)

    s2 = sub.add_parser('build-mix')
    s2.add_argument('--output-dir', type=Path, required=True)
    s2.add_argument('--ctrate-open-path', type=Path, default=DEFAULT_CTRATE_OPEN_INPUT)
    s2.add_argument('--m3d-mcq-path', type=Path, default=DEFAULT_M3D_MCQ_INPUT)
    s2.add_argument('--lidc-mixed-path', type=Path, default=DEFAULT_LIDC_MIXED_INPUT)
    s2.add_argument('--total-rows', type=int, default=2425)
    s2.add_argument('--task4-min-target', type=int, default=TASK4_CATEGORY_MIN_TARGET)
    s2.add_argument('--limit-per-source', type=int, default=None)

    s3 = sub.add_parser('run-version')
    s3.add_argument('--training-data-root', type=Path, default=DEFAULT_TRAINING_DATA_ROOT)
    s3.add_argument('--version', type=str, default=None)
    s3.add_argument('--total-rows', type=int, default=2425)
    s3.add_argument('--ctrate-open-target', type=int, default=None)
    s3.add_argument('--m3d-mcq-target', type=int, default=None)
    s3.add_argument('--lidc-mixed-target', type=int, default=None)
    s3.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
    s3.add_argument('--gpu-devices', type=str, default='0,1,2,3')
    s3.add_argument('--tensor-parallel-size', type=int, default=None)
    s3.add_argument('--batch-size', type=int, default=8)
    s3.add_argument('--max-model-len', type=int, default=8192)
    s3.add_argument('--temperature', type=float, default=0.7)
    s3.add_argument('--top-p', type=float, default=0.9)
    s3.add_argument('--max-tokens', type=int, default=3072)
    s3.add_argument('--distributed-executor-backend', choices=['mp', 'ray'], default='mp')
    s3.add_argument('--task4-min-target', type=int, default=TASK4_CATEGORY_MIN_TARGET)
    s3.add_argument('--skip-build-mix', action='store_true')

    return parser.parse_args()


def resolve_tensor_parallel_size(gpu_devices: str, explicit_tp: int | None) -> int:
    visible = gpu_devices.strip()
    visible_count = max(1, len([x for x in visible.split(',') if x.strip()])) if visible else 1
    if explicit_tp is not None:
        if explicit_tp > visible_count:
            raise SystemExit(
                f'tensor_parallel_size={explicit_tp} exceeds visible GPU count {visible_count} '
                f"under CUDA_VISIBLE_DEVICES='{visible}'."
            )
        return explicit_tp
    return visible_count


def main() -> None:
    args = parse_args()
    if args.command == 'synthesize-ctrate-open':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        tp = resolve_tensor_parallel_size(args.gpu_devices, args.tensor_parallel_size)
        state = run_ctrate_open(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            ctrate_root=args.ctrate_root,
            ctrate_input=args.ctrate_open_input,
            batch_size=args.batch_size,
            tensor_parallel_size=tp,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            resume=args.resume,
            test_limit=args.test_limit,
            target_count=args.target_count,
        )
        print(state)
    elif args.command == 'synthesize-ctrate-mcq':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        tp = resolve_tensor_parallel_size(args.gpu_devices, args.tensor_parallel_size)
        state = run_ctrate_mcq(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            ctrate_root=args.ctrate_root,
            ctrate_input=args.ctrate_mcq_input,
            batch_size=args.batch_size,
            tensor_parallel_size=tp,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            resume=args.resume,
            test_limit=args.test_limit,
            target_count=args.target_count,
        )
        print(state)
    elif args.command == 'build-mix':
        summary = run_build_mix(
            ctrate_open_path=args.ctrate_open_path,
            m3d_mcq_path=args.m3d_mcq_path,
            lidc_mixed_path=args.lidc_mixed_path,
            output_dir=args.output_dir,
            total_rows=args.total_rows,
            task4_min_target=args.task4_min_target,
            limit_per_source=args.limit_per_source,
        )
        print(summary)
    elif args.command == 'run-version':
        source_targets = build_source_targets(args.total_rows)
        for key, value in [('ctrate_open', args.ctrate_open_target), ('m3d_mcq', args.m3d_mcq_target), ('lidc_mixed', args.lidc_mixed_target)]:
            if value is not None:
                source_targets[key] = value
        summary = run_versioned_synthesis(
            training_data_root=args.training_data_root,
            version=args.version,
            total_rows=args.total_rows,
            source_targets=source_targets,
            model_dir=args.model_dir,
            gpu_devices=args.gpu_devices,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            task4_min_target=args.task4_min_target,
            build_mix_after=not args.skip_build_mix,
        )
        print(summary)


if __name__ == '__main__':
    main()

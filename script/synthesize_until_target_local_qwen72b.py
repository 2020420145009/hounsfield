#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WRAPPER_SCRIPT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/synthesize_m3d_ctrate_cot_local_qwen72b.py')
MODE_TO_OUTPUT = {
    'm3d_mcq': 'm3d_mcq_cot.jsonl',
    'm3d_open': 'm3d_open_cot.jsonl',
    'ctrate_mcq': 'ctrate_mcq_cot.jsonl',
    'ctrate_open': 'ctrate_open_cot.jsonl',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Keep synthesizing until target number of accepted rows is reached, with resume support.'
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=sorted(MODE_TO_OUTPUT), default='ctrate_open')
    parser.add_argument('--target-count', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu-devices', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=None)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--wait-for-idle-gpu', action='store_true')
    parser.add_argument('--max-rounds', type=int, default=100)
    parser.add_argument('--max-stall-rounds', type=int, default=3)
    parser.add_argument('--sleep-sec', type=int, default=5)
    parser.add_argument('--test-limit', type=int, default=None)
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def state_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f'{mode}.until_target_state.json'


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def build_cmd(args: argparse.Namespace, effective_limit: int) -> list[str]:
    cmd = [
        sys.executable,
        str(WRAPPER_SCRIPT),
        '--mode', args.mode,
        '--output-dir', str(args.output_dir),
        '--batch-size', str(args.batch_size),
        '--max-model-len', str(args.max_model_len),
        '--temperature', str(args.temperature),
        '--top-p', str(args.top_p),
        '--max-tokens', str(args.max_tokens),
        '--distributed-executor-backend', 'mp',
        '--test-limit', str(effective_limit),
    ]
    if args.gpu_devices is not None:
        cmd.extend(['--gpu-devices', args.gpu_devices])
    if args.tensor_parallel_size is not None:
        cmd.extend(['--tensor-parallel-size', str(args.tensor_parallel_size)])
    if args.wait_for_idle_gpu:
        cmd.append('--wait-for-idle-gpu')
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / MODE_TO_OUTPUT[args.mode]
    spath = state_path(args.output_dir, args.mode)

    current_count = count_rows(output_path)
    stall_rounds = 0
    round_idx = 0

    state = {
        'mode': args.mode,
        'target_count': args.target_count,
        'output_path': str(output_path),
        'status': 'running',
        'current_count': current_count,
        'rounds_completed': 0,
        'stall_rounds': 0,
    }
    save_state(spath, state)

    while current_count < args.target_count and round_idx < args.max_rounds:
        round_idx += 1
        effective_limit = args.test_limit or args.target_count
        if effective_limit < args.target_count:
            effective_limit = args.target_count
        cmd = build_cmd(args, effective_limit)
        print(f'[round {round_idx}] current={current_count} target={args.target_count} limit={effective_limit}')
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd)
        new_count = count_rows(output_path)
        progress = new_count - current_count
        current_count = new_count

        if result.returncode != 0:
            state.update({
                'status': 'child_failed',
                'current_count': current_count,
                'rounds_completed': round_idx,
                'stall_rounds': stall_rounds,
                'last_returncode': result.returncode,
            })
            save_state(spath, state)
            raise SystemExit(result.returncode)

        if progress <= 0:
            stall_rounds += 1
        else:
            stall_rounds = 0

        state.update({
            'status': 'running',
            'current_count': current_count,
            'rounds_completed': round_idx,
            'stall_rounds': stall_rounds,
        })
        save_state(spath, state)

        print(f'[round {round_idx}] new_count={current_count} progress={progress}')

        if current_count >= args.target_count:
            break
        if stall_rounds >= args.max_stall_rounds:
            state.update({
                'status': 'stalled',
                'current_count': current_count,
                'rounds_completed': round_idx,
                'stall_rounds': stall_rounds,
            })
            save_state(spath, state)
            print(f'Stopped after {stall_rounds} stalled rounds at {current_count} rows.')
            return
        time.sleep(args.sleep_sec)

    state.update({
        'status': 'complete' if current_count >= args.target_count else 'max_rounds_reached',
        'current_count': current_count,
        'rounds_completed': round_idx,
        'stall_rounds': stall_rounds,
    })
    save_state(spath, state)
    print(f'Finished with {current_count} rows. State: {spath}')


if __name__ == '__main__':
    main()

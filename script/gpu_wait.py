import os
import time
import subprocess


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, (list, tuple)):
        return [int(x) for x in gpu_ids]
    return [int(x.strip()) for x in str(gpu_ids).split(',') if x.strip()]


def _query_nvidia_smi():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    stats = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            stats[idx] = {'used_memory_mb': int(parts[1]), 'utilization': int(parts[2])}
        except ValueError:
            continue
    return stats


def wait_for_gpus(gpu_ids=None, max_used_memory_mb=4096, max_utilization=20, poll_interval_sec=60, label='job'):
    ids = parse_gpu_ids(gpu_ids)
    if not ids:
        return
    while True:
        stats = _query_nvidia_smi()
        if not stats:
            return
        ready = []
        for gid in ids:
            st = stats.get(gid)
            if not st:
                continue
            if st['used_memory_mb'] <= max_used_memory_mb and st['utilization'] <= max_utilization:
                ready.append(gid)
        if len(ready) == len(ids):
            print(f'[gpu_wait] {label}: GPUs ready -> {ready}')
            return
        print(f'[gpu_wait] {label}: waiting for GPUs {ids}, ready={ready}, sleep={poll_interval_sec}s')
        time.sleep(poll_interval_sec)

#!/usr/bin/env python3
import atexit
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _service_config():
    env_dir = os.environ.get("ENV_DIR", "/volume/med-train/users/shuyan/envs/qwen-vllm-517")
    model_dir = os.environ.get("MODEL_DIR", "/volume/med-train/users/shuyan/models/Qwen3.6-35B-A3B")
    service_dir = os.environ.get("SERVICE_DIR", "/volume/med-train/users/shuyan/services/qwen36_35b_a3b_vllm")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "18036"))
    tensor_parallel_size = os.environ.get("TENSOR_PARALLEL_SIZE", "4")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "131072")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.92")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES_VALUE", "0,1,2,3")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "Qwen/Qwen3.6-35B-A3B")
    return {
        "env_dir": env_dir,
        "model_dir": model_dir,
        "service_dir": service_dir,
        "host": host,
        "port": port,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "cuda_visible_devices": cuda_visible_devices,
        "served_model_name": served_model_name,
    }


def _probe(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as resp:
            return 200 <= resp.status < 500
    except (URLError, TimeoutError, ConnectionError, OSError):
        return False


def wait_for_server(base_url: str, timeout: float = 600.0):
    deadline = time.time() + timeout
    probe_url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        if _probe(probe_url):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Qwen server at {probe_url}")


def start_server(log_file: str | None = None):
    cfg = _service_config()
    env_dir = Path(cfg["env_dir"])
    model_dir = Path(cfg["model_dir"])
    if not (env_dir / "bin" / "vllm").exists():
        raise FileNotFoundError(f"vllm binary not found: {env_dir / 'bin' / 'vllm'}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    service_dir = Path(cfg["service_dir"])
    service_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = str(service_dir / "qwen36_oneshot.log")
    log_handle = open(log_file, "a")

    cmd = [
        str(env_dir / "bin" / "vllm"),
        "serve",
        str(model_dir),
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "--tensor-parallel-size", str(cfg["tensor_parallel_size"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--reasoning-parser", "qwen3",
        "--served-model-name", cfg["served_model_name"],
    ]
    proc_env = os.environ.copy()
    proc_env["PYTHONNOUSERSITE"] = "1"
    proc_env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=proc_env,
        start_new_session=True,
    )
    process._qwen_log_handle = log_handle
    atexit.register(stop_server, process)
    return process


def stop_server(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
    log_handle = getattr(process, "_qwen_log_handle", None)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()


@contextmanager
def managed_server(start_if_needed: bool = True, timeout: float = 600.0):
    cfg = _service_config()
    base_url = os.environ.get("OPENAI_BASE_URL", f"http://{cfg['host']}:{cfg['port']}/v1")
    if not start_if_needed:
        yield base_url, cfg["served_model_name"], None
        return
    process = start_server()
    try:
        wait_for_server(base_url, timeout=timeout)
        yield base_url, cfg["served_model_name"], process
    finally:
        stop_server(process)

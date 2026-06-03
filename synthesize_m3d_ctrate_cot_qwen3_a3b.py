#!/usr/bin/env python3
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LARGER_OFFICIAL_MOE_CANDIDATE = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LOCAL_FALLBACK_CANDIDATES = [
    "/volume/med-train/shared_models/DeepSeek-R1-Distill-Qwen-72B",
    "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B",
]


def ensure_default_model_arg(argv: list[str]) -> list[str]:
    has_model_dir = any(arg == "--model-dir" or arg.startswith("--model-dir=") for arg in argv[1:])
    if has_model_dir:
        return argv
    return [argv[0], "--model-dir", DEFAULT_MODEL_DIR, *argv[1:]]


def extract_model_dir(argv: list[str]) -> str:
    for idx, arg in enumerate(argv[1:], start=1):
        if arg == "--model-dir" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--model-dir="):
            return arg.split("=", 1)[1]
    return DEFAULT_MODEL_DIR


def looks_like_local_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("./") or value.startswith("../")


def assert_model_accessible(model_dir: str) -> None:
    if looks_like_local_path(model_dir):
        if Path(model_dir).exists():
            return
        raise SystemExit(f"Local model path does not exist: {model_dir}")

    message = [
        f"Remote model repo configured: {model_dir}",
        "This environment is currently trying to query Hugging Face online during vLLM init.",
        "The model is not present locally, so initialization hangs or fails before synthesis starts.",
        "",
        "Options:",
        f"1. Download the model to a local directory first, then run with `--model-dir /local/path/to/model`.",
        f"2. Use an existing local model, for example one of: {', '.join(p for p in LOCAL_FALLBACK_CANDIDATES if Path(p).exists()) or 'none found'}.",
        f"3. If you want a larger official Qwen MoE later, the next candidate is `{LARGER_OFFICIAL_MOE_CANDIDATE}`, but it also needs local weights first.",
    ]
    raise SystemExit("\n".join(message))


def main() -> None:
    sys.argv = ensure_default_model_arg(sys.argv)
    model_dir = extract_model_dir(sys.argv)
    assert_model_accessible(model_dir)
    from synthesize_m3d_ctrate_cot import main as base_main

    base_main()


if __name__ == "__main__":
    main()

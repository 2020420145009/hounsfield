#!/usr/bin/env python3
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"


def ensure_default_args(argv: list[str]) -> list[str]:
    out = list(argv)
    if not any(arg == "--model-dir" or arg.startswith("--model-dir=") for arg in out[1:]):
        out = [out[0], "--model-dir", DEFAULT_MODEL_DIR, *out[1:]]
    if not any(arg == "--ctrate-clean-profile" or arg.startswith("--ctrate-clean-profile=") for arg in out[1:]):
        out = [out[0], "--ctrate-clean-profile", "task4_analysis_v2", *out[1:]]
    if not any(arg == "--distributed-executor-backend" or arg.startswith("--distributed-executor-backend=") for arg in out[1:]):
        out = [out[0], "--distributed-executor-backend", "mp", *out[1:]]
    if "--write-filter-audit" not in out[1:]:
        out = [*out, "--write-filter-audit"]
    return out


def assert_local_model_exists(model_dir: str) -> None:
    if not Path(model_dir).exists():
        raise SystemExit(f"Local model path does not exist: {model_dir}")


def main() -> None:
    sys.argv = ensure_default_args(sys.argv)
    model_dir = sys.argv[sys.argv.index("--model-dir") + 1]
    assert_local_model_exists(model_dir)
    from synthesize_m3d_ctrate_cot import main as base_main

    base_main()


if __name__ == "__main__":
    main()

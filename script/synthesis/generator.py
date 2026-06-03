from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .prompt_builder import build_system_prompt, build_user_content
from .schema import SynthesisTask


@dataclass
class GeneratorConfig:
    model_dir: str
    tensor_parallel_size: int
    max_model_len: int
    temperature: float
    top_p: float
    max_tokens: int
    distributed_executor_backend: str


class LocalGenerator:
    def __init__(self, config: GeneratorConfig):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=config.model_dir,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
            distributed_executor_backend=config.distributed_executor_backend,
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

    def generate_batch(self, tasks: List[SynthesisTask]) -> List[str]:
        messages = []
        for task in tasks:
            messages.append(
                [
                    {'role': 'system', 'content': build_system_prompt(task)},
                    {'role': 'user', 'content': build_user_content(task)},
                ]
            )
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

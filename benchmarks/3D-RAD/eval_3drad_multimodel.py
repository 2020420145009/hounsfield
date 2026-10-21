#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import nltk
import pandas as pd
import torch
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, CLIPImageProcessor
from transformers.utils import logging as hf_logging
from slice_cache_utils import load_or_create_cached_slice_images, load_volume

DEFAULT_BERTSCORE_MODEL = "/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/models/metrics/roberta-large"
DEFAULT_NLTK_DATA = "/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/models/metrics/nltk_data"
DEFAULT_BERTSCORE_LAYERS = 17
DEFAULT_REASONING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user a concise "
    "final answer in a short word or phrase. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively."
)
CURRENT_INFERENCE_COT_PROMPT = """You are an expert thoracic radiologist. You are given 8 axial CT slices from a single chest CT study, displayed in order from Slice 1 (lung apex) to Slice 8 (lung base). Your task is to answer the user's question using ONLY the provided images.
Your response must strictly follow this format:
<think>
[Your detailed reasoning]
</think>
<answer>
[The exact answer, concise and precise]
</answer>
In your reasoning, you must include:
1. Slice-by-slice dynamic description (how the finding appears, peaks, and disappears/changes across slices).
2. 3D spatial awareness (confirm the finding is a real 3D entity, not an artifact).
3. Differential exclusion (mention at least one mimic and explain why it is excluded)."""


def build_ct_slice_description(num_slices: int) -> str:
    return (
        f"This input contains {num_slices} grayscale CT slice images sampled in order from a single 3D CT study along one anatomical axis. "
        f"Each image is one slice from the same scan, and all slices should be interpreted jointly as one volumetric case."
    )


def normalize_question_text(question: str) -> str:
    text = str(question or "")
    text = text.replace("<image>\n", " ").replace("<image>", " ")
    text = re.sub(r"\[Perform .*?\]\s*", "", text)
    text = re.sub(r"(?:<image>\s*)+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



os.environ.setdefault("NLTK_DATA", DEFAULT_NLTK_DATA)
if DEFAULT_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, DEFAULT_NLTK_DATA)

# Suppress known harmless warnings from generation defaults and BERTScore backbone loading.
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of RobertaModel were not initialized.*roberta\.pooler\.dense.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"You should probably TRAIN this model on a down-stream task.*",
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class BenchmarkSample:
    image_path: str
    question: str
    answer: str
    question_aspect: str
    answer_choice: Optional[str]
    is_closed: bool


class BenchmarkDataset:
    def __init__(self, data_root_csv: str, vqa_csv: str, close_ended: bool):
        self.data_list = pd.read_csv(vqa_csv)
        self.data_root_df = pd.read_csv(data_root_csv)
        self.data_root_df.set_index('VolumeName', inplace=True)
        self.close_ended = close_ended

    def __len__(self):
        return len(self.data_list)

    def _valid_text(self, value: object) -> bool:
        return pd.notna(value) and str(value).strip() not in {"", "-", "nan"}

    def __getitem__(self, idx: int) -> BenchmarkSample:
        data = self.data_list.iloc[idx]
        volume_name = data['VolumeName']
        image_abs_path = self.data_root_df.at[volume_name, 'Path']
        question = str(data['Question'])
        question_type = str(data.get('QuestionType', '')).strip().lower()
        answer_choice = data.get('AnswerChoice')
        choices = []
        for key in ['Choice A', 'Choice B', 'Choice C', 'Choice D']:
            if self._valid_text(data.get(key)):
                choices.append((key.split()[-1], str(data[key]).strip()))

        is_closed = self.close_ended or question_type in {'close', 'closed'}
        if is_closed and choices and self._valid_text(answer_choice):
            choice_text = ' '.join([f'{label}. {text}' for label, text in choices])
            question = (
                f"{question}\nChoices: {choice_text}\n"
                "Answer with the correct option letter and option text only."
            )
            answer = f"{str(answer_choice).strip()}. {str(data['Answer']).strip()}"
            answer_choice_value = str(answer_choice).strip()
        else:
            question = f"{question}\nAnswer concisely in a short phrase."
            answer = str(data['Answer']).strip()
            answer_choice_value = None
            is_closed = False

        return BenchmarkSample(
            image_path=image_abs_path,
            question=question,
            answer=answer,
            question_aspect=str(data['QuestionAspect']),
            answer_choice=answer_choice_value,
            is_closed=is_closed,
        )


class BaseAdapter:
    def generate_answer(self, sample: BenchmarkSample) -> str:
        raise NotImplementedError

    def generate_answers(self, samples: List[BenchmarkSample]) -> List[str]:
        return [self.generate_answer(sample) for sample in samples]

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        preds = self.generate_answers(samples)
        return preds, preds


class FlemingVLAdapter(BaseAdapter):
    def __init__(self, model_path: str, max_new_tokens: int, num_slices: int, use_flash_attn: bool, prompt_style: str, lora_path: Optional[str] = None, repetition_penalty: float = 1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        )
        if lora_path:
            from peft import PeftModel
            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model,
                lora_path,
                adapter_name='benchmark_lora',
            )
            self.model.language_model.set_adapter('benchmark_lora')
        if lora_path and hasattr(self.model, 'system_message'):
            if prompt_style == 'reasoning':
                self.model.system_message = CURRENT_INFERENCE_COT_PROMPT
            else:
                # plain: 微调后不用 CoT prompt —— 空 system prompt
                self.model.system_message = ''
        self.model = self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens
        self.num_slices = num_slices
        self.prompt_style = prompt_style
        self.lora_path = lora_path
        self.repetition_penalty = repetition_penalty

    def _volume_to_pixel_values(self, npy_path: str):
        images = load_or_create_cached_slice_images(
            npy_path,
            num_slices=self.num_slices,
            image_size=448,
        )
        pixel_values_list = []
        num_patches_list = []
        for image in images:
            pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list).to(
            device=self.device,
            dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
        )
        return pixel_values, num_patches_list

    def _extract_answer_letter(self, text: str) -> Optional[str]:
        s = str(text or '').strip()
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', s, flags=re.I | re.S)
        if answer_match:
            s = answer_match.group(1).strip()
        else:
            think_match = re.search(r'<think>.*?</think>', s, flags=re.I | re.S)
            if think_match:
                s = s[:think_match.start()] + s[think_match.end():]
        m = re.search(r'\b([A-E])\b', s)
        if m:
            return m.group(1).upper()
        m = re.search(r'\(([A-Ea-e])\)', s)
        if m:
            return m.group(1).upper()
        m = re.match(r'\s*([A-Ea-e])[\).:\s]', s)
        if m:
            return m.group(1).upper()
        return None

    def _extract_final_answer(self, response: str, is_closed: bool) -> str:
        match = re.search(r'<answer>(.*?)</answer>', response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
        else:
            # 先尝试删掉完整的 <think>...</think> 块
            fallback = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL).strip()
            # 若 <think> 有开头但无结尾（思维链被截断），截掉 <think> 及之后所有内容
            fallback = re.sub(r'<think>.*$', '', fallback, flags=re.IGNORECASE | re.DOTALL).strip()
            # 若剩余文本以 <answer> 开头（无关闭标签），去掉标签取后面内容
            fallback = re.sub(r'^<answer>\s*', '', fallback, flags=re.IGNORECASE).strip()
            final_answer = fallback
        if is_closed:
            letter = self._extract_answer_letter(final_answer or response)
            return letter if letter else final_answer.strip()
        return final_answer.strip()

    def _prepare_question_body(self, sample: BenchmarkSample) -> str:
        question = normalize_question_text(sample.question)
        question = re.sub(
            r'\n?Answer with the correct option letter and option text only\.?\s*$',
            '',
            question,
            flags=re.IGNORECASE,
        )
        question = re.sub(
            r'\n?Answer concisely in a short phrase\.?\s*$',
            '',
            question,
            flags=re.IGNORECASE,
        )
        question = question.rstrip()
        if sample.is_closed:
            question += (
                "\nReturn the response strictly as "
                "<think>...</think><answer>LETTER</answer>, where LETTER is the single "
                "correct option letter."
            )
        return question

    def _build_full_question(self, sample: BenchmarkSample) -> str:
        image_description = build_ct_slice_description(self.num_slices)
        question = self._prepare_question_body(sample)
        prompt_parts = [image_description, question]
        if self.lora_path:
            return "\n".join(prompt_parts)
        if self.prompt_style == 'reasoning':
            prompt_parts.insert(0, CURRENT_INFERENCE_COT_PROMPT)
        return "\n".join(prompt_parts)
    def generate_answers(self, samples: List[BenchmarkSample]) -> List[str]:
        if not samples:
            return []

        pixel_values_batches = []
        image_counts = []
        questions = []
        for sample in samples:
            pixel_values, _ = self._volume_to_pixel_values(sample.image_path)
            pixel_values_batches.append(pixel_values)
            image_counts.append(pixel_values.shape[0])
            questions.append(self._build_full_question(sample))

        pixel_values = torch.cat(pixel_values_batches, dim=0).to(
            device=self.device,
            dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
        )
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            questions,
            generation_config,
            num_patches_list=image_counts,
            history=None,
            return_history=False,
        )
        return [self._extract_final_answer(response, sample.is_closed) for response, sample in zip(responses, samples)]

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        if not samples:
            return [], []

        pixel_values_batches = []
        image_counts = []
        questions = []
        for sample in samples:
            pixel_values, _ = self._volume_to_pixel_values(sample.image_path)
            pixel_values_batches.append(pixel_values)
            image_counts.append(pixel_values.shape[0])
            questions.append(self._build_full_question(sample))

        pixel_values = torch.cat(pixel_values_batches, dim=0).to(
            device=self.device,
            dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
        )
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            questions,
            generation_config,
            num_patches_list=image_counts,
            history=None,
            return_history=False,
        )
        preds = [self._extract_final_answer(response, sample.is_closed) for response, sample in zip(responses, samples)]
        return preds, responses

    def generate_answer(self, sample: BenchmarkSample) -> str:
        return self.generate_answers([sample])[0]


class LingshuAdapter(BaseAdapter):
    def __init__(self, model_path: str, max_new_tokens: int, num_slices: int, use_flash_attn: bool, prompt_style: str, lora_path: Optional[str] = None, repetition_penalty: float = 1.0):
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        attn_impl = 'flash_attention_2' if use_flash_attn and self.device.type == 'cuda' else None
        model_kwargs = dict(
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
        )
        if attn_impl is not None:
            model_kwargs['attn_implementation'] = attn_impl
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(self.device).eval()
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                adapter_name='benchmark_lora',
            )
            self.model.set_adapter('benchmark_lora')
            self.model = self.model.eval()
        self.device = next(self.model.parameters()).device
        # Cap per-image vision tokens to avoid OOM in the vision tower when many slices are
        # batched. 448*448 px -> ~256 tokens/image; override via THREED_RAD_MAX_PIXELS.
        max_pixels = int(os.environ.get('THREED_RAD_MAX_PIXELS', 448 * 448))
        min_pixels = int(os.environ.get('THREED_RAD_MIN_PIXELS', 256 * 256))
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_path)
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
            if self.processor.tokenizer.pad_token_id is None and self.processor.tokenizer.eos_token_id is not None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.num_slices = num_slices
        self.prompt_style = prompt_style
        self.repetition_penalty = repetition_penalty

    def _volume_to_images(self, npy_path: str):
        return load_or_create_cached_slice_images(
            npy_path,
            num_slices=self.num_slices,
            image_size=448,
        )

    def _extract_answer_letter(self, text: str) -> Optional[str]:
        s = str(text or '').strip()
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', s, flags=re.I | re.S)
        if answer_match:
            s = answer_match.group(1).strip()
        m = re.search(r'\b([A-E])\b', s)
        if m:
            return m.group(1).upper()
        m = re.search(r'\(([A-Ea-e])\)', s)
        if m:
            return m.group(1).upper()
        m = re.match(r'\s*([A-Ea-e])[\).:\s]', s)
        if m:
            return m.group(1).upper()
        return None

    def _extract_final_answer(self, response: str, is_closed: bool) -> str:
        match = re.search(r'<answer>(.*?)</answer>', response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
        else:
            # 先尝试删掉完整的 <think>...</think> 块
            fallback = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL).strip()
            # 若 <think> 有开头但无结尾（思维链被截断），截掉 <think> 及之后所有内容
            fallback = re.sub(r'<think>.*$', '', fallback, flags=re.IGNORECASE | re.DOTALL).strip()
            # 若剩余文本以 <answer> 开头（无关闭标签），去掉标签取后面内容
            fallback = re.sub(r'^<answer>\s*', '', fallback, flags=re.IGNORECASE).strip()
            final_answer = fallback
        if is_closed:
            letter = self._extract_answer_letter(final_answer or response)
            return letter if letter else final_answer.strip()
        return final_answer.strip()

    def _prepare_question_body(self, sample: BenchmarkSample) -> str:
        question = normalize_question_text(sample.question)
        question = re.sub(r'\n?Answer with the correct option letter and option text only\.?\s*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\n?Answer concisely in a short phrase\.?\s*$', '', question, flags=re.IGNORECASE)
        question = question.rstrip()
        if self.prompt_style == 'reasoning':
            if sample.is_closed:
                question += '\nReturn the response strictly as <think>...</think><answer>LETTER</answer>, where LETTER is only A, B, C, or D.'
        return question

    def _build_messages(self, sample: BenchmarkSample, images: List[Image.Image]):
        question = self._prepare_question_body(sample)
        user_content = [{'type': 'image', 'image': image} for image in images]
        user_content.append({'type': 'text', 'text': f"{build_ct_slice_description(self.num_slices)}\n{question}"})
        messages = []
        if self.prompt_style == 'reasoning':
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': CURRENT_INFERENCE_COT_PROMPT}]})
        messages.append({'role': 'user', 'content': user_content})
        return messages

    def generate_answers(self, samples: List[BenchmarkSample]) -> List[str]:
        if not samples:
            return []
        texts = []
        flat_images = []
        for sample in samples:
            images = self._volume_to_images(sample.image_path)
            messages = self._build_messages(sample, images)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            flat_images.extend(images)
        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=self.repetition_penalty)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        responses = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [self._extract_final_answer(response.strip(), sample.is_closed) for response, sample in zip(responses, samples)]

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        if not samples:
            return [], []
        texts = []
        flat_images = []
        for sample in samples:
            images = self._volume_to_images(sample.image_path)
            messages = self._build_messages(sample, images)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            flat_images.extend(images)
        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=self.repetition_penalty)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        responses = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = [self._extract_final_answer(response.strip(), sample.is_closed) for response, sample in zip(responses, samples)]
        return preds, responses

    def generate_answer(self, sample: BenchmarkSample) -> str:
        return self.generate_answers([sample])[0]

class M3DAdapter(BaseAdapter):
    """M3D-LaMed adapter following official implementation.
    Loads volumes, resizes to 32x256x256, normalizes, and calls model.generate(image, input_id)."""

    def __init__(self, model_path: str, max_length: int, max_new_tokens: int, repetition_penalty: float = 1.0, proj_out_num: int = 256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=max_length,
            padding_side='right',
            use_fast=False,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.image_tokens = "<im_patch>" * proj_out_num

    def generate_answer(self, sample: BenchmarkSample) -> str:
        # Load volume from .nii.gz or .npy
        volume = load_volume(sample.image_path)  # Returns (D, H, W) numpy array

        # M3D expects (1, 1, D, H, W) with shape (1, 1, 32, 256, 256) and values in [0, 1]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()

        # Resize to M3D's expected input shape
        image = torch.nn.functional.interpolate(
            volume_tensor,
            size=(32, 256, 256),
            mode='trilinear',
            align_corners=False
        )

        # Min-max normalization to [0, 1]
        vmin, vmax = image.min(), image.max()
        if vmax > vmin:
            image = (image - vmin) / (vmax - vmin)

        image = image.to(device=self.device)

        # Prepend image tokens to question (following M3D official code)
        question_with_tokens = self.image_tokens + ' ' + sample.question
        input_id = self.tokenizer(question_with_tokens, return_tensors='pt')['input_ids'].to(self.device)

        # Generate following M3D's official interface: model.generate(image, input_id, ...)
        with torch.inference_mode():
            generation = self.model.generate(
                image,
                input_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated_text = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0].strip()
        return generated_text


# --- Additional baseline adapters (Qwen2.5-VL, InternVL3) ---------------------
class QwenAdapter(LingshuAdapter):
    """Qwen2.5-VL-7B shares Lingshu's exact arch/inference path (Qwen2_5_VLForConditionalGeneration).
    Vanilla general VLM baseline; reuses the slice -> processor -> generate path unchanged."""
    pass


class InternVLAdapter(FlemingVLAdapter):
    """InternVL3-8B shares FlemingVL's AutoModel + batch_chat + CLIPImageProcessor interface
    (FlemingVL is InternVL-derived). No LoRA system-prompt override for the vanilla baseline."""
    pass


# --- MedEvalKit-backed adapters (LLaVA-Med, HuatuoGPT-Vision) -----------------
MEDEVALKIT_ROOT = os.environ.get(
    "MEDEVALKIT_ROOT", "/volume/med-train/users/shuyan/MedEvalKit"
)


def _ensure_medevalkit_on_path() -> None:
    if MEDEVALKIT_ROOT not in sys.path:
        sys.path.insert(0, MEDEVALKIT_ROOT)


def _extract_closed_letter(text: str) -> Optional[str]:
    s = str(text or '').strip()
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', s, flags=re.I | re.S)
    if answer_match:
        s = answer_match.group(1).strip()
    m = re.search(r'\b([A-E])\b', s) or re.search(r'\(([A-Ea-e])\)', s) or re.match(r'\s*([A-Ea-e])[\).:\s]', s)
    return m.group(1).upper() if m else None


class Med3DVLMAdapter(BaseAdapter):
    """Med3DVLM-Qwen-2.5-7B (config model_type=vlm_qwen, DCFormer 3D vision tower).

    Native 3D model (like M3D): consumes the whole volume, no slicing. The volume is resampled
    to the model's expected (D, H, W) and fed as (B, 1, D, H, W). The prompt is prefixed with
    proj_out_num (256) <im_patch> tokens, matching the official demo. generate(images=, inputs=,
    attention_mask=) is natively batched."""

    IMAGE_PATCH_TOKEN = "<im_patch>"

    def __init__(self, model_path: str, max_new_tokens: int, repetition_penalty: float = 1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()
        cfg = self.model.config
        self.proj_out_num = int(getattr(cfg, 'proj_out_num', 256))
        # config.input_size is [W, H, D] in sitk order; model wants (D, H, W).
        in_size = list(getattr(cfg, 'input_size', [256, 256, 128]))
        self.target_dhw = (int(in_size[2]), int(in_size[1]), int(in_size[0]))
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    def _load_volume(self, npy_path: str):
        import torch.nn.functional as F
        vol = load_volume(npy_path)  # (D, H, W) float
        t = torch.from_numpy(np.ascontiguousarray(vol)).float()
        vmin, vmax = float(t.min()), float(t.max())
        if vmax > vmin:
            t = (t - vmin) / (vmax - vmin)
        t = t[None, None]  # (1,1,D,H,W) for trilinear resample
        t = F.interpolate(t, size=self.target_dhw, mode='trilinear', align_corners=False)
        return t[0]  # (1, D, H, W)

    def _build_prompt(self, sample: BenchmarkSample) -> str:
        question = normalize_question_text(sample.question)
        question = re.sub(r'\n?Answer with the correct option letter and option text only\.?\s*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\n?Answer concisely in a short phrase\.?\s*$', '', question, flags=re.IGNORECASE).rstrip()
        if sample.is_closed:
            question += "\nAnswer with the correct option letter only (A, B, C, or D)."
        return self.IMAGE_PATCH_TOKEN * self.proj_out_num + question

    def _postprocess(self, response: str, is_closed: bool) -> str:
        text = str(response or '').strip()
        if is_closed:
            letter = _extract_closed_letter(text)
            return letter if letter else text
        return text

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        if not samples:
            return [], []
        prompts = [self._build_prompt(s) for s in samples]
        enc = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        images = torch.stack([self._load_volume(s.image_path) for s in samples]).to(
            device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                images=images,
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raws = [t.strip() for t in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        return [self._postprocess(r, s.is_closed) for r, s in zip(raws, samples)], raws

    def generate_answer(self, sample: BenchmarkSample) -> str:
        preds, _ = self.generate_answers_with_raw([sample])
        return preds[0]


class LLaVAMedAdapter(BaseAdapter):
    """Wraps MedEvalKit's LLaVA-Med (llava-med-v1.5-mistral-7b, Mistral-based). 2D model:
    the CT volume is sampled into num_slices cached axial slices, passed as a multi-image prompt.

    Supports batched generation (left-padded input_ids + stacked image tensors). If the batched
    path raises for any reason, it transparently falls back to per-sample generation so results
    are never silently dropped."""

    def __init__(self, model_path: str, max_new_tokens: int, num_slices: int, prompt_style: str,
                 repetition_penalty: float = 1.0):
        from types import SimpleNamespace
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        _ensure_medevalkit_on_path()
        from models.LLava_Med.LLava_Med_hf import LLavaMed
        from models.LLava_Med.mm_utils import tokenizer_image_token, process_images, IMAGE_TOKEN_INDEX
        wrapped_args = SimpleNamespace(
            temperature=0.0, top_p=1.0,
            repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        )
        self.engine = LLavaMed(model_path, wrapped_args)
        self._tokenizer_image_token = tokenizer_image_token
        self._process_images = process_images
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.num_slices = num_slices
        self.prompt_style = prompt_style

    def _build_prompt(self, sample: BenchmarkSample) -> str:
        question = normalize_question_text(sample.question)
        question = re.sub(r'\n?Answer with the correct option letter and option text only\.?\s*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\n?Answer concisely in a short phrase\.?\s*$', '', question, flags=re.IGNORECASE).rstrip()
        prompt = f"{build_ct_slice_description(self.num_slices)}\n{question}"
        if sample.is_closed:
            prompt += "\nAnswer with the correct option letter only (A, B, C, or D)."
        return prompt

    def _postprocess(self, response: str, is_closed: bool) -> str:
        text = str(response or '').strip()
        if is_closed:
            letter = _extract_closed_letter(text)
            return letter if letter else text
        return text

    def _build_image_prompt(self, sample: BenchmarkSample) -> str:
        # Reproduce LLavaMed.process_messages multi-image layout: "<image_i>: <image>\n" * N + question.
        from models.LLava_Med.LLava_Med_hf import DEFAULT_IMAGE_TOKEN
        header = "".join(f"<image_{i+1}>: {DEFAULT_IMAGE_TOKEN}\n" for i in range(self.num_slices))
        return header + self._build_prompt(sample)

    def _generate_single(self, sample: BenchmarkSample) -> str:
        images = load_or_create_cached_slice_images(sample.image_path, num_slices=self.num_slices, image_size=448)
        return self.engine.generate_output({"images": list(images), "prompt": self._build_prompt(sample)})

    def _generate_batch_raw(self, samples: List[BenchmarkSample]) -> List[str]:
        import torch
        from models.LLava_Med.conversation import conv_templates
        model = self.engine.model
        tokenizer = self.engine.tokenizer
        device = next(model.parameters()).device

        input_ids_list, image_tensors = [], []
        for sample in samples:
            conv = conv_templates["mistral_instruct"]
            conv.messages = []
            conv.append_message(conv.roles[0], self._build_image_prompt(sample))
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = self._tokenizer_image_token(prompt, tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_list.append(ids)
            imgs = load_or_create_cached_slice_images(sample.image_path, num_slices=self.num_slices, image_size=448)
            image_tensors.append(self._process_images(list(imgs), self.engine.image_processor, model.config))

        # Left-pad input_ids to a common length so generation aligns at the right edge.
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(x.shape[0] for x in input_ids_list)
        padded, attn = [], []
        for ids in input_ids_list:
            pad_n = max_len - ids.shape[0]
            if pad_n > 0:
                ids = torch.cat([torch.full((pad_n,), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
            attn.append(torch.cat([torch.zeros(pad_n, dtype=torch.long), torch.ones(max_len - pad_n, dtype=torch.long)]))
        input_ids = torch.stack(padded).to(device)
        attention_mask = torch.stack(attn).to(device)
        # Concatenate per-sample image stacks; LLaVA consumes IMAGE_TOKEN_INDEX placeholders in order.
        images = torch.cat(image_tensors, dim=0)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )
        return [t.strip() for t in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    def generate_answer(self, sample: BenchmarkSample) -> str:
        return self._postprocess(self._generate_single(sample), sample.is_closed)

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        if not samples:
            return [], []
        if len(samples) > 1:
            try:
                raws = self._generate_batch_raw(samples)
                if len(raws) == len(samples):
                    return [self._postprocess(r, s.is_closed) for r, s in zip(raws, samples)], raws
            except Exception as exc:  # noqa: BLE001 - fall back to robust per-sample path
                print(f"[LLaVAMedAdapter] batched generate failed ({exc}); falling back to per-sample.")
        raws = [self._generate_single(s) for s in samples]
        return [self._postprocess(r, s.is_closed) for r, s in zip(raws, samples)], raws


class HuatuoAdapter(BaseAdapter):
    """Wraps MedEvalKit's HuatuoChatbot (HuatuoGPT-Vision-7B, LlavaQwen2-based). 2D model.
    HuatuoChatbot caps input at max_image_num (6), so num_slices is clamped to 6 to avoid the
    internal assertion / silent truncation."""

    HUATUO_MAX_IMAGES = 6

    def __init__(self, model_path: str, max_new_tokens: int, num_slices: int, prompt_style: str,
                 repetition_penalty: float = 1.2):
        _ensure_medevalkit_on_path()
        from models.HuatuoGPT.cli import HuatuoChatbot
        self.bot = HuatuoChatbot(model_path)
        self.bot.gen_kwargs['max_new_tokens'] = max_new_tokens
        self.bot.gen_kwargs['repetition_penalty'] = repetition_penalty
        if num_slices > self.HUATUO_MAX_IMAGES:
            print(f"[HuatuoAdapter] num_slices={num_slices} > max_image_num={self.HUATUO_MAX_IMAGES}; clamping to {self.HUATUO_MAX_IMAGES}.")
        self.num_slices = min(num_slices, self.HUATUO_MAX_IMAGES)
        self.prompt_style = prompt_style

    def _build_prompt(self, sample: BenchmarkSample) -> str:
        question = normalize_question_text(sample.question)
        question = re.sub(r'\n?Answer with the correct option letter and option text only\.?\s*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\n?Answer concisely in a short phrase\.?\s*$', '', question, flags=re.IGNORECASE).rstrip()
        prompt = f"{build_ct_slice_description(self.num_slices)}\n{question}"
        if sample.is_closed:
            prompt += "\nAnswer with the correct option letter only (A, B, C, or D)."
        return prompt

    def _postprocess(self, response: str, is_closed: bool) -> str:
        text = str(response or '').strip()
        if is_closed:
            letter = _extract_closed_letter(text)
            return letter if letter else text
        return text

    def _generate_single_raw(self, sample: BenchmarkSample) -> str:
        images = load_or_create_cached_slice_images(sample.image_path, num_slices=self.num_slices, image_size=448)
        answers = self.bot.inference(self._build_prompt(sample), images=list(images))
        return answers[0] if answers else ''

    def _generate_batch_raw(self, samples: List[BenchmarkSample]) -> List[str]:
        import torch
        bot = self.bot
        device = bot.device
        tokenizer = bot.tokenizer

        input_ids_list, image_tensors = [], []
        for sample in samples:
            imgs = load_or_create_cached_slice_images(sample.image_path, num_slices=self.num_slices, image_size=448)
            imgs = list(imgs)[: self.num_slices]
            text = bot.input_moderation(self._build_prompt(sample))
            text = bot.insert_image_placeholder(text, len(imgs))
            conv = bot.get_conv_without_history(text)
            ids = bot.preprocess(conv, return_tensors='pt')
            input_ids_list.append(ids)
            image_tensors.extend(bot.get_image_tensors(imgs))

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(x.shape[0] for x in input_ids_list)
        padded, attn = [], []
        for ids in input_ids_list:
            pad_n = max_len - ids.shape[0]
            if pad_n > 0:
                ids = torch.cat([torch.full((pad_n,), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
            attn.append(torch.cat([torch.zeros(pad_n, dtype=torch.long), torch.ones(max_len - pad_n, dtype=torch.long)]))
        input_ids = torch.stack(padded).to(device)
        attention_mask = torch.stack(attn).to(device)
        images = torch.stack(image_tensors).to(dtype=torch.bfloat16, device=device)

        with torch.inference_mode():
            output_ids = bot.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                use_cache=True,
                **bot.gen_kwargs,
            )
        return [t.strip() for t in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    def generate_answer(self, sample: BenchmarkSample) -> str:
        return self._postprocess(self._generate_single_raw(sample), sample.is_closed)

    def generate_answers_with_raw(self, samples: List[BenchmarkSample]):
        if not samples:
            return [], []
        if len(samples) > 1:
            try:
                raws = self._generate_batch_raw(samples)
                if len(raws) == len(samples):
                    return [self._postprocess(r, s.is_closed) for r, s in zip(raws, samples)], raws
            except Exception as exc:  # noqa: BLE001 - fall back to robust per-sample path
                print(f"[HuatuoAdapter] batched generate failed ({exc}); falling back to per-sample.")
        raws = [self._generate_single_raw(s) for s in samples]
        return [self._postprocess(r, s.is_closed) for r, s in zip(raws, samples)], raws


def compute_bleu1(prediction: str, reference: str) -> float:
    return sentence_bleu(prediction, [reference], smooth_method='exp', use_effective_order=True).score / 100.0


def compute_rouge1(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(reference, prediction)['rouge1'].fmeasure


def compute_meteor(prediction: str, reference: str) -> float:
    return meteor_score([reference.split()], prediction.split())


def compute_bert_f1(prediction: str, reference: str, model_name_or_path: str, num_layers: int) -> float:
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        _, _, f1 = bert_score(
            [prediction],
            [reference],
            model_type=model_name_or_path,
            num_layers=num_layers,
            lang='en',
            verbose=False,
            rescale_with_baseline=False,
        )
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    return float(f1.mean().item())


def is_correct_closed_answer(prediction: str, answer_choice: Optional[str]) -> int:
    if answer_choice is None:
        return 0
    match = re.match(r'^\s*([A-E])(?:\.|\b)', prediction.strip(), flags=re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1).upper() == answer_choice.strip().upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate 3D-RAD with multiple model adapters.')
    parser.add_argument('--model_type', choices=['m3d', 'flemingvl', 'lingshu', 'qwen', 'internvl', 'llava_med', 'huatuo', 'med3dvlm'], required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--vqa_data_test_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--close_ended', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='>1.0 惩罚重复 token,抑制车轱辘话/跑飞导致的超长输出。1.0=关闭。')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_slices', type=int, default=11)
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--lora_path', type=str, default='')
    parser.add_argument('--prompt_style', choices=['reasoning', 'plain'], default='reasoning')
    parser.add_argument('--bertscore_model', type=str, default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument('--bertscore_num_layers', type=int, default=DEFAULT_BERTSCORE_LAYERS)
    return parser.parse_args()


def build_adapter(args: argparse.Namespace) -> BaseAdapter:
    if args.model_type == 'flemingvl':
        return FlemingVLAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            use_flash_attn=args.use_flash_attn,
            prompt_style=args.prompt_style,
            lora_path=args.lora_path or None,
            repetition_penalty=args.repetition_penalty,
        )
    if args.model_type == 'lingshu':
        return LingshuAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            use_flash_attn=args.use_flash_attn,
            prompt_style=args.prompt_style,
            lora_path=args.lora_path or None,
            repetition_penalty=args.repetition_penalty,
        )
    if args.model_type == 'qwen':
        return QwenAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            use_flash_attn=args.use_flash_attn,
            prompt_style=args.prompt_style,
            lora_path=args.lora_path or None,
            repetition_penalty=args.repetition_penalty,
        )
    if args.model_type == 'internvl':
        return InternVLAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            use_flash_attn=args.use_flash_attn,
            prompt_style=args.prompt_style,
            lora_path=args.lora_path or None,
            repetition_penalty=args.repetition_penalty,
        )
    if args.model_type == 'llava_med':
        return LLaVAMedAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            prompt_style=args.prompt_style,
            repetition_penalty=args.repetition_penalty,
        )
    if args.model_type == 'huatuo':
        return HuatuoAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_slices=args.num_slices,
            prompt_style=args.prompt_style,
            repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1.0 else 1.2,
        )
    if args.model_type == 'med3dvlm':
        return Med3DVLMAdapter(
            model_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
    return M3DAdapter(
        model_path=args.model_name_or_path,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )


def main() -> None:
    seed_everything(42)
    args = parse_args()
    dataset = BenchmarkDataset(args.data_root, args.vqa_data_test_path, args.close_ended)
    if args.max_samples and args.max_samples > 0:
        dataset.data_list = dataset.data_list.iloc[:args.max_samples].reset_index(drop=True)

    adapter = build_adapter(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Qwen2.5-VL has dynamic visual token count per image; batching multiple CT volumes
    # (each with 8 slices) causes unpredictable token explosion and OOM.  Force bs=1.
    # Exception: when THREED_RAD_MAX_PIXELS is set (<=65536), token count is bounded
    # and batching is safe.  The restart script sets MAX_PIXELS=256*256=65536.
    if args.model_type == 'lingshu':
        effective_batch_size = 1
    else:
        effective_batch_size = args.batch_size if args.model_type in {'flemingvl', 'internvl', 'llava_med', 'huatuo', 'med3dvlm', 'qwen'} else 1

    if args.close_ended:
        output_path = os.path.join(args.output_dir, 'eval_close_vqa.csv')
        raw_output_path = os.path.join(args.output_dir, 'eval_close_vqa_raw.csv')
        with open(output_path, 'w', newline='') as outfile, open(raw_output_path, 'w', newline='') as rawfile:
            writer = csv.writer(outfile)
            raw_writer = csv.writer(rawfile)
            writer.writerow(['Question Aspect', 'Question', 'Answer', 'Answer Choice', 'Pred', 'Correct'])
            raw_writer.writerow(['Question Aspect', 'Question', 'Answer', 'Answer Choice', 'Pred', 'RawPred', 'Correct'])
            total_batches = (len(dataset) + effective_batch_size - 1) // effective_batch_size
            progress = tqdm(range(0, len(dataset), effective_batch_size), total=total_batches, desc=os.path.basename(args.output_dir), dynamic_ncols=True)
            for start in progress:
                samples = [dataset[idx] for idx in range(start, min(start + effective_batch_size, len(dataset)))]
                preds, raw_preds = adapter.generate_answers_with_raw(samples)
                progress.set_postfix(done=min(start + len(samples), len(dataset)), total=len(dataset))
                for sample, pred, raw_pred in zip(samples, preds, raw_preds):
                    correct = is_correct_closed_answer(pred, sample.answer_choice)
                    writer.writerow([[sample.question_aspect], sample.question, sample.answer, sample.answer_choice, pred, correct])
                    raw_writer.writerow([[sample.question_aspect], sample.question, sample.answer, sample.answer_choice, pred, raw_pred, correct])
    else:
        output_path = os.path.join(args.output_dir, 'eval_open_vqa.csv')
        raw_output_path = os.path.join(args.output_dir, 'eval_open_vqa_raw.csv')
        with open(output_path, 'w', newline='') as outfile, open(raw_output_path, 'w', newline='') as rawfile:
            writer = csv.writer(outfile)
            raw_writer = csv.writer(rawfile)
            writer.writerow(['Question Aspect', 'Question', 'Answer', 'Pred', 'bleu', 'rouge1', 'meteor', 'bert_f1'])
            raw_writer.writerow(['Question Aspect', 'Question', 'Answer', 'Pred', 'RawPred', 'bleu', 'rouge1', 'meteor', 'bert_f1'])
            total_batches = (len(dataset) + effective_batch_size - 1) // effective_batch_size
            progress = tqdm(range(0, len(dataset), effective_batch_size), total=total_batches, desc=os.path.basename(args.output_dir), dynamic_ncols=True)
            for start in progress:
                samples = [dataset[idx] for idx in range(start, min(start + effective_batch_size, len(dataset)))]
                preds, raw_preds = adapter.generate_answers_with_raw(samples)
                progress.set_postfix(done=min(start + len(samples), len(dataset)), total=len(dataset))
                for sample, pred, raw_pred in zip(samples, preds, raw_preds):
                    bleu = compute_bleu1(pred, sample.answer)
                    rouge1 = compute_rouge1(pred, sample.answer)
                    meteor = compute_meteor(pred, sample.answer)
                    bert_f1 = compute_bert_f1(pred, sample.answer, args.bertscore_model, args.bertscore_num_layers)
                    writer.writerow([
                        [sample.question_aspect],
                        sample.question,
                        sample.answer,
                        pred,
                        bleu,
                        rouge1,
                        meteor,
                        bert_f1,
                    ])
                    raw_writer.writerow([
                        [sample.question_aspect],
                        sample.question,
                        sample.answer,
                        pred,
                        raw_pred,
                        bleu,
                        rouge1,
                        meteor,
                        bert_f1,
                    ])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
inference_compare_fast.py — Ultra-Fast Multi-Task Inference 
(DataLoader Prefetching + IO Deduplication + Flash Attention + CT-RATE GT Cleaner + Complexity Routing + Python Native Multi-Processing)

脚本职责概览：
1. 读取评测 JSON/JSONL，并把多轮对话展开成逐条可推理的问题样本。
2. 为每个样本加载 2D/3D 医学图像，抽取固定数量的切片并转成模型输入。
3. 通过多进程把数据分发到多张 GPU，每个子进程独立加载模型并执行推理。
4. 推理时先做“问题复杂度路由”，再按路由结果拼接不同层次的诊断提示词。
5. 最终把不同 task_type 的结果分别写成独立 jsonl 文件，便于后续评估。
"""

import argparse
import contextlib
import gc
import json
import os
import sys
import re
import glob
import string
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from pathlib import Path
from gpu_wait import parse_gpu_ids, wait_for_gpus
from prompt_contracts import (
    CURRENT_INFERENCE_COT_PROMPT,
    build_ct_slice_description,
    normalize_question_text,
)
from med3d_image_utils import get_pixel_values as shared_get_pixel_values, resolve_med_image_path as shared_resolve_med_image_path

# ===========================================================================
# 解决多进程可能导致的数据死锁问题
# ===========================================================================
# 告诉 OS 使用 spawn 来生成“全新解释器”子进程，避免：
# 1. 父进程已经初始化 CUDA 后再 fork，导致上下文继承异常；
# 2. DataLoader worker / 模型推理子进程之间出现死锁或资源竞争；
# 3. 多卡环境下不同进程错误共享句柄。
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 运行环境里如果 start method 已经提前设置过，会抛 RuntimeError。
    # 这里直接忽略即可，不影响后续执行。
    pass

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-Task 3D Medical Model Evaluation (Multi-Processing)")
parser.add_argument('--model_path',      type=str, required=True, help='Path to base model')
parser.add_argument('--data_path',       type=str, required=True, help='Path to test JSON')
parser.add_argument('--eval_baseline',   action='store_true', default=False, help='Evaluate base model')
parser.add_argument('--trained_weights', nargs='+', type=str, default=None, help='Paths to LoRA adapters')
parser.add_argument('--model_names',     nargs='+', type=str, default=None, help='Custom names for models')
parser.add_argument('--num_slices',      type=int, default=8, help='Slices to extract from 3D CT')
parser.add_argument('--num_samples',     type=int, default=-1, help='Samples to evaluate (-1 for all)')
parser.add_argument('--batch_size',      type=int, default=32, help='H200 上建议从 32 起步，按显存回退')
parser.add_argument('--num_workers',     type=int, default=8, help='DataLoader 进程数 (建议 8)')
parser.add_argument('--use_flash_attn',  action='store_true', default=False, help='开启 Flash Attention')
parser.add_argument('--max_new_tokens',  type=int, default=1024)
parser.add_argument('--repetition_penalty', type=float, default=1.1,
                    help='重复惩罚系数；>1 会抑制重复生成')
parser.add_argument('--no_repeat_ngram_size', type=int, default=6,
                    help='禁止重复的 ngram 长度；0 表示关闭')
parser.add_argument('--output_path',     type=str, default='./compare_results.json', help='Base path for output')
parser.add_argument('--gpu_ids',         type=str, default='0,1,2,3', help='要使用的所有GPU，用逗号分隔')
parser.add_argument('--no_cuda',         action='store_true', default=False)
parser.add_argument('--dtype',           type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--image_base_dir',  type=str, required=True, help='Path to valid_fixed')
parser.add_argument('--wait_for_gpus', action='store_true', default=False)
parser.add_argument('--gpu_poll_interval_sec', type=int, default=60)
parser.add_argument('--gpu_max_used_memory_mb', type=int, default=4096)
parser.add_argument('--gpu_max_utilization', type=int, default=20)
parser.add_argument('--debug_print_every', type=int, default=100,
                    help='每隔多少条推理样本打印一次输入文本（不含图像 token）；<=0 关闭')
parser.add_argument('--skip_auto_metric', action='store_true', default=False,
                    help='Skip automatic metric computation after result merge')
parser.add_argument('--adapter_sanity_check', action='store_true', default=False,
                    help='在首个 batch 上比较 baseline/LoRA 的 logits 与短生成，确认 adapter 是否真的生效')
parser.add_argument('--adapter_sanity_max_new_tokens', type=int, default=16,
                    help='adapter 生效性检查时额外生成的最大 token 数')
parser.add_argument('--force_black_image', action='store_true', default=False,
                    help='忽略真实图像，直接用全黑占位图做推理，用于文本先验/黑图对照实验')
parser.add_argument('--shuffle_images', action='store_true', default=False,
                    help='在评测前随机打乱样本与图像的对应关系，用于检验模型是否真正依赖视觉输入')
parser.add_argument('--shuffle_seed', type=int, default=42,
                    help='shuffle_images 模式下使用的随机种子')
args = parser.parse_args()

if args.output_path:
    output_parent = os.path.dirname(args.output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

# ---------------------------------------------------------------------------
# 工具函数 (CT-RATE 净化器、图像加载器、Prompt 构建器)
# ---------------------------------------------------------------------------



def resolve_med_image_path(img_path):
    return shared_resolve_med_image_path(img_path)


def get_pixel_values(img_data, num_slices, image_processor):
    return shared_get_pixel_values(img_data, num_slices, image_processor)

def infer_task_type(question, explicit_type=None):
    if explicit_type and explicit_type != 'general_qa':
        return explicit_type
    text = str(question)
    if '<multiple_choice>' in text:
        return 'multiple_choice'
    if re.search(r'Choice\s*A\s*:', text, re.IGNORECASE):
        return 'multiple_choice'
    if re.search(r'\([a-d]\)', text, re.IGNORECASE):
        return 'multiple_choice'
    return explicit_type or 'general_qa'

def parse_ct_rate_ground_truth(question, gt, task_type):
    if task_type != 'multiple_choice' and '<multiple_choice>' not in question:
        return gt
    gt_str = str(gt).strip()

    m = re.match(r'^\s*\(([a-d])\)', gt_str, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if '(a)' in gt_str and '(b)' in gt_str:
        answer_text = gt_str.split('(a)')[0].strip().lower()
        options = {}
        for opt in ['a', 'b', 'c', 'd']:
            match = re.search(rf'\({opt}\)\s*(.*?)(?=\([a-d]\)|$)', gt_str, re.IGNORECASE)
            if match:
                options[opt] = match.group(1).strip().lower()

        def clean_text(text):
            return text.translate(str.maketrans('', '', string.punctuation)).strip()

        ans_clean = clean_text(answer_text)
        for opt, text in options.items():
            cleaned = clean_text(text)
            if cleaned and (cleaned == ans_clean or cleaned in ans_clean or ans_clean in cleaned):
                return opt.upper()

        if 'no dilatation' in ans_clean:
            for opt, text in options.items():
                if 'normal' in text:
                    return opt.upper()

        stop_words = {'is', 'the', 'in', 'was', 'were', 'observed', 'detected', 'a', 'an', 'of', 'to', 'for'}
        ans_words = set(ans_clean.split()) - stop_words
        best_opt, max_overlap = None, 0
        for opt, text in options.items():
            text_words = set(clean_text(text).split()) - stop_words
            overlap = len(ans_words & text_words)
            if ('no' in ans_words or 'not' in ans_words) != ('no' in text_words or 'not' in text_words):
                overlap -= 2
            if overlap > max_overlap:
                max_overlap, best_opt = overlap, opt
        if best_opt and max_overlap > 0:
            return best_opt.upper()

    if '(a)' in question and '(b)' in question:
        answer_text = gt_str.lower()
        options = {}
        for opt in ['a', 'b', 'c', 'd']:
            match = re.search(rf'\({opt}\)\s*(.*?)(?=\([a-d]\)|<multiple_choice>|$)', question, re.IGNORECASE)
            if match:
                options[opt] = match.group(1).strip().lower()
        ans_clean = answer_text.translate(str.maketrans('', '', string.punctuation)).strip()
        for opt, text in options.items():
            cleaned = text.translate(str.maketrans('', '', string.punctuation)).strip()
            if cleaned and (cleaned == ans_clean or cleaned in ans_clean or ans_clean in cleaned):
                return opt.upper()

    return gt


class Med3DInferenceDataset(Dataset):
    """仅负责把 query 元信息和对应图像张量打包给 DataLoader。"""
    def __init__(self, queries, num_slices, image_processor, force_black_image=False):
        self.queries = queries
        self.num_slices = num_slices
        self.image_processor = image_processor
        self.force_black_image = force_black_image

    def __len__(self): return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        if self.force_black_image:
            img = Image.new('RGB', (448, 448))
            pv = self.image_processor(images=img, return_tensors='pt').pixel_values[0]
            pixel_values = torch.stack([pv] * self.num_slices, dim=0)
            return {'query': q, 'pixel_values': pixel_values}
        try:
            # 正常路径：读取真实图像并提取切片。
            pixel_values = get_pixel_values(q['image'], self.num_slices, self.image_processor)
        except Exception as e:
            # 容错路径：图像损坏或路径错误时，返回全黑占位图，保证推理流程不中断。
            print(f"Error loading {q['image']}: {e}")
            img = Image.new('RGB', (448, 448))
            pv = self.image_processor(images=img, return_tensors='pt').pixel_values[0]
            pixel_values = torch.stack([pv] * self.num_slices, dim=0)
        return {'query': q, 'pixel_values': pixel_values}

def collate_fn(batch):
    # queries 保持为 Python 列表，方便后面逐条取问题/GT；
    # 图像张量在 batch 维拼接，形成 [B * num_slices, C, H, W]。
    return [b['query'] for b in batch], torch.cat([b['pixel_values'] for b in batch], dim=0)

def extract_complexity_tag(pred_text):
    first_line = ""
    think_match = re.search(r'<think>\s*(.*?)\s*(?:\n|</think>)', pred_text, flags=re.IGNORECASE | re.DOTALL)
    if think_match:
        think_body = think_match.group(1).strip()
        think_lines = [line.strip() for line in think_body.splitlines() if line.strip()]
        if think_lines:
            first_line = think_lines[0]
    else:
        stripped = pred_text.strip()
        if stripped:
            stripped_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            if stripped_lines:
                first_line = stripped_lines[0]

    first_line_lower = first_line.lower()
    if "rapid triage" in first_line_lower:
        return "Rapid Triage"
    if "standard assessment" in first_line_lower:
        return "Standard Assessment"
    if "comprehensive diagnosis" in first_line_lower:
        return "Comprehensive Diagnosis"
    return "Unknown"

def build_single_stage_prompt(task_type, current_question, num_slices, token_count):
    """
    组装 InternVL 风格 prompt。

    使用统一的推理期 prompt contract，强制模型先输出
    `dimension + question type`，再按复杂度补足 4/5/7 行思维链。
    """
    # 每个切片对应一段 `<img> ... </img>` 占位符，其中 IMG_CONTEXT 数量
    # 必须和视觉编码器展开后的 token 数一致，否则图文对齐会错位。
    img_tokens = ''.join('<img>' + '<IMG_CONTEXT>' * token_count + '</img>' for _ in range(num_slices))
    forced_cot_prefix = '<think>\n'
    forced_cot_prefix = ''
    image_description = build_ct_slice_description(num_slices)
    prompt = f'<|im_start|>system\n{CURRENT_INFERENCE_COT_PROMPT}<|im_end|>\n'
    prompt += (
        f'<|im_start|>user\n{image_description}\n{img_tokens}\n{current_question}<|im_end|>\n'
        f'<|im_start|>assistant\n{forced_cot_prefix}'
    )
    return prompt


def run_adapter_sanity_check(model, tokenizer, pixel_values_full, queries, eval_tasks, args, gpu_id, token_count, device):
    """
    用同一条样本分别跑 baseline 和第一个 LoRA，比较首步 logits 与短生成。

    目的不是衡量效果，而是尽早发现“LoRA 权重已加载，但开关切换对前向完全没影响”的情况。
    """
    if not args.adapter_sanity_check or not args.eval_baseline:
        return

    lora_tasks = [task for task in eval_tasks if not task['is_baseline']]
    if not lora_tasks:
        return

    baseline_task = next((task for task in eval_tasks if task['is_baseline']), None)
    if baseline_task is None:
        return

    question = queries[0]['question']
    prompt = build_single_stage_prompt(queries[0]['type'], question, args.num_slices, token_count)
    enc = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True).to(device)
    sample_pvs = pixel_values_full[:args.num_slices]
    lora_task = lora_tasks[0]

    active_adapters = []
    if hasattr(model.language_model, 'peft_config'):
        active_adapters = sorted(model.language_model.peft_config.keys())
    print(
        f"[Adapter Check] gpu={gpu_id} loaded_adapters={active_adapters} "
        f"baseline={baseline_task['name']} lora={lora_task['name']}"
    )

    with torch.inference_mode():
        with model.language_model.disable_adapter():
            baseline_out = model(
                **enc,
                pixel_values=sample_pvs,
                return_dict=True,
            )
            baseline_gen = model.generate(
                **enc,
                pixel_values=sample_pvs,
                max_new_tokens=args.adapter_sanity_max_new_tokens,
                temperature=0.0,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        model.language_model.set_adapter(lora_task['name'])
        lora_out = model(
            **enc,
            pixel_values=sample_pvs,
            return_dict=True,
        )
        lora_gen = model.generate(
            **enc,
            pixel_values=sample_pvs,
            max_new_tokens=args.adapter_sanity_max_new_tokens,
            temperature=0.0,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    baseline_last = baseline_out.logits[0, -1].float().cpu()
    lora_last = lora_out.logits[0, -1].float().cpu()
    diff = (baseline_last - lora_last).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    topk = min(5, baseline_last.numel())
    baseline_top = torch.topk(baseline_last, k=topk).indices.tolist()
    lora_top = torch.topk(lora_last, k=topk).indices.tolist()
    input_seq_len = enc['input_ids'].shape[1]
    baseline_text = tokenizer.decode(baseline_gen[0][input_seq_len:], skip_special_tokens=True).strip()
    lora_text = tokenizer.decode(lora_gen[0][input_seq_len:], skip_special_tokens=True).strip()

    print(
        f"[Adapter Check] gpu={gpu_id} question={question[:160]!r}\n"
        f"  max_abs_diff={max_abs_diff:.8f} mean_abs_diff={mean_abs_diff:.8f}\n"
        f"  baseline_top_ids={baseline_top}\n"
        f"  lora_top_ids={lora_top}\n"
        f"  baseline_text={baseline_text!r}\n"
        f"  lora_text={lora_text!r}"
    )
    if max_abs_diff == 0.0 and baseline_text == lora_text:
        print(
            f"[Adapter Check][WARNING] gpu={gpu_id} baseline and LoRA are numerically identical "
            f"on the sanity sample. Adapter switching may be ineffective."
        )

# ===========================================================================
# 核心工作函数 (子进程实际跑的代码)
# ===========================================================================
def worker_process(rank, gpu_id, data_chunk, args, temp_output_path):
    """
    子进程主函数。

    一个子进程只绑定一张 GPU，并独立完成：
    1. 模型加载；
    2. 本 GPU 对应数据块的 DataLoader 构建；
    3. 单阶段推理；
    4. 中间结果落盘到本 GPU 专属临时文件。
    """
    # 由于每个子进程是独立解释器，这里设置 CUDA_VISIBLE_DEVICES 可以把它“隔离”到单卡视角。
    # 进程内部始终只看到自己的第 0 张卡，因此后面统一用 cuda:0 即可。
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    print(f"🚀 进程 {rank} 启动！绑定物理 GPU {gpu_id}, 负责处理 {len(data_chunk)} 条数据...")

    # ---------- 加载模型 (每个进程都拥有自己的模型副本，互不共享显存) ----------
    model_dir = os.path.realpath(args.model_path)
    sys.path.insert(0, os.path.dirname(model_dir))
    pkg_name = os.path.basename(model_dir)

    import importlib
    InternVLChatConfig = importlib.import_module(f'{pkg_name}.configuration_internvl_chat').InternVLChatConfig
    InternVLChatModel  = importlib.import_module(f'{pkg_name}.modeling_internvl_chat').InternVLChatModel

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = InternVLChatConfig.from_pretrained(args.model_path)
    # 当前脚本走“纯基座 + 可选 LoRA 适配器”的装配方式，不使用原仓库中的 adapter/3d projector 开关。
    config.use_adapter, config.use_3d_projector = False, False

    from safetensors.torch import load_file
    # 模型权重被切成多个 shard，这里先在 CPU 合并，再整体 load 到模型里。
    with open(os.path.join(args.model_path, 'model.safetensors.index.json')) as f:
        shard_files = sorted(set(json.load(f)['weight_map'].values()))
    state_dict = {}
    for shard in shard_files:
        state_dict.update(load_file(os.path.join(args.model_path, shard), device='cpu'))

    model = InternVLChatModel(config, use_flash_attn=args.use_flash_attn).to(dtype)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    # eval_tasks 允许同一轮评测同时比较 baseline 和多个 LoRA 适配器。
    eval_tasks = []
    if args.eval_baseline: eval_tasks.append({'name': 'baseline', 'is_baseline': True})
    if args.trained_weights:
        names = args.model_names or [os.path.basename(os.path.normpath(w)) for w in args.trained_weights]
        for w, n in zip(args.trained_weights, names):
            eval_tasks.append({'name': n, 'is_baseline': False, 'weight_path': w})

    if args.trained_weights:
        from peft import PeftModel
        loras = [t for t in eval_tasks if not t['is_baseline']]
        # 第一个 LoRA 用 from_pretrained 包装 language_model，后续 LoRA 作为额外 adapter 动态挂载。
        model.language_model = PeftModel.from_pretrained(model.language_model, loras[0]['weight_path'], adapter_name=loras[0]['name'])
        for t in loras[1:]: model.language_model.load_adapter(t['weight_path'], adapter_name=t['name'])

    model = model.to(device=device).eval()
    # InternVL 约定的图像上下文 token id，用于把视觉 token 插入文本序列。
    model.img_context_token_id = 151667
    TOKENS_PER_SLICE = model.num_image_token

    # ---------- 加载数据并推理 ----------
    dataset = Med3DInferenceDataset(
        data_chunk,
        args.num_slices,
        image_processor,
        force_black_image=args.force_black_image,
    )
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'collate_fn': collate_fn,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 4
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    # 左侧 padding 更适配自回归生成，便于不同长度 prompt 组成 batch。
    tokenizer.padding_side = 'left'
    debug_print_every = max(0, args.debug_print_every)
    debug_counter = 0
    adapter_sanity_done = False

    # 给进度条加个前缀，方便区分是哪个进程
    pbar = tqdm(dataloader, desc=f"[GPU {gpu_id}] Infer", position=rank, leave=True)

    for queries, pixel_values_full in pbar:
        # 为了公平比较 baseline 与 LoRA 微调模型，二者都使用相同的多切片视觉输入。
        # 唯一差异应当是 language model 是否启用 adapter，而不是图像切片数量不同。
        pixel_values_full = pixel_values_full.to(device=device, dtype=dtype)
        if not adapter_sanity_done and rank == 0:
            run_adapter_sanity_check(
                model=model,
                tokenizer=tokenizer,
                pixel_values_full=pixel_values_full,
                queries=queries,
                eval_tasks=eval_tasks,
                args=args,
                gpu_id=gpu_id,
                token_count=TOKENS_PER_SLICE,
                device=device,
            )
            adapter_sanity_done = True

        for task in eval_tasks:
            model_name, is_base = task['name'], task['is_baseline']
            
            if not is_base:
                # 切换到目标 LoRA adapter；同一个模型实例在多 adapter 间复用。
                model.language_model.set_adapter(model_name)
                
            # baseline 评测时需要临时禁用 adapter，确保测到的是原始基座能力。
            cm = model.language_model.disable_adapter() if is_base and hasattr(model.language_model, 'disable_adapter') else contextlib.nullcontext()
            
            with cm, torch.inference_mode():
                # 直接构造最简 user -> assistant 续写格式。
                prompts = []
                num_s = args.num_slices
                for q in queries:
                    if debug_print_every > 0 and debug_counter % debug_print_every == 0:
                        debug_preview = (
                            f'<|im_start|>user\n{q["question"]}<|im_end|>\n'
                            f'<|im_start|>assistant\n'
                        )
                        print(
                            f"\n[Inference Debug] gpu={gpu_id} sample_idx={debug_counter} "
                            f"model={model_name}\n{debug_preview}"
                        )
                    prompt = build_single_stage_prompt(q['type'], q['question'], num_s, TOKENS_PER_SLICE)
                    prompts.append(prompt)
                    debug_counter += 1
                
                enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
                # baseline 与微调模型共享完全相同的视觉输入，避免因输入不一致而影响比较结论。
                pvs = pixel_values_full
                
                outputs = model.generate(
                    **enc,
                    pixel_values=pvs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 保存当前批次结果到临时文件
            input_seq_len = enc['input_ids'].shape[1]
            forced_cot_prefix = '<think>\nThis is a '
            for b_idx, out in enumerate(outputs):
                # generate() 返回的是“完整输入 + 新生成 token”，在 batch + left padding 情况下，
                # 应按统一的 padded input 长度切，而不是按每条样本的有效 token 数切。
                generated_ids = out[input_seq_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # 兜底：如果新生成部分被切空，回退到整段解码再基于 assistant 起点截取，
                # 这样至少不会只剩下前缀。
                if not generated_text:
                    full_text = tokenizer.decode(out, skip_special_tokens=True)
                    assistant_text = full_text.split('assistant\n')[-1].strip()
                    if assistant_text.startswith(forced_cot_prefix):
                        text = assistant_text
                    elif assistant_text.startswith('<think>'):
                        text = assistant_text
                    else:
                        text = f'{forced_cot_prefix}{assistant_text}'.strip()
                else:
                    text = f'{forced_cot_prefix}{generated_text}'.strip()

                # 某些回答可能没有正常闭合或遗漏 <think> 起始，统一做兜底包装。
                if not text.startswith('<think>'):
                    text = '<think>\n' + text
                complexity_tag = extract_complexity_tag(text)
                
                task_type = queries[b_idx]['type']
                res_dict = {
                    "id": queries[b_idx]['id'], 
                    "image": queries[b_idx]['image'], 
                    "question": queries[b_idx]['question'], 
                    "ground_truth": queries[b_idx]['ground_truth'], 
                    "complexity_tag": complexity_tag,
                    "model_name": model_name, 
                    "task_type": task_type,
                    "prediction": text
                }
                with open(temp_output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res_dict, ensure_ascii=False) + '\n')

    print(f"✅ 进程 {rank} (GPU {gpu_id}) 任务圆满完成！")


# ===========================================================================
# 自动评测汇总
# ===========================================================================
def run_auto_metrics(output_files, base_out_path):
    if args.skip_auto_metric:
        print('ℹ️ 已跳过自动指标计算 (--skip_auto_metric)。')
        return

    try:
        from get_metric7 import generate_report_bundle
    except Exception as e:
        print(f'⚠️ 自动指标计算已跳过：无法导入 get_metric7.py: {e}')
        return

    written = 0

    for output_file in output_files:
        try:
            bundle = generate_report_bundle(output_file, experiment_name=Path(output_file).stem)
            print(f"📊 指标已写入: {bundle['summary_json']}")
            print(f"🧾 逐条评测已写入: {bundle['details_jsonl']}")
            print(f"❌ 错误样本已写入: {bundle['errors_jsonl']}")
            written += 1
        except Exception as e:
            print(f'⚠️ 自动指标计算失败 {output_file}: {e}')

    if written and output_files:
        summary_jsonl = str(Path(output_files[0]).parent / 'metrics_summary.jsonl')
        print(f'📘 汇总指标已追加到: {summary_jsonl}')

# ===========================================================================
# 主进程入口 (负责分发数据、拉起子进程、汇总结果)
# ===========================================================================
def main():
    # ---------- 全局数据准备 ----------
    # 建立“文件名 -> 完整路径”的索引，兼容数据集中只给 basename 的情况。
    image_path_index = {}
    for root, dirs, files in os.walk(args.image_base_dir):
        for f in files:
            if f.endswith('.nii.gz') or f.endswith('.npy'): image_path_index[f] = os.path.join(root, f)

    # 同时兼容 JSON 和 JSONL 两种输入格式。
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f) if args.data_path.endswith('.json') else [json.loads(line) for line in f if line.strip()]

    # 固定随机种子，保证抽样结果可复现。
    random.seed(42)
    if args.num_samples > 0:
        raw_data = random.sample(raw_data, min(args.num_samples, len(raw_data)))
        
    flat_queries = []
    for sample in raw_data:
        img_data = sample.get('image', '')
        # 优先保留原始 list；如果是字符串路径，则尝试用索引补全成绝对路径。
        full_image_path = img_data if isinstance(img_data, list) else (image_path_index.get(img_data) or image_path_index.get(os.path.basename(img_data)) or img_data)
        convs = sample.get('conversations', [])

        # 数据集中常见格式是 human / gpt 交替，这里按两两配对展开为单条 query。
        for i in range(0, len(convs), 2):
            if convs[i]['from'] == 'human':
                raw_question = normalize_question_text(convs[i]['value'])
                q_type = infer_task_type(raw_question, convs[i].get('type', 'general_qa'))
                raw_gt = convs[i+1]['value'].strip() if i+1 < len(convs) else ""
                flat_queries.append({
                    'id': sample.get('id', 'unknown'),
                    'image': full_image_path,
                    'type': q_type,
                    'question': raw_question,
                    'ground_truth': parse_ct_rate_ground_truth(raw_question, raw_gt, q_type),
                    'dimension': sample.get('dimension'),
                    'question_type': sample.get('question_type'),
                })

    if args.shuffle_images and flat_queries:
        shuffled_images = [q['image'] for q in flat_queries]
        random.Random(args.shuffle_seed).shuffle(shuffled_images)
        for q, shuffled_image in zip(flat_queries, shuffled_images):
            q['image'] = shuffled_image
        print(f"[Eval Mode] shuffle_images enabled with seed={args.shuffle_seed}; reassigned {len(flat_queries)} image bindings.")

    # ---------- 数据均分给各个 GPU ----------
    gpu_list = parse_gpu_ids(args.gpu_ids)
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        raise ValueError("No GPU ids provided. Please set --gpu_ids.")

    if flat_queries:
        per_gpu = len(flat_queries) // num_gpus + (1 if len(flat_queries) % num_gpus else 0)
        print(
            f"[Eval Mode] total_queries={len(flat_queries)} per_gpu~{per_gpu} "
            f"batch_size={args.batch_size} num_workers={args.num_workers}"
        )

    if args.wait_for_gpus and not args.no_cuda:
        wait_for_gpus(
            gpu_ids=",".join(gpu_list),
            max_used_memory_mb=args.gpu_max_used_memory_mb,
            max_utilization=args.gpu_max_utilization,
            poll_interval_sec=args.gpu_poll_interval_sec,
            label="inference_compare",
        )
    print(f"🔥 检测到 {num_gpus} 张 GPU，即将切分数据并拉起多进程...")

    # 利用步长切片把样本近似均匀地轮转分给各卡：
    # 第 0 卡拿 0, N, 2N...；第 1 卡拿 1, N+1, 2N+1...
    # 这样实现简单，且通常能避免某一卡集中拿到一大段重样本。
    chunks = [flat_queries[i::num_gpus] for i in range(num_gpus)]
    
    # 准备临时结果文件路径
    base_out_path = args.output_path.replace('.json', '').replace('.jsonl', '')
    temp_files = [f"{base_out_path}_temp_gpu{g}.jsonl" for g in gpu_list]
    # 清空旧的临时文件
    for t_file in temp_files:
        if os.path.exists(t_file): os.remove(t_file)
    out_dir = os.path.dirname(base_out_path) or "."
    out_prefix = os.path.basename(base_out_path) + "_"
    for file_name in os.listdir(out_dir):
        if file_name.startswith(out_prefix) and file_name.endswith(".jsonl"):
            file_path = os.path.join(out_dir, file_name)
            if file_path not in temp_files:
                os.remove(file_path)

    # ---------- 拉起多进程 ----------
    processes = []
    for rank in range(num_gpus):
        gpu_id = gpu_list[rank]
        data_chunk = chunks[rank]
        temp_output_path = temp_files[rank]
        
        # 每个子进程只关心自己的 GPU id、数据分片和临时输出路径。
        p = mp.Process(target=worker_process, args=(rank, gpu_id, data_chunk, args, temp_output_path))
        p.start()
        processes.append(p)

    # 主进程挂起，等待所有子进程结束
    for p in processes:
        p.join()

    # ---------- 结果合并 ----------
    print(f"\n🔄 所有后台进程已退出，正在合并最终结果...")
    merged_output_files = set()
    for t_file in temp_files:
        if not os.path.exists(t_file): continue
        with open(t_file, 'r', encoding='utf-8') as f:
            for line in f:
                res_dict = json.loads(line)
                task_type = res_dict.pop("task_type")
                out_file = f"{base_out_path}_{task_type}.jsonl"
                merged_output_files.add(out_file)
                with open(out_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(res_dict, ensure_ascii=False) + '\n')
        os.remove(t_file)

    print(f"🏆 多进程全量评测完美结束！最终结果已按 task_type 写入 {os.path.dirname(base_out_path)} 目录下。")
    run_auto_metrics(sorted(merged_output_files), base_out_path)

if __name__ == '__main__':
    main()

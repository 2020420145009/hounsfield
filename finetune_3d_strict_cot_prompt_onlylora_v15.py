#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from gpu_wait import wait_for_gpus
from med3d_image_utils import get_pixel_values, resolve_med_image_path, save_slice_cache
from prompt_contracts import CURRENT_INFERENCE_COT_PROMPT, build_ct_slice_description, normalize_question_text

parser = argparse.ArgumentParser(description='Fleming-VL-8B v15 fine-tuning with LLM LoRA only')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--use_llm_lora', type=int, default=1)
parser.add_argument('--lora_r', type=int, default=64)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.1)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--num_slices', type=int, default=8)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--grad_accum_steps', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--save_dir', type=str, default='./work_dir')
parser.add_argument('--model_name', type=str, default='custom_model')
parser.add_argument('--gem_alpha', type=float, default=0.1)
parser.add_argument('--skip_weight_loading', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_ids', type=str, default=None)
parser.add_argument('--use_flash_attn', action='store_true', default=False)
parser.add_argument('--wait_for_gpus', action='store_true', default=False)
parser.add_argument('--gpu_poll_interval_sec', type=int, default=60)
parser.add_argument('--gpu_max_used_memory_mb', type=int, default=4096)
parser.add_argument('--gpu_max_utilization', type=int, default=20)
parser.add_argument('--debug_print_every', type=int, default=100)
parser.add_argument('--slice_cache_mode', type=str, default='auto', choices=['auto', 'require', 'disable'])
parser.add_argument('--precompute_slice_cache', action='store_true', default=False)
parser.add_argument('--precompute_slice_cache_limit', type=int, default=-1)
args = parser.parse_args()

preinit_local_rank = int(os.environ.get('LOCAL_RANK', 0))
if args.wait_for_gpus and (not args.no_cuda) and preinit_local_rank == 0:
    wait_for_gpus(
        gpu_ids=args.gpu_ids,
        max_used_memory_mb=args.gpu_max_used_memory_mb,
        max_utilization=args.gpu_max_utilization,
        poll_interval_sec=args.gpu_poll_interval_sec,
        label='finetune_v15',
    )

world_size = int(os.environ.get('WORLD_SIZE', 1))
is_distributed = world_size > 1
if is_distributed:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    local_rank = 0
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda:0')
if is_distributed:
    dist.barrier()

dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

import importlib
model_dir = os.path.realpath(args.model_path)
parent_dir = os.path.dirname(model_dir)
pkg_name = os.path.basename(model_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
init_path = os.path.join(model_dir, '__init__.py')
if not os.path.exists(init_path):
    open(init_path, 'w').close()
InternVLChatConfig = importlib.import_module(f'{pkg_name}.configuration_internvl_chat').InternVLChatConfig
InternVLChatModel = importlib.import_module(f'{pkg_name}.modeling_internvl_chat').InternVLChatModel

config = InternVLChatConfig.from_pretrained(args.model_path)
config.use_adapter = False
config.use_3d_projector = False
config.use_llm_lora = 0
config.lora_r = args.lora_r
config.lora_alpha = args.lora_alpha
config.lora_dropout = args.lora_dropout
enable_llm_lora = bool(args.use_llm_lora)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = InternVLChatModel(config, use_flash_attn=args.use_flash_attn)
model = model.to(torch.float32)
if not args.skip_weight_loading:
    index_path = os.path.join(args.model_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index['weight_map'].values()))
        state_dict = {}
        for shard in shard_files:
            state_dict.update(load_file(os.path.join(args.model_path, shard), device='cpu'))
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
model = model.to(device=device, dtype=dtype)
model.language_model.gradient_checkpointing_enable()

for param in model.parameters():
    param.requires_grad = False
if enable_llm_lora:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=args.lora_dropout,
        task_type='CAUSAL_LM',
        bias='none',
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

TOKENS_PER_SLICE = model.module.num_image_token if is_distributed else model.num_image_token
IMG_CONTEXT_TOKEN_ID = 151667
if is_distributed:
    model.module.img_context_token_id = IMG_CONTEXT_TOKEN_ID
else:
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID


class Med3DDataset(Dataset):
    def __init__(self, data_path, num_slices, tokenizer, image_processor, dtype, slice_cache_mode):
        self.data = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.data = raw if isinstance(raw, list) else [raw]
        self.num_slices = num_slices
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dtype = dtype
        self.slice_cache_mode = slice_cache_mode
        self.debug_print_every = max(0, args.debug_print_every)

    def __len__(self):
        return len(self.data)

    def iter_unique_volume_paths(self):
        seen = set()
        for sample in self.data:
            image_value = sample.get('image')
            if not isinstance(image_value, str):
                continue
            resolved = resolve_med_image_path(image_value)
            if not isinstance(resolved, str):
                continue
            lower = resolved.lower()
            if not (lower.endswith('.npy') or lower.endswith('.nii') or lower.endswith('.nii.gz')):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved

    def precompute_slice_cache(self, limit=-1):
        total = 0
        for total, img_path in enumerate(self.iter_unique_volume_paths(), start=1):
            save_slice_cache(img_path, overwrite=False)
            if limit > 0 and total >= limit:
                break
        return total

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = resolve_med_image_path(sample['image'])
        pixel_values = get_pixel_values(
            img_path,
            self.num_slices,
            self.image_processor,
            slice_cache_mode=self.slice_cache_mode,
        ).to(self.dtype)

        conv = sample['conversations']
        human_text = normalize_question_text(conv[0]['value'])
        gpt_text = conv[1]['value']

        image_description = build_ct_slice_description(self.num_slices)
        img_start, img_end, img_ctx = '<img>', '</img>', '<IMG_CONTEXT>'
        image_tokens = ''.join(img_start + img_ctx * TOKENS_PER_SLICE + img_end for _ in range(self.num_slices))
        user_content = f'{image_description}\n{image_tokens}\n{human_text}'

        context_str = (
            f'<|im_start|>system\n{CURRENT_INFERENCE_COT_PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n{user_content}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        target_str = f'{gpt_text}<|im_end|>\n'

        if self.debug_print_every > 0 and idx % self.debug_print_every == 0:
            debug_preview = (
                f'<|im_start|>user\n{image_description}\n{human_text}<|im_end|>\n'
                f'<|im_start|>assistant\n{gpt_text}<|im_end|>\n'
            )
            print(f"\n[Finetune Debug] sample_idx={idx}\n{debug_preview}")

        context_ids = self.tokenizer.encode(context_str, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
        max_len = self.tokenizer.model_max_length
        input_ids_list = context_ids + target_ids
        if len(input_ids_list) > max_len:
            input_ids_list = input_ids_list[:max_len]
            allowed_target_len = max(0, max_len - len(context_ids))
            target_ids = target_ids[:allowed_target_len]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([-100] * len(context_ids) + target_ids, dtype=torch.long)
        image_flags = torch.ones(self.num_slices, 1, dtype=torch.long)
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_flags': image_flags,
        }


def collate_fn(batch):
    pixel_values = torch.cat([b['pixel_values'] for b in batch], dim=0)
    image_flags = torch.cat([b['image_flags'] for b in batch], dim=0)
    input_ids = torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'image_flags': image_flags,
    }


train_dataset = Med3DDataset(args.data_path, args.num_slices, tokenizer, image_processor, dtype, args.slice_cache_mode)
if args.precompute_slice_cache:
    if local_rank == 0:
        num_precomputed = train_dataset.precompute_slice_cache(limit=args.precompute_slice_cache_limit)
        print(f'[Slice Cache] precomputed_or_reused={num_precomputed} mode={args.slice_cache_mode}')
    if is_distributed:
        dist.barrier()
if is_distributed:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
else:
    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

model.train()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
for epoch in range(1, args.epochs + 1):
    if is_distributed:
        train_sampler.set_epoch(epoch)
    if local_rank == 0:
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    else:
        progress_bar = train_loader
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        is_accumulating = (batch_idx + 1) % args.grad_accum_steps != 0 and (batch_idx + 1) != len(train_loader)
        sync_context = model.no_sync() if is_distributed and is_accumulating else nullcontext()
        with sync_context:
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image_flags=batch['image_flags'],
                labels=batch['labels'],
                return_dict=True,
            )
            loss = outputs.loss
            if args.gem_alpha > 0 and hasattr(outputs, 'logits'):
                logits = outputs.logits
                labels = batch['labels']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                valid_mask = shift_labels != -100
                valid_logits = shift_logits[valid_mask]
                if valid_logits.numel() > 0:
                    probs = torch.softmax(valid_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    loss = loss - args.gem_alpha * entropy
            loss = loss / args.grad_accum_steps
            loss.backward()
        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
if local_rank == 0:
    from datetime import datetime
    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{run_tag}_{args.model_name}')
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if is_distributed else model
    if enable_llm_lora:
        model_to_save.language_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
if is_distributed:
    dist.barrier()
    dist.destroy_process_group()

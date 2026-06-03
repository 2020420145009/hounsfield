#!/usr/bin/env python3
"""Lingshu-7B (Qwen2.5-VL) LoRA fine-tuning for 3D medical VQA."""
import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from med3d_image_utils import load_or_create_cached_slice_images, resolve_med_image_path
from prompt_contracts import CURRENT_INFERENCE_COT_PROMPT, build_ct_slice_description, normalize_question_text

parser = argparse.ArgumentParser(description='Lingshu-7B LoRA fine-tuning v1')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='./work_dir')
parser.add_argument('--model_name', type=str, default='lingshu_lora')
parser.add_argument('--num_slices', type=int, default=8)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--grad_accum_steps', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--lora_r', type=int, default=64)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.1)
parser.add_argument('--gpu_ids', type=str, default=None)
parser.add_argument('--use_flash_attn', action='store_true', default=False)
parser.add_argument('--debug_print_every', type=int, default=100)
parser.add_argument('--slice_cache_mode', type=str, default='auto', choices=['auto', 'require', 'disable'])
args = parser.parse_args()

# ── distributed setup ──────────────────────────────────────────────────────────
dist.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
is_main = (local_rank == 0)

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(args.save_dir, f'{timestamp}_{args.model_name}')
if is_main:
    os.makedirs(save_path, exist_ok=True)
    print(f'[INFO] save_path={save_path}')

dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

# ── model & processor ──────────────────────────────────────────────────────────
attn_kwargs = {}
if args.use_flash_attn:
    attn_kwargs['attn_implementation'] = 'flash_attention_2'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    **attn_kwargs,
)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
model = get_peft_model(model, lora_config)
if is_main:
    model.print_trainable_parameters()

model = model.to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

processor = AutoProcessor.from_pretrained(args.model_path)
if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
tokenizer = processor.tokenizer


# ── dataset ────────────────────────────────────────────────────────────────────
class Med3DDataset(Dataset):
    def __init__(self, data_path: str, num_slices: int, slice_cache_mode: str):
        self.num_slices = num_slices
        self.slice_cache_mode = slice_cache_mode
        with open(data_path) as f:
            self.samples = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = resolve_med_image_path(sample['image'])
        convs = sample['conversations']
        human_turn = next((c['value'] for c in convs if c['from'] == 'human'), '')
        gpt_turn = next((c['value'] for c in convs if c['from'] == 'gpt'), '')

        question = normalize_question_text(human_turn)
        answer = gpt_turn.strip()

        images = load_or_create_cached_slice_images(
            image_path,
            num_slices=self.num_slices,
            image_size=448,
        )
        return {'question': question, 'answer': answer, 'images': images}


def collate_fn(batch):
    return batch


dataset = Med3DDataset(args.data_path, args.num_slices, args.slice_cache_mode)
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                    collate_fn=collate_fn, num_workers=2, pin_memory=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
)

slice_desc = build_ct_slice_description(args.num_slices)


def build_messages(question: str, answer: str, images):
    user_content = [{'type': 'image', 'image': img} for img in images]
    user_content.append({'type': 'text', 'text': f'{slice_desc}\n{question}'})
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': CURRENT_INFERENCE_COT_PROMPT}]},
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
    ]


# ── training loop ──────────────────────────────────────────────────────────────
global_step = 0
for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}', disable=not is_main)):
        texts, all_images = [], []
        for item in batch:
            msgs = build_messages(item['question'], item['answer'], item['images'])
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            all_images.extend(item['images'])

        inputs = processor(
            text=texts,
            images=all_images,
            return_tensors='pt',
            padding=True,
        ).to(device)

        # build labels: mask everything except the assistant turn
        labels = inputs['input_ids'].clone()
        for i, text in enumerate(texts):
            # find assistant turn start via tokenizing prompt-only version
            prompt_msgs = build_messages(batch[i]['question'], '', batch[i]['images'])
            prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'][0]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / args.grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if is_main and args.debug_print_every > 0 and global_step % args.debug_print_every == 0:
                print(f'[step {global_step}] loss={accum_loss:.4f}')
            accum_loss = 0.0

    # flush remaining gradients
    if (len(loader) % args.grad_accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

dist.barrier()

# ── save LoRA ──────────────────────────────────────────────────────────────────
if is_main:
    model.module.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f'[INFO] LoRA saved to {save_path}')

dist.barrier()
dist.destroy_process_group()

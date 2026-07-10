#!/usr/bin/env python3
# =============================================================================
# 文件名      : finetune_3d.py
# 功能描述    : Fleming-VL-8B 医学多模态大模型微调脚本（引入 GEM 熵最大化损失）。
#               保持模型原始架构不变，仅通过 LoRA 对语言模型进行参数高效微调。
# 依赖环境    : Python 3.10, PyTorch >= 2.0, transformers, peft,
#               nibabel, Pillow, tqdm, safetensors
# =============================================================================

import argparse
import json
import os
import sys
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor
from contextlib import nullcontext
from gpu_wait import wait_for_gpus
from prompt_contracts import normalize_question_text,UNIVERSAL_INFERENCE_COT_PROMPT

# -----------------------------------------------------------------------------
# 命令行参数定义
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Fleming-VL-8B Fine-tuning with GEM Loss')
parser.add_argument('--model_path', type=str, required=True,
                    help='预训练模型目录路径，需包含 config.json 及 safetensors 权重文件')

# LoRA 参数
parser.add_argument('--use_llm_lora', type=int, default=1,
                    help='是否对语言模型启用 LoRA（1=启用，0=关闭）')
parser.add_argument('--lora_r',       type=int,   default=16)
parser.add_argument('--lora_alpha',   type=int,   default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)

# 数据与训练超参数
parser.add_argument('--data_path',   type=str,   required=True,
                    help='训练数据集 JSON 文件路径')
parser.add_argument('--num_slices',  type=int,   default=16,
                    help='每个 CT 体积均匀采样的切片数量')
parser.add_argument('--epochs',      type=int,   default=3,
                    help='训练总轮数')
parser.add_argument('--batch_size',  type=int,   default=1,
                    help='单张卡的 Batch Size，显存不足时保持为 1')
parser.add_argument('--grad_accum_steps', type=int, default=8,
                    help='单卡梯度累加步数')
parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='梯度裁剪阈值，防止梯度爆炸')
parser.add_argument('--lr',          type=float, default=2e-5,
                    help='AdamW 优化器学习率（8B级别模型建议 1e-5 ~ 5e-5）')
parser.add_argument('--dtype',       type=str,   default='bfloat16',
                    choices=['bfloat16', 'float32'])
parser.add_argument('--save_dir',    type=str,   default='./work_dir',
                    help='训练结果保存根目录')
parser.add_argument('--model_name',  type=str,   default='custom_model',
                    help='自定义模型名称，将拼接到保存目录的后缀中')

# GEM 参数
parser.add_argument('--gem_alpha',   type=float, default=0.1,
                    help='GEM 熵最大化损失权重。设定 > 0 时启用。越大越鼓励输出多样性。')

# 调试与硬件控制参数
parser.add_argument('--skip_weight_loading', action='store_true', default=False)
parser.add_argument('--no_cuda',      action='store_true', default=False)
parser.add_argument('--gpu_ids',      type=str, default=None)
parser.add_argument('--use_flash_attn', action='store_true', default=False)
parser.add_argument('--wait_for_gpus', action='store_true', default=False)
parser.add_argument('--gpu_poll_interval_sec', type=int, default=60)
parser.add_argument('--gpu_max_used_memory_mb', type=int, default=4096)
parser.add_argument('--gpu_max_utilization', type=int, default=20)
parser.add_argument('--debug_print_every', type=int, default=100,
                    help='每隔多少个样本打印一次微调输入文本（不含图像 token）；<=0 关闭')

args = parser.parse_args()

preinit_local_rank = int(os.environ.get("LOCAL_RANK", 0))
if args.wait_for_gpus and (not args.no_cuda) and preinit_local_rank == 0:
    wait_for_gpus(
        gpu_ids=args.gpu_ids,
        max_used_memory_mb=args.gpu_max_used_memory_mb,
        max_utilization=args.gpu_max_utilization,
        poll_interval_sec=args.gpu_poll_interval_sec,
        label="finetune",
    )

# -----------------------------------------------------------------------------
# DDP (Distributed Data Parallel) 多卡环境初始化
# -----------------------------------------------------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 自动检测是否处于 torchrun 分布式环境
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed = world_size > 1

if is_distributed:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
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

# -----------------------------------------------------------------------------
# 模型包路径注册
# -----------------------------------------------------------------------------
import importlib
model_dir  = os.path.realpath(args.model_path)
parent_dir = os.path.dirname(model_dir)
pkg_name   = os.path.basename(model_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

init_path = os.path.join(model_dir, '__init__.py')
if not os.path.exists(init_path):
    open(init_path, 'w').close()

pkg                = importlib.import_module(pkg_name)
InternVLChatConfig = importlib.import_module(f'{pkg_name}.configuration_internvl_chat').InternVLChatConfig
InternVLChatModel  = importlib.import_module(f'{pkg_name}.modeling_internvl_chat').InternVLChatModel

# -----------------------------------------------------------------------------
# 配置、分词器与模型初始化
# -----------------------------------------------------------------------------
config = InternVLChatConfig.from_pretrained(args.model_path)
config.use_adapter      = False           
config.use_3d_projector = False           
# 训练脚本会在下方显式调用 get_peft_model() 包装 language_model。
# 这里若也打开模型构造阶段的内部 LoRA 包装，会导致双重 PEFT 包装，
# 保存出来的 key 多一层 `base_model.model.base_model.model.` 前缀，推理时无法匹配。
config.use_llm_lora     = 0
config.lora_r           = args.lora_r
config.lora_alpha       = args.lora_alpha
config.lora_dropout     = args.lora_dropout
enable_llm_lora         = bool(args.use_llm_lora)

tokenizer       = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(args.model_path)

model = InternVLChatModel(config, use_flash_attn=args.use_flash_attn)
model = model.to(torch.float32)

if not args.skip_weight_loading:
    try:
        from safetensors.torch import load_file
        index_path = os.path.join(args.model_path, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index['weight_map'].values()))
            state_dict = {}
            for shard in shard_files:
                state_dict.update(load_file(os.path.join(args.model_path, shard), device='cpu'))
            if config.use_3d_projector:
                state_dict = model._remap_mlp1_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            import gc; gc.collect()
    except ImportError:
        pass

model = model.to(device=device, dtype=dtype)
model.language_model.gradient_checkpointing_enable()

# -----------------------------------------------------------------------------
# 可训练参数策略与 PEFT / DDP 包装
# -----------------------------------------------------------------------------
for param in model.parameters():
    param.requires_grad = False

if enable_llm_lora:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model.language_model = get_peft_model(model.language_model, lora_config)

for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True

if is_distributed:
    # 包装为 DDP 模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

# -----------------------------------------------------------------------------
# 数据集与数据加载器初始化
# -----------------------------------------------------------------------------
TOKENS_PER_SLICE     = model.module.num_image_token if is_distributed else model.num_image_token   
IMG_CONTEXT_TOKEN_ID = 151667                  

if is_distributed:
    model.module.img_context_token_id = IMG_CONTEXT_TOKEN_ID
else:
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID

class Med3DDataset(Dataset):
    def __init__(self, data_path, num_slices, tokenizer, image_processor, dtype):
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
        self.num_slices    = num_slices
        self.tokenizer     = tokenizer
        self.image_processor = image_processor
        self.dtype         = dtype
        self.debug_print_every = max(0, args.debug_print_every)

        self.assistant_marker = self.tokenizer.encode('<|im_start|>assistant', add_special_tokens=False)
        self.marker_len       = len(self.assistant_marker)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample   = self.data[idx]
        img_path = sample['image']

        if img_path.endswith('.npy'):
            vol = np.load(img_path)
        else:
            import nibabel as nib
            vol = nib.load(img_path).get_fdata()
            
        if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:
            vol = vol.transpose(2, 0, 1)
        D = vol.shape[0]

        indices = [int(round(i)) for i in torch.linspace(0, D - 1, self.num_slices).tolist()]

        pixel_values_list = []
        for i in indices:
            sl = vol[i]
            sl = np.squeeze(sl)
            if sl.ndim > 2:
                sl = sl[..., 0] if sl.shape[-1] < sl.shape[0] else sl[0, ...]
                sl = np.squeeze(sl)
                
            sl  = ((sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255).astype(np.uint8)
            rgb = np.stack([sl, sl, sl], axis=-1) 
            img = Image.fromarray(rgb).resize((448, 448))
            pv  = self.image_processor(images=img, return_tensors='pt').pixel_values[0]
            pixel_values_list.append(pv)
        pixel_values = torch.stack(pixel_values_list, dim=0).to(self.dtype)  

        conv       = sample['conversations']
        human_text = normalize_question_text(conv[0]['value'])
        gpt_text   = conv[1]['value']

        IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
        image_tokens = ''.join(
            IMG_START + IMG_CTX * TOKENS_PER_SLICE + IMG_END
            for _ in range(self.num_slices)
        )

        user_content = f'{image_tokens}\n{human_text}'
        debug_user_content = human_text
        
        context_str = (
            f'<|im_start|>system\n{UNIVERSAL_INFERENCE_COT_PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n{user_content}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        target_str = f'{gpt_text}<|im_end|>\n'

        if self.debug_print_every > 0 and idx % self.debug_print_every == 0:
            debug_preview = (
                f'<|im_start|>user\n{debug_user_content}<|im_end|>\n'
                f'<|im_start|>assistant\n{gpt_text}<|im_end|>\n'
            )
            print(f"\n[Finetune Debug] sample_idx={idx}\n{debug_preview}")

        context_ids = self.tokenizer.encode(context_str, add_special_tokens=False)
        target_ids  = self.tokenizer.encode(target_str, add_special_tokens=False)

        max_len = self.tokenizer.model_max_length
        input_ids_list = context_ids + target_ids
        
        if len(input_ids_list) > max_len:
            input_ids_list = input_ids_list[:max_len]
            allowed_target_len = max(0, max_len - len(context_ids))
            target_ids = target_ids[:allowed_target_len]
            
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        labels_list = [-100] * len(context_ids) + target_ids
        labels = torch.tensor(labels_list, dtype=torch.long)

        image_flags = torch.ones(self.num_slices, 1, dtype=torch.long)

        return {
            'pixel_values':   pixel_values,
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'image_flags':    image_flags,
        }

def collate_fn(batch):
    pixel_values = torch.cat([b['pixel_values'] for b in batch], dim=0)
    image_flags  = torch.cat([b['image_flags']  for b in batch], dim=0)

    input_ids      = torch.nn.utils.rnn.pad_sequence(
        [b['input_ids']      for b in batch], batch_first=True,
        padding_value=tokenizer.pad_token_id or 0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True,
        padding_value=0)
    labels         = torch.nn.utils.rnn.pad_sequence(
        [b['labels']         for b in batch], batch_first=True,
        padding_value=-100) 

    return {
        'pixel_values':   pixel_values,
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
        'image_flags':    image_flags,
    }

train_dataset = Med3DDataset(args.data_path, args.num_slices, tokenizer, image_processor, dtype)

# 配置多卡采样器 (DistributedSampler) 防止卡之间处理重复数据
if is_distributed:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
else:
    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# -----------------------------------------------------------------------------
# 训练循环（集成 GEM 熵最大化损失与多卡梯度优化）
# -----------------------------------------------------------------------------
model.train()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    if is_distributed:
        train_sampler.set_epoch(epoch)

    epoch_loss   = 0.0
    epoch_ce_loss = 0.0
    epoch_gem_loss = 0.0
    
    # 仅在主卡（Rank 0）渲染进度条，防止多卡打印乱码
    if local_rank == 0:
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    else:
        progress_bar = train_loader
    
    optimizer.zero_grad() 

    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}

        # 梯度累加期间取消多卡通讯同步，大幅提升速度
        is_accumulating = (batch_idx + 1) % args.grad_accum_steps != 0 and (batch_idx + 1) != len(train_loader)
        sync_context = model.no_sync() if is_distributed and is_accumulating else nullcontext()

        with sync_context:
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image_flags=batch['image_flags'],
                labels=batch['labels'],
                is_3d=False,
                return_dict=True,
            )
            
            loss = outputs.loss
            ce_loss_val = loss.item()
            
            gem_loss_val = 0.0
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
                    gem_loss = -args.gem_alpha * entropy
                    loss = loss + gem_loss
                    gem_loss_val = gem_loss.item()

            loss = loss / args.grad_accum_steps
            loss.backward()

        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        real_total_loss = loss.item() * args.grad_accum_steps
        epoch_loss += real_total_loss
        epoch_ce_loss += ce_loss_val
        epoch_gem_loss += gem_loss_val
        
        if local_rank == 0:
            progress_bar.set_postfix({
                'tot_L': f'{real_total_loss:.4f}',
                'ce_L': f'{ce_loss_val:.4f}',
                'gem_L': f'{gem_loss_val:.4f}'
            })

# -----------------------------------------------------------------------------
# 权重保存 (仅由主进程执行，避免文件冲突)
# -----------------------------------------------------------------------------
if local_rank == 0:
    from datetime import datetime
    run_tag  = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"{run_tag}_{args.model_name}")

    os.makedirs(save_dir, exist_ok=True)

    # 剔除 DDP 的 wrapper 提取真实模型
    model_to_save = model.module if is_distributed else model

    if enable_llm_lora:
        model_to_save.language_model.save_pretrained(save_dir)
    else:
        trainable_names      = {n for n, p in model_to_save.named_parameters() if p.requires_grad}
        trainable_state_dict = {k: v.cpu() for k, v in model_to_save.state_dict().items() if k in trainable_names}
        torch.save(trainable_state_dict, os.path.join(save_dir, 'trainable_params.bin'))

    tokenizer.save_pretrained(save_dir) 
    config.save_pretrained(save_dir)

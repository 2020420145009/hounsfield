#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
from pathlib import Path

# =============================================================================
# 0. 环境变量设置 (🔴 极其重要：必须写在导入 vllm 之前)
# =============================================================================

# 1. 指定 C 编译器，解决一些环境中可能出现的报错 (根据实际服务器环境调整，如不需要可注释掉)
# os.environ["CC"] = "x86_64-conda-linux-gnu-gcc"
# os.environ["CXX"] = "x86_64-conda-linux-gnu-g++"

# 2. 锁定 GPU (🔴 防止 vLLM 乱占显卡，指定卡 2 和卡 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# =============================================================================
# 现在再导入 vLLM 核心库
# =============================================================================
from vllm import LLM, SamplingParams

# =============================================================================
# 1. 基础设置与参数
# =============================================================================
# 模型路径 (请根据实际路径调整)
MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"
# 病例数据根目录
BASE_DIR = Path("/volume/med-train/users/shuyan/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset")

# 输出数据集路径
OUTPUT_JSON_PATH = BASE_DIR / "medical_short_cot_conversations_local.json"
OUTPUT_JSONL_PATH = BASE_DIR / "medical_short_cot_conversations_local.jsonl"
# 短 CoT 只需要很短的输出长度即可
MAX_OUTPUT_TOKENS = 1024

# =============================================================================
# 2. Prompt 设计 (🔴 这里是针对 Homogenization 问题进行核心修改的地方)
# =============================================================================

# 通用 Short CoT 生成规则：强调高密度医学信息、自然流动和严格格式
COMMON_RULES = """
You are an expert medical imaging specialist. Your task is to convert the provided structured radiological JSON into a sophisticated, highly compact Short Medical Chain-of-Thought (Short CoT) reasoning text.

You MUST strictly adhere to these hard constraints for output formatting:
1.  **Format:** Output only a list of six sequentially numbered sentences, format as "1. [Sentence 1] 2. [Sentence 2] ... 6. [Sentence 6]". strictly do NOT include bullet points, bolding, headings (like "Step 1:"), or standard paragraphs.
2.  **Strict Sentence Count:** Generate exactly SIX (6) distinct sentences.
3.  **One Sentence per Logical Step:** Each of the six logical steps below MUST be expressed in EXACTLY ONE sentence:
    - Step 1: Identify and describe the primary anatomical region/structures based on the scan.
    - Step 2: Formally observe and report key abnormal imaging features (size, texture, logic, laterality).
    - Step 3: Compare with normal anatomy to establish the specific pathological baseline.
    - Step 4: Determine the primary abnormality type (nodule, consolidation, fracture, etc.).
    - Step 5: Evaluate the severity and clinical risk based on characteristics.
    - Step 6: Output the definitive final clinical judgment or diagnosis summary.
4.  **Adverbs as Connectors:** Smoothly link logical steps using transition words or conjunctive adverbs naturally within or between the numbered sentences (e.g., further investigation reveals, consistent with this, subsequently indicating, ultimately leading to).
5.  **Information Density (Critical):** Every single sentence MUST contain at least one specific core medical entity or attribute derived from the input JSON (anatomical structure, lesion name, measurement, density/signal/intensity, morphological descriptor, pathological term, laterality, etc.). Eliminate all filler text or boilerplate phrases (like "Moving on to abdominal findings...").
6.  **Preserve Details:** Retain all clinically meaningful details provided in the JSON while remaining compact.
7.  **Variation:** Step 1 (the start) and Step 6 (the end) must have distinct starting/phrasing styles for natural variation across cases.
"""

# Prompt 变体：微调文风，防止 Homogenization 进一步同质化
PROMPT_VARIANTS = [
    COMMON_RULES + "\nWriting preference:\nUse a highly professional,AUTHORITATIVE, and direct radiological report style, focusing strictly on diagnostic precision with smooth causal transitions.\n",
    COMMON_RULES + "\nWriting preference:\nApply a slightly more analytical ACADEMIC tone, where the reasoning reads like a polished verbal walk-through of the diagnostic deduction process, while adhering strictly to the sentence constraints.\n",
    COMMON_RULES + "\nWriting preference:\nUtilize a sophisticated narrative reporting style within the numbered constraints. Vary sentence structures naturally across the diagnostic flow, avoiding repetitive starting phrasings.\n",
]

# 用于最终 Dataset Human 端的 Prompt 模板
HUMAN_PROMPT_TEMPLATES = [
    "Analyze this medical image case carefully and walk through the diagnostic reasoning step by step.",
    "Examine these imaging findings and explain your clinical diagnostic thought process in a structured manner.",
    "Review the radiological features and provide a professional medical imaging chain-of-thought analysis leading to the diagnosis.",
    "Based on this case data, generate a detailed clinical imaging reasoning workflow.",
    "Examine this radiological case and provide your diagnostic chain of thought using a standard logical diagnostic workflow.",
    "Examine the medical images and Walk through the diagnostic reasoning sequentially, from anatomy identification to final diagnosis."
]

# =============================================================================
# 3. 工具函数 (保留原有高效逻辑)
# =============================================================================
def stable_index(key: str, mod: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def pick_prompt_variant(case_dir: Path) -> str:
    return PROMPT_VARIANTS[stable_index(str(case_dir), len(PROMPT_VARIANTS))]

def pick_human_prompt(case_id: str) -> str:
    return HUMAN_PROMPT_TEMPLATES[stable_index(case_id, len(HUMAN_PROMPT_TEMPLATES))]

def load_json_safely(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_case_id(case_dir: Path, case_data: dict) -> str:
    return str(case_data.get("id") or case_data.get("case_id") or case_data.get("patient_id") or case_dir.name)

def infer_image_path(case_dir: Path, case_data: dict) -> str:
    candidate_keys = ["image_path", "image", "img_path", "volume_path", "nii_path", "scan_path"]
    for key in candidate_keys:
        if key in case_data and case_data[key]:
            return str(case_data[key])
    # 如果 JSON 里没有路径，则寻找 NIfTI 文件或图像文件作为兜底
    for pattern in ["*.nii.gz", "*.nii", "*.mha", "*.mhd", "*.nrrd"]:
        matches = list(case_dir.glob(pattern))
        if matches: return str(matches[0])
    return ""

def read_generated_cot(out_path: Path) -> str:
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# =============================================================================
# 4. 主函数 (保持极限并发推理逻辑)
# =============================================================================
def main():
    if not BASE_DIR.exists():
        print(f"❌ 找不到目录: {BASE_DIR}")
        return

    # 获取所有病例文件夹
    case_folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
    print(f"📂 共发现 {len(case_folders)} 个病例文件夹。")

    # 第一阶段：准备需要推理的 Batch 列表
    batch_messages = []
    valid_case_dirs = []
    
    # 我们为短 CoT 使用新的输出文件名，防止覆盖旧 CoT 文件
    COT_OUTPUT_NAME = "medical_short_cot_reasoning_local2.txt"
    
    print(f"⏳ 正在扫描病例文件夹，构建推理队列...")
    for case_dir in case_folders:
        out_path = case_dir / COT_OUTPUT_NAME
        json_path = case_dir / "integrated_meta.json"
        err_path = case_dir / "medical_short_cot_error.log"

        if out_path.exists():
            continue  # 支持断点续传，跳过已生成的

        if not json_path.exists():
            continue

        try:
            case_data = load_json_safely(json_path)
        except Exception as e:
            # 记录严重错误但不中断任务
            err_path.write_text(f"JSON Read error: {e}\n", encoding="utf-8")
            continue

        # 构造给 DeepSeek 的 Human 输入
        user_prompt = (
            "Patient Data (JSON):\n"
            f"{json.dumps(case_data, ensure_ascii=False, indent=2)}\n\n"
            "Now generate the Short Medical Chain-of-Thought strictly following the required constraints."
        )
        # 根据病例 ID 稳定选择一个 Prompt 变体
        system_prompt = pick_prompt_variant(case_dir)

        # 构造标准的 Chat 对话格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        batch_messages.append(messages)
        valid_case_dirs.append(case_dir)

    print(f"🎯 需要新生成的短 CoT 病例数量: {len(batch_messages)}")

    # 第二阶段：灌入 vLLM 进行极限并发推理
    if batch_messages:
        print(f"\n🚀 正在初始化 vLLM 引擎，加载 70B 模型到 GPU {os.environ['CUDA_VISIBLE_DEVICES']} 中 (预计耗时 2-3 分钟)...")
        # 针对 70B 模型和 GPU 指定进行显存和并行设置
        llm = LLM(
            model=MODEL_DIR,
            tensor_parallel_size=2,        # 必须与上面的卡数保持一致 (卡 2 和 3，共 2 张)
            max_model_len=4096,            # 控制上下文长度防止 OOM
            gpu_memory_utilization=0.95,   # 显存利用率 (极限压榨卡性能)
            dtype="bfloat16",
            trust_remote_code=True
        )

        sampling_params = SamplingParams(
            temperature=0.7,               # R1-Distill 推荐设置
            top_p=0.9,
            max_tokens=MAX_OUTPUT_TOKENS,  # 短思维链不需要太多 token
            stop=["<|im_end|>", "<|im_start|>", "User:", "Patient Data (JSON)"] # 增加防幻觉 stop_tokens
        )

        print("⚡ 开始极限并发推理...")
        # vLLM .chat 会自动处理 batch 推理，use_tqdm 显示进度
        outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=True)

        print("💾 正在将推理结果写入硬盘病例文件夹...")
        for case_dir, output in zip(valid_case_dirs, outputs):
            out_path = case_dir / COT_OUTPUT_NAME
            # 提取推理生成的纯文本结果
            cot_result = output.outputs[0].text.strip()
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cot_result)

        # 主动释放引擎和显存
        print("🧼 正在清理模型引擎释放显存...")
        del llm
        import torch
        torch.cuda.empty_cache()

    # 第三阶段：组装 Dataset
    print("\n📦 正在组装整合最终的短 CoT 对话数据集...")
    dataset = []
    # 重新遍历排序好的文件夹，确保数据集顺序稳定
    for case_dir in case_folders:
        json_path = case_dir / "integrated_meta.json"
        cot_path = case_dir / COT_OUTPUT_NAME

        if not json_path.exists() or not cot_path.exists():
            continue

        try:
            case_data = load_json_safely(json_path)
            # 读取刚才模型生成的 Short CoT
            generated_cot_text = read_generated_cot(cot_path)
        except Exception:
            continue

        # 检查模型是否输出了有效内容，跳过空结果或 ERROR 结果
        if len(generated_cot_text) < 10 or "ERROR:" in generated_cot_text:
            continue

        case_id = infer_case_id(case_dir, case_data)
        dataset.append({
            "id": case_id,
            "image": infer_image_path(case_dir, case_data),
            "conversations": [
                {"from": "human", "value": pick_human_prompt(case_id)},
                {"from": "gpt", "value": generated_cot_text}
            ]
        })

    # 保存文件
    print(f"\n💾 正在保存最终数据集条目到 {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 任务完成！")
    print(f"📊 成功生成并打包条目: {len(dataset)} 条")
    print(f"📄 数据集文件:  {OUTPUT_JSON_PATH}")

# =============================================================================
# 5. 进程保护入口
# =============================================================================
if __name__ == "__main__":
    main()
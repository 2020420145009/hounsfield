# -*- coding: utf-8 -*-

import os
import json
import hashlib
import re
from pathlib import Path

# =============================================================================
# 0. 环境变量设置
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from vllm import LLM, SamplingParams

BASE_DIR = Path("/volume/med-train/users/shuyan/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset")
MODEL_DIR = "/volume/med-train/shared_models/DeepSeek-R1-Distill-Llama-70B"

# 分别保存三个子数据集的路径 (JSON 格式)
OUTPUT_OPEN_VQA_PATH = BASE_DIR / "medical_lidc_open_vqa_dataset4.json"
OUTPUT_MCQ_VQA_PATH  = BASE_DIR / "medical_lidc_mcq_vqa_dataset4.json"
OUTPUT_REPORT_PATH   = BASE_DIR / "medical_lidc_report_dataset4.json"

# 整合后用于直接微调的数据集路径 (JSONL 格式)
OUTPUT_ALL_MIXED_JSONL_PATH = BASE_DIR / "medical_lidc_all_mixed_dataset4.jsonl"

# =============================================================================
# 1. 模块 C & D & 全局质量控制 (Grounding, Persona, Anti-Robotic)
# =============================================================================
GLOBAL_RULES = """
[MODULE C: ANATOMICAL GROUNDING (CRITICAL)]
Whenever describing an abnormality in your reasoning, you MUST explicitly anchor it to its specific anatomical location or coordinates derived from the JSON data. Do not discuss findings in a vacuum.

[GLOBAL QUALITY & ANTI-ROBOTIC CONSTRAINTS]
1. Information Density: Every single reasoning sentence MUST contain specific medical entities (e.g., measurements, HU density, specific lobes).
2. Absolute Anti-Colloquialism: DO NOT use conversational filler (e.g., "Let me analyze", "Based on the JSON").
3. Question Diversity: Your questions ("q" field) MUST use highly diverse sentence structures. Avoid always starting with "What is" or "Where is".
4. Fluent Reasoning: DO NOT use robotic templates like "The anatomical target is...". Write fluent, clinical sentences that naturally contain the required logic.
5. Numerical Precision: You MUST round all measurements (diameter, volume) to a maximum of 2 decimal places. NEVER output long floating-point numbers like 44.662703mm.
"""

PERSONAS = [
    "Persona: Fast-paced ER Doctor. Focus on direct, life-threatening implications.",
    "Persona: Senior Radiology Professor. Focus on deductive, precise feature analysis.",
    "Persona: Meticulous Diagnostician. Always consider and rule out alternatives."
]

# =============================================================================
# 2. 拦截器、JSON 提取与图像路径推断工具
# =============================================================================
def extract_json_array(text: str) -> list:
    """暴力剥离 R1 的 <think> 废话，提取纯净的 JSON 数组"""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []

def infer_image_path(case_dir: Path, case_data: dict) -> str:
    """自动推断图像路径，确保多模态微调时 image 字段不为空"""
    # 1. 尝试从原 JSON 中读取
    if "image_path" in case_data and case_data["image_path"]:
        return str(case_data["image_path"])
    if "image" in case_data and case_data["image"]:
        return str(case_data["image"])
    
    # 2. 动态扫描文件夹下存在的影像文件
    for ext in [".nii.gz", ".nii", ".npy"]:
        possible_img = case_dir / f"{case_dir.name}{ext}"
        if possible_img.exists():
            return str(possible_img.resolve()) # 返回绝对路径
            
    # 3. 兜底策略：假设标准命名格式
    return f"{case_dir.name}.nii.gz"

# =============================================================================
# 3. 主函数 (双重遍历两步反思机制)
# =============================================================================
def main():
    case_folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
    valid_cases = []
    
    for case_dir in case_folders:
        json_path = case_dir / "integrated_meta.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                valid_cases.append({"dir": case_dir, "data": json.load(f)})

    if not valid_cases:
        return

    print(f"🎯 初始化 vLLM, 准备为 {len(valid_cases)} 个病例合成三栖(Open/MCQ/Report)数据...")
    llm = LLM(model=MODEL_DIR, tensor_parallel_size=2, max_model_len=4096, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)

    # ==========================================
    # Pass 1: 初稿生成 (Model-Driven Routing)
    # ==========================================
    pass1_messages = []
    for case in valid_cases:
        case_id = str(case["data"].get("id", case["dir"].name))
        
        # 角色注入 (Module D)
        persona = PERSONAS[int(hashlib.md5(case_id.encode("utf-8")).hexdigest(), 16) % len(PERSONAS)]
        
        # 【核心升级】：把打标签和决定步数的权力完全交给大模型 (Module A)
        system_prompt = f"""You are an expert medical annotator creating a mixed-task dataset for 3D CT scans.
{GLOBAL_RULES}
[MODULE D: PERSONA] {persona}

[TASK MATRIX - THREE FORMATS]
Generate EXACTLY 3 items for this case, one for each format below:
1. "open_vqa": A specific open-ended visual question with diverse phrasing.
2. "mcq_vqa": A multiple-choice question. The "q" field MUST explicitly include 4 options labeled A), B), C), D).
3. "report": A broad request for a comprehensive analysis report.

[MODULE A: MODEL-DRIVEN ROUTING & REASONING PARADIGM]
For EACH item you generate, you must evaluate its complexity and assign an appropriate "trigger_tag".
You MUST strictly format the "reasoning" field with numbered steps corresponding to your chosen tag:

Tag Option 1: "[Perform Comprehensive Diagnosis]"
-> Use for complex VQA or detailed reports.
-> STRICT CONSTRAINT: EXACTLY 6 numbered sentences.
  1. Identify and describe the primary anatomical region/structures.
  2. Formally observe and report key abnormal imaging features.
  3. Compare with normal anatomy to establish the pathological baseline.
  4. Determine the primary abnormality type.
  5. Evaluate the severity and clinical risk.
  6. Output the definitive final clinical judgment.

Tag Option 2: "[Perform Standard Assessment]"
-> Use for medium complexity VQA.
-> STRICT CONSTRAINT: EXACTLY 4 numbered sentences.
  1. Locate target. 2. Extract core features. 3. Synthesize clinical meaning. 4. Final diagnosis.

Tag Option 3: "[Perform Rapid Triage]"
-> Use for simple, direct visual queries.
-> STRICT CONSTRAINT: EXACTLY 3 numbered sentences.
  1. Locate target. 2. State deviation from normal. 3. Immediate conclusion.

[OUTPUT FORMAT]
Output ONLY a JSON array of 3 objects. NO XML TAGS. NO MARKDOWN.
Example:
[
  {{
    "type": "open_vqa",
    "trigger_tag": "[Perform Rapid Triage]",
    "q": "Assess the dimension of the primary lesion.",
    "reasoning": "1. ... \\n2. ... \\n3. ...",
    "a": "It measures 15.00mm."
  }},
  {{
    "type": "mcq_vqa",
    "trigger_tag": "[Perform Standard Assessment] ",
    "q": "Which lobe contains the nodule? A) ... B) ... C) ... D) ...",
    "reasoning": "1. ... \\n2. ... \\n3. ... \\n4. ... \\n5. ... \\n6. ...",
    "a": "A) ..."
  }},
  {{
    "type": "report",
    "trigger_tag": "[Perform Comprehensive Diagnosis]",
    "q": "Provide a detailed diagnostic report.",
    "reasoning": "1. ... \\n2. ... \\n3. ... \\n4. ... \\n5. ... \\n6. ...",
    "a": "..."
  }}
]"""

        user_prompt = f"Patient Data:\n{json.dumps(case['data'], ensure_ascii=False)}"
        pass1_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    print("\n⚡ [Pass 1/2] 正在依据大模型自主意图路由生成初稿...")
    pass1_outputs = llm.chat(messages=pass1_messages, sampling_params=sampling_params, use_tqdm=True)

    # ==========================================
    # Pass 2: 反思重写 (Module B - Reflection)
    # ==========================================
    pass2_messages = []
    for i, output in enumerate(pass1_outputs):
        draft_text = output.outputs[0].text
        
        history = pass1_messages[i].copy()
        history.append({"role": "assistant", "content": draft_text})
        
        # 强力校验题型完整性、模型自主标签以及思维链编号匹配度
        reflection_prompt = f"""Review your generated JSON array strictly against the rules:
1. FORMAT: Are there exactly 3 objects with types "open_vqa", "mcq_vqa", and "report"?
2. MCQ INTEGRITY: Does the "mcq_vqa" question ("q") ACTUALLY contain 4 explicitly labeled options (A, B, C, D)?
3. MODEL-DRIVEN ROUTING COMPLIANCE: 
   - If you chose "trigger_tag": "[Perform Comprehensive Diagnosis]", does the "reasoning" contain EXACTLY 6 numbered steps?
   - If "[Perform Standard Assessment]", EXACTLY 4 steps?
   - If "[Perform Rapid Triage]", EXACTLY 3 steps?
4. ANTI-ROBOTIC & PRECISION: Is there ANY conversational filler? Are all numbers rounded to max 2 decimal places?

If there are any violations, REWRITE the JSON. Output ONLY the raw, pure JSON array. DO NOT wrap it in ```json blocks or <think> tags."""
        
        history.append({"role": "user", "content": reflection_prompt})
        pass2_messages.append(history)

    print("\n🔄 [Pass 2/2] 反思机制启动：强力校验模型自主标签与 N 步范式的匹配度...")
    pass2_outputs = llm.chat(messages=pass2_messages, sampling_params=sampling_params, use_tqdm=True)

    # ==========================================
    # 组装与分流保存 (Python 强制解耦拼装标签)
    # ==========================================
    print("\n💾 正在提取自主标签、拼装显式指令、探测图像路径并分流数据集...")
    dataset_open = []
    dataset_mcq = []
    dataset_report = []
    failed = 0
    
    for case, output in zip(valid_cases, pass2_outputs):
        raw_text = output.outputs[0].text
        qa_pairs = extract_json_array(raw_text)
        
        if not qa_pairs or len(qa_pairs) != 3:
            failed += 1
            continue
            
        case_id = str(case["data"].get("id", case["dir"].name))
        
        # 【新增】：动态探测并补全图像路径
        image_path = infer_image_path(case["dir"], case["data"])
        
        for qa in qa_pairs:
            q_type = qa.get("type", "unknown")
            trigger = qa.get("trigger_tag", "[Perform Standard Assessment]")
            q_text = qa.get("q", "")
            reasoning_text = qa.get("reasoning", "").strip()
            a_text = qa.get("a", "").strip()
            
            # 【核心组装】：模型给出的 trigger + 真实提问
            explicit_human_prompt = f"{trigger} {q_text}"
            
            # Python 稳稳拼装内部标签
            perfect_cot_format = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{a_text}\n</answer>"
            
            entry = {
                "id": f"{case_id}_{q_type}",
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": explicit_human_prompt},
                    {"from": "gpt", "value": perfect_cot_format}
                ],
                "task_type": q_type
            }
            
            if q_type == "open_vqa":
                dataset_open.append(entry)
            elif q_type == "mcq_vqa":
                dataset_mcq.append(entry)
            elif q_type == "report":
                dataset_report.append(entry)

    # 释放显存
    del llm
    import torch
    torch.cuda.empty_cache()

    # ==========================================
    # 写入文件系统 (独立的 JSON 和 整合的 JSONL)
    # ==========================================
    
    # 1. 保存为三个独立 JSON 文件 (方便查看和单项抽检)
    with open(OUTPUT_OPEN_VQA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_open, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_MCQ_VQA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_mcq, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_report, f, ensure_ascii=False, indent=2)

    # 2. 【新增】：将全部数据合并为标准 .jsonl 格式 (方便 LLaVA 或 SFT 脚本直接读取)
    all_mixed_dataset = dataset_open + dataset_mcq + dataset_report
    with open(OUTPUT_ALL_MIXED_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in all_mixed_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 终极三栖数据集合成完毕！")
    print(f"📊 成功生成独立分类文件:")
    print(f"   - 开放式 VQA: {len(dataset_open)} 条 -> {OUTPUT_OPEN_VQA_PATH.name}")
    print(f"   - 多选题 VQA: {len(dataset_mcq)} 条 -> {OUTPUT_MCQ_VQA_PATH.name}")
    print(f"   - 分析报告:   {len(dataset_report)} 条 -> {OUTPUT_REPORT_PATH.name}")
    print(f"\n📦 已合并为完整训练集:")
    print(f"   - 混合微调数据: {len(all_mixed_dataset)} 条 -> {OUTPUT_ALL_MIXED_JSONL_PATH.name}")
    print(f"\n⚠️ 失败/格式残缺病例: {failed}")

if __name__ == "__main__":
    main()
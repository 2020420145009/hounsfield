#!/usr/bin/env python3
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
OUTPUT_VQA_JSON_PATH = BASE_DIR / "medical_lidc_vqa_cot_dataset.json"

# =============================================================================
# 1. 模块 C & D：全局铁律 (解剖学锚定与角色池)
# =============================================================================
GLOBAL_RULES = """
[MODULE C: ANATOMICAL GROUNDING (CRITICAL)]
Whenever describing an abnormality in your reasoning, you MUST explicitly anchor it to its specific anatomical location or coordinates derived from the JSON data. Do not discuss findings in a vacuum.

[GLOBAL QUALITY CONSTRAINTS]
1. Information Density: Every single reasoning sentence MUST contain specific medical entities (e.g., measurements, HU density, specific lobes).
2. Absolute Anti-Colloquialism: DO NOT use conversational filler (e.g., "Let me analyze", "Based on the JSON").
"""

PERSONAS = [
    "Persona: Fast-paced ER Doctor. Focus on direct, life-threatening implications.",
    "Persona: Senior Radiology Professor. Focus on deductive, precise feature analysis.",
    "Persona: Meticulous Diagnostician. Always consider and rule out alternatives."
]

# =============================================================================
# 2. 模块 A：动态路由与 VQA 思维链范式
# =============================================================================
ROUTING_MAP = {
    "Complex_VQA": {"trigger": "[Perform Comprehensive Diagnosis]", "steps": 5},
    "Medium_VQA": {"trigger": "[Perform Standard Assessment]", "steps": 4},
    "Simple_VQA": {"trigger": "[Perform Rapid Triage]", "steps": 3},
    "Knowledge": {"trigger": "[Explain Medical Concept]", "steps": 3}
}

def analyze_case_complexity(case_data: dict) -> str:
    """模块 A：基于 JSON 复杂度的探测函数"""
    text = json.dumps(case_data).lower()
    if "metastasis" in text or text.count("nodule") > 1 or len(text) > 1000:
        return "Complex_VQA"
    if len(text) < 400:
        return "Simple_VQA"
    return "Medium_VQA"

def get_vqa_paradigm(steps: int) -> str:
    """为 VQA 的每道题专门定制对应的思维链步骤"""
    if steps == 5:
        return """- STRICT CONSTRAINT: EXACTLY 5 numbered sentences.
  1. Locate the specific target in the anatomical region.
  2. Extract the core imaging feature related to the question.
  3. Compare this feature to normal baseline anatomy.
  4. Synthesize the clinical meaning of this feature.
  5. Conclude specifically to answer the question."""
    elif steps == 4:
        return """- STRICT CONSTRAINT: EXACTLY 4 numbered sentences.
  1. Locate the specific target in the anatomical region.
  2. Extract the core imaging feature related to the question.
  3. Synthesize the clinical meaning.
  4. Conclude specifically to answer the question."""
    else: # 3 steps
        return """- STRICT CONSTRAINT: EXACTLY 3 numbered sentences.
  1. Locate the specific target and extract the relevant feature.
  2. State its deviation from normal anatomy.
  3. Conclude immediately to answer the question."""

# =============================================================================
# 3. 拦截器与 JSON 提取工具
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

# =============================================================================
# 4. 主函数 (双重遍历两步反思机制)
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

    print(f"🎯 初始化 vLLM, 准备为 {len(valid_cases)} 个病例合成 VQA-CoT 矩阵...")
    llm = LLM(model=MODEL_DIR, tensor_parallel_size=2, max_model_len=4096, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)

    # ==========================================
    # Pass 1: 初稿生成 (Drafting 5 VQA pairs)
    # ==========================================
    pass1_messages = []
    for case in valid_cases:
        case_id = str(case["data"].get("id", case["dir"].name))
        
        # 路由分发 (Module A)
        task_type = analyze_case_complexity(case["data"])
        case["trigger"] = ROUTING_MAP[task_type]["trigger"]
        steps = ROUTING_MAP[task_type]["steps"]
        case["target_steps"] = steps
        
        # 角色注入 (Module D)
        persona = PERSONAS[int(hashlib.md5(case_id.encode("utf-8")).hexdigest(), 16) % len(PERSONAS)]
        
        system_prompt = f"""You are an expert medical annotator creating Visual Question Answering (VQA) data.
{GLOBAL_RULES}
[MODULE D: PERSONA] {persona}

[TASK MATRIX - 3D-RadVQA Strategy]
Generate exactly 5 questions for this case covering: 1. Location, 2. Measurement, 3. Morphology, 4. Texture, 5. Malignancy.

[MODULE A: REASONING PARADIGM]
For EACH of the 5 questions, your "reasoning" field MUST follow this strict numbered structure:
{get_vqa_paradigm(steps)}

[OUTPUT FORMAT]
Output ONLY a JSON array of 5 objects. NO XML TAGS. NO MARKDOWN.
Example:
[
  {{
    "q": "Where is the nodule?",
    "reasoning": "1. The anatomical target is... \\n2. The feature is... \\n3. Conclusion...",
    "a": "Right upper lobe."
  }}
]"""

        user_prompt = f"Patient Data:\n{json.dumps(case['data'], ensure_ascii=False)}"
        pass1_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    print("\n⚡ [Pass 1/2] 正在依据 5 维策略与严谨思维链生成 VQA 初稿...")
    pass1_outputs = llm.chat(messages=pass1_messages, sampling_params=sampling_params, use_tqdm=True)

    # ==========================================
    # Pass 2: 反思重写 (Module B - Reflection)
    # ==========================================
    pass2_messages = []
    for i, output in enumerate(pass1_outputs):
        draft_text = output.outputs[0].text
        steps = valid_cases[i]["target_steps"]
        
        history = pass1_messages[i].copy()
        history.append({"role": "assistant", "content": draft_text})
        
        # 强力校验思维链编号与口语化废话
        reflection_prompt = f"""Review your generated JSON array strictly against the rules:
1. Are there exactly 5 questions covering the 5 categories?
2. Does the "reasoning" for EVERY question contain EXACTLY {steps} sequentially numbered sentences (1., 2., etc.)?
3. Is there ANY conversational filler (e.g., "Here is the JSON", "Okay") in the text?

If there are violations, REWRITE the JSON. Output ONLY the raw, pure JSON array. DO NOT wrap it in ```json blocks or <think> tags."""
        
        history.append({"role": "user", "content": reflection_prompt})
        pass2_messages.append(history)

    print("\n🔄 [Pass 2/2] 反思机制启动：强力剔除口语化，校验思维链范式...")
    pass2_outputs = llm.chat(messages=pass2_messages, sampling_params=sampling_params, use_tqdm=True)

    # ==========================================
    # 组装与保存 (Python 强制解耦拼装标签)
    # ==========================================
    print("\n💾 正在应用隐式激活与 Python 标签拼装...")
    dataset = []
    failed = 0
    
    for case, output in zip(valid_cases, pass2_outputs):
        raw_text = output.outputs[0].text
        qa_pairs = extract_json_array(raw_text)
        
        if not qa_pairs or len(qa_pairs) != 5:
            failed += 1
            continue
            
        case_id = str(case["data"].get("id", case["dir"].name))
        image_path = str(case["data"].get("image_path", ""))
        trigger = case["trigger"]
        
        for idx, qa in enumerate(qa_pairs):
            q_text = qa.get("q", "")
            reasoning_text = qa.get("reasoning", "").strip()
            a_text = qa.get("a", "").strip()
            
            # 【核心组装】：方案一 (显式激活) + Python 无错误拼装标签
            explicit_human_prompt = f"{trigger} {q_text}"
            perfect_cot_format = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{a_text}\n</answer>"
            
            dataset.append({
                "id": f"{case_id}_vqa_{idx+1}",
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": explicit_human_prompt},
                    {"from": "gpt", "value": perfect_cot_format}
                ],
                "task_type": "VQA-CoT"
            })

    del llm
    import torch
    torch.cuda.empty_cache()

    with open(OUTPUT_VQA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 VQA-CoT 数据集合成完毕！")
    print(f"📊 成功生成 {len(dataset)} 条带严谨思维链的短问答数据。失败病例: {failed}。")

if __name__ == "__main__":
    main()
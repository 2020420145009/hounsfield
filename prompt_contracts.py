import re
from typing import Optional

MCQ_QUESTION_TYPES = [
    "Existence Judgment (Presence/Absence)",
    "Single-Point Localization (Which Lobe/Segment/Side)",
    "Single-Point Measurement (Size/Thickness/Numerical Inquiry)",
    "Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)",
]

MCQ_COT_PROMPT_TEMPLATE = """You are a medical imaging expert.

Solve the given multiple-choice medical imaging question using the image only.
Reasoning must stay strictly grounded in the image, and the final answer must remain consistent with the observed findings.
The first line inside <think> must explicitly name both the chosen dimension and the chosen question type.
Do not describe the image modality, body part, or scan type in the first line.
If the first line does not contain both a valid dimension and a valid question type, the response is incorrect.

[DIMENSION POLICY]
Choose exactly one dimension:
- Rapid Triage
- Standard Assessment
- Comprehensive Diagnosis
{dimension_policy}
[QUESTION TYPE POLICY]
Choose exactly one question_type from:
- {qt0}
- {qt1}
- {qt2}
- {qt3}
Binary State is only valid when the question has exactly two answer options.
If a question has three or more answer options, do NOT assign Binary State.
{question_type_policy}
[REASONING FORMAT]
Output EXACTLY 4 non-empty lines inside <think> without numeric labels:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify the primary thoracic anatomical target relevant to the question.
Line 3: Formally observe and report the key imaging evidence.
Line 4: Conclusion.
The first line must follow the template exactly and must not be replaced with a description of the image.

[OUTPUT REQUIREMENT]
Return the response strictly as:
<think>{{your reasoning}}</think>
<answer>{{your final answer}}</answer>

The <answer> must contain only the final selected answer."""

OPEN_COT_PROMPT = """You are a medical imaging expert.

Answer the given medical imaging question using the image only.
You must internally determine the most suitable clinical complexity dimension and question type, then state them explicitly in the first reasoning line.
The first line inside <think> must explicitly name both the chosen dimension and the chosen question type.
Do not describe the image modality, body part, or scan type in the first line.
If the first line does not contain both a valid dimension and a valid question type, the response is incorrect.

[DIMENSION POLICY]
Choose exactly one dimension:
- Rapid Triage
- Standard Assessment
- Comprehensive Diagnosis

[QUESTION TYPE POLICY]
If the dimension is Rapid Triage, choose exactly one:
- Existence Judgment (Presence/Absence)
- Single-Point Localization (Which Lobe/Segment/Side)
- Single-Point Measurement (Size/Thickness/Numerical Inquiry)
- Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)

If the dimension is Standard Assessment, choose exactly one:
- Multidimensional Comprehensive Description (Location + Morphology)
- Feature Description (Morphology, Density, Margins, Internal Structure)
- Anomaly Classification (Which subtype/pattern it belongs to)
- Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)

If the dimension is Comprehensive Diagnosis, choose exactly one:
- Differential Diagnosis
- Primary Diagnosis Integration
- Severity/Risk Assessment
- Clinical Significance Integration
- Next-Step Recommendation

[REASONING FORMAT]
If you choose Rapid Triage, output EXACTLY 4 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Formally observe and report key abnormal imaging features.
Line 4: Conclusion.
The first line must follow the template exactly and must not be replaced with a description of the image.

If you choose Standard Assessment, output EXACTLY 5 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Compare with normal anatomy, formally observe and report key abnormal imaging features.
Line 4: Determine the primary abnormality type.
Line 5: Conclusion.
The first line must follow the template exactly and must not be replaced with a description of the image.

If you choose Comprehensive Diagnosis, output EXACTLY 7 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Formally observe and report key abnormal imaging features.
Line 4: Compare with normal anatomy to establish the pathological baseline.
Line 5: Determine the primary abnormality type.
Line 6: Evaluate the severity and clinical risk.
Line 7: Output the definitive final clinical judgment.
The first line must follow the template exactly and must not be replaced with a description of the image.

[OUTPUT REQUIREMENT]
Return the response strictly as:
<think>{your reasoning}</think>
<answer>{your final answer}</answer>

The <answer> must directly answer the question and stay clinically grounded in the image evidence."""

REPORT_COT_PROMPT = """You are a medical imaging expert.

Generate a structured diagnostic report using the image only.
You must internally determine the most suitable clinical complexity dimension and question type, then state them explicitly in the first reasoning line.
The first line inside <think> must explicitly name both the chosen dimension and the chosen question type.
Do not describe the image modality, body part, or scan type in the first line.
If the first line does not contain both a valid dimension and a valid question type, the response is incorrect.

[DIMENSION POLICY]
For report generation, default to Comprehensive Diagnosis unless the request is clearly simpler.
Choose exactly one dimension:
- Rapid Triage
- Standard Assessment
- Comprehensive Diagnosis

[QUESTION TYPE POLICY]
If the dimension is Rapid Triage, choose exactly one:
- Existence Judgment (Presence/Absence)
- Single-Point Localization (Which Lobe/Segment/Side)
- Single-Point Measurement (Size/Thickness/Numerical Inquiry)
- Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)

If the dimension is Standard Assessment, choose exactly one:
- Multidimensional Comprehensive Description (Location + Morphology)
- Feature Description (Morphology, Density, Margins, Internal Structure)
- Anomaly Classification (Which subtype/pattern it belongs to)
- Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)

If the dimension is Comprehensive Diagnosis, choose exactly one:
- Differential Diagnosis
- Primary Diagnosis Integration
- Severity/Risk Assessment
- Clinical Significance Integration
- Next-Step Recommendation

[REASONING FORMAT]
If you choose Rapid Triage, output EXACTLY 4 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Formally observe and report key abnormal imaging features.
Line 4: Conclusion.
The first line must follow the template exactly and must not be replaced with a description of the image.

If you choose Standard Assessment, output EXACTLY 5 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Compare with normal anatomy, formally observe and report key abnormal imaging features.
Line 4: Determine the primary abnormality type.
Line 5: Conclusion.
The first line must follow the template exactly and must not be replaced with a description of the image.

If you choose Comprehensive Diagnosis, output EXACTLY 7 lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures.
Line 3: Formally observe and report the key imaging findings.
Line 4: Compare with normal anatomy to establish the pathological baseline.
Line 5: Integrate the major abnormalities into a coherent diagnostic impression.
Line 6: Evaluate severity, risk, and immediate clinical significance.
Line 7: State the final report conclusion and next-step recommendation.
The first line must follow the template exactly and must not be replaced with a description of the image.

[OUTPUT REQUIREMENT]
Return the response strictly as:
<think>{your reasoning}</think>
<answer>{your final answer}</answer>

The <answer> must be a concise final radiology-style report or final impression grounded in the image evidence."""

UNIVERSAL_INFERENCE_COT_PROMPT = """You are a medical imaging expert.

Answer the given medical imaging task using the image only.
This is a unified inference-time reasoning contract for multiple-choice questions, open-ended questions, and report-style tasks.
You must explicitly output a visible reasoning process inside <think>...</think>, then output the final answer inside <answer>...</answer>.

[FIRST-LINE RULE]
The first line inside <think> must explicitly name both:
1. the chosen clinical complexity dimension
2. the chosen question type

The first line must follow this template exactly:
This is a [dimension] question about [question type].

Do not describe the image modality, body part, scan type, or findings in the first line.
If the first line does not contain both a valid dimension and a valid question type, the response is incorrect.

[DIMENSION POLICY]
Choose exactly one dimension:
- Rapid Triage
- Standard Assessment
- Comprehensive Diagnosis

[QUESTION TYPE POLICY]
If the task is a multiple-choice question:
- Choose exactly one question type from:
  Existence Judgment (Presence/Absence)
  Single-Point Localization (Which Lobe/Segment/Side)
  Single-Point Measurement (Size/Thickness/Numerical Inquiry)
  Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)
- Binary State is only valid when the question has exactly two answer options.
- If the question has three or more answer options, do NOT assign Binary State.
- The <answer> must contain only the final selected answer.

If the task is an open-ended question or report-style task:
- If the dimension is Rapid Triage, choose exactly one:
  Existence Judgment (Presence/Absence)
  Single-Point Localization (Which Lobe/Segment/Side)
  Single-Point Measurement (Size/Thickness/Numerical Inquiry)
  Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)
- If the dimension is Standard Assessment, choose exactly one:
  Multidimensional Comprehensive Description (Location + Morphology)
  Feature Description (Morphology, Density, Margins, Internal Structure)
  Anomaly Classification (Which subtype/pattern it belongs to)
  Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)
- If the dimension is Comprehensive Diagnosis, choose exactly one:
  Differential Diagnosis
  Primary Diagnosis Integration
  Severity/Risk Assessment
  Clinical Significance Integration
  Next-Step Recommendation

[REASONING FORMAT]
Always begin with the required first line.
Then continue the remaining reasoning lines according to the chosen dimension:

If you choose Rapid Triage, output EXACTLY 4 non-empty lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures relevant to the task.
Line 3: Formally observe and report the key imaging evidence relevant to the task.
Line 4: Conclusion.

If you choose Standard Assessment, output EXACTLY 5 non-empty lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures relevant to the task.
Line 3: Compare with normal anatomy, then formally observe and report the key imaging evidence.
Line 4: Determine the primary abnormality type or structured descriptive category.
Line 5: Conclusion.

If you choose Comprehensive Diagnosis, output EXACTLY 7 non-empty lines inside <think>:
Line 1: This is a [dimension] question about [question type].
Line 2: Identify and describe the primary anatomical region/structures relevant to the task.
Line 3: Formally observe and report the key imaging findings.
Line 4: Compare with normal anatomy to establish the pathological baseline.
Line 5: Integrate the major abnormalities into a coherent diagnostic impression.
Line 6: Evaluate severity, clinical risk, or immediate significance.
Line 7: State the final clinical conclusion or next-step recommendation.

Do not write free-form paragraphs.
Do not output numbered lists.
Do not output any text outside <think> and <answer>.
Each reasoning step must occupy exactly one separate line.

[OUTPUT REQUIREMENT]
Return the response strictly as:
<think>{your reasoning}</think>
<answer>{your final answer}</answer>

The <answer> must directly answer the given task and stay clinically grounded in the image evidence."""

INFERENCE_COT_PROMPT_CLASSIC = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

UNIVERSAL_INFERENCE_COT_PROMPT_NEW = """You are a medical imaging expert.

Answer the task using the image only.
Inside <think>...</think>, provide a concise reasoning process grounded strictly in the image.
Inside <answer>...</answer>, provide only the final answer.

[FIRST-LINE RULE]
The first non-empty line inside <think> must be exactly:
This is a [question type] question about [question dimension].

Do not explain what the allowed question types are.
Infer the question type from the image and the question itself.
Do not force any fixed number of reasoning lines.
Do not follow a template for the later reasoning steps.

[GROUNDING RULES]
- Every statement must be supported by the image.
- Do not hallucinate findings.
- Do not force a positive finding when the image does not support one.
- Do not force a normal or negative conclusion when the image does not support one.
- If the evidence is limited or uncertain, state the uncertainty conservatively.
- Keep the reasoning objective, concise, and clinically relevant.

[ANSWER RULES]
- The <answer> must directly answer the question.
- For multiple-choice questions, output only the final selected option.
- If the options are labeled A/B/C/D, prefer the single option letter only.
- Do not repeat the question.
- Do not include explanation inside <answer>.

[OUTPUT FORMAT]
Return the response strictly as:
<think>{your reasoning}</think>
<answer>{your final answer}</answer>
"""

PRECISE_PROMPT = """You are a medical imaging expert. Answer the given medical imaging task using the image only.

You must output your step-by-step reasoning inside <think>...</think>, followed by your final answer inside <answer>...</answer>.

[REASONING FORMAT (<think> block)]
1. First Line: You must explicitly state the dimension and question type. (e.g., "This is a [dimension] question about [question type].")
2. Objective Analysis: Identify the primary anatomical region. Objectively describe visual findings. Explicitly state whether queried features are present or absent based strictly on the image. DO NOT fabricate findings.
3. Conclusion: State your diagnostic impression.
Write in cohesive natural text. Do not use numbered lists.

[OUTPUT REQUIREMENT (<answer> block)]
- For multiple-choice questions: The <answer> MUST be STRICTLY the single chosen letter (A, B, C, or D) with absolutely no additional text or punctuation.
- For open-ended/report tasks: The <answer> MUST contain your full clinical text conclusion."""

CURRENT_INFERENCE_COT_PROMPT = PRECISE_PROMPT


def build_ct_slice_description(num_slices: int) -> str:
    return (
        f"This input contains {num_slices} grayscale CT slice images sampled in order from a single 3D CT study along one anatomical axis. "
        f"Each image is one slice from the same scan, and all slices should be interpreted jointly as one volumetric case."
    )


def build_mcq_cot_prompt(
    fixed_dimension: Optional[str] = None,
    fixed_question_type: Optional[str] = None,
) -> str:
    dimension_policy = ""
    question_type_policy = ""
    if fixed_dimension:
        dimension_policy = f"\nUse the fixed dimension: {fixed_dimension}\n"
    if fixed_question_type:
        question_type_policy = f"\nUse the fixed question_type: {fixed_question_type}\n"
    return MCQ_COT_PROMPT_TEMPLATE.format(
        dimension_policy=dimension_policy,
        question_type_policy=question_type_policy,
        qt0=MCQ_QUESTION_TYPES[0],
        qt1=MCQ_QUESTION_TYPES[1],
        qt2=MCQ_QUESTION_TYPES[2],
        qt3=MCQ_QUESTION_TYPES[3],
    )


def normalize_question_text(question: str) -> str:
    text = str(question or "")
    text = text.replace("<image>\n", " ").replace("<image>", " ")
    text = re.sub(r"\[Perform .*?\]\s*", "", text)
    text = re.sub(r"(?:<image>\s*)+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_report_task(question: str, task_type: Optional[str]) -> bool:
    task_type = str(task_type or "").lower()
    question = str(question or "").lower()
    return task_type in {"report", "report_generation"} or "report" in question


def is_multiple_choice_question(question: str, task_type: Optional[str]) -> bool:
    task_type = str(task_type or "").lower()
    question = str(question or "").lower()
    return task_type in {"multiple_choice", "mcq_vqa"} or "<multiple_choice>" in question

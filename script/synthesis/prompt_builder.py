from __future__ import annotations

from .schema import SynthesisTask


ALLOWED_QUESTION_TYPES = {
    'Rapid Triage': [
        'Existence Judgment (Presence/Absence)',
        'Single-Point Localization (Which Lobe/Segment/Side)',
        'Single-Point Measurement (Size/Thickness/Numerical Inquiry)',
        'Binary State (Yes/No; Normal/Abnormal; Solid/Ground-Glass)',
    ],
    'Standard Assessment': [
        'Multidimensional Comprehensive Description (Location + Morphology)',
        'Feature Description (Morphology, Density, Margins, Internal Structure)',
        'Anomaly Classification (Which subtype/pattern it belongs to)',
        'Regional Comparison (Left-to-Right Ratio, Superior-to-Inferior Ratio, Central-to-Peripheral Ratio)',
    ],
    'Comprehensive Diagnosis': [
        'Differential Diagnosis',
        'Primary Diagnosis Integration',
        'Severity/Risk Assessment',
        'Clinical Significance Integration',
        'Next-Step Recommendation',
    ],
}


EXPECTED_REASONING_LINES = {
    'Rapid Triage': 4,
    'Standard Assessment': 5,
    'Comprehensive Diagnosis': 7,
}

GLOBAL_PROMPT_CAUTION = """
[QUALITY GOAL]
Generate a training QA sample that is image-grounded, medically coherent, and information-dense.

[AVOID THESE FAILURES]
1. Do not produce repeated yes/no samples with near-identical wording.
2. Do not output generic answers if a subtype, distribution, or severity answer is possible.
3. Do not preserve factually wrong imaging wording from weak supervision.
4. Do not mention reports, metadata, prompt rules, JSON, or hidden labels.
5. Do not copy synthetic instruction text into the final reasoning.
6. If the source sample is weak or noisy, rewrite it into a cleaner and more clinically useful version while preserving the medically valid target.
""".strip()

REJECT_FORMAT = """
[IMPORTANT]
You are NOT allowed to invent a new question type.
You are NOT allowed to switch to another reasoning type.
If none of the allowed question types fit the source sample, return exactly one JSON object in this form:
{
  "status": "reject",
  "reason": "why this source sample does not fit any allowed question type"
}
Do not include q, reasoning, or a in a reject response.
""".strip()

OUTPUT_FORMAT = """
[OUTPUT FORMAT]
Return exactly one JSON object and nothing else.
If the sample is valid, return these keys:
- type
- dimension
- question_type
- q
- reasoning
- a

[STRICT REASONING FORMAT]
- reasoning MUST be a JSON array of strings.
- Do not return reasoning as one long paragraph.
- Do not add prose before or after the JSON object.
- The first reasoning item MUST be exactly:
  This is a [type] question about [question_type].
""".strip()


def build_category_guardrail(task: SynthesisTask) -> str:
    category = task.category or ''
    if category == 'Emphysema':
        return """
[EMPHYSEMA-SPECIFIC REQUIREMENTS]
1. Prefer clinically useful answers over generic presence/absence statements.
2. If supported by supervision, preserve subtype, location, distribution, or severity detail.
3. Avoid repetitive phrasing such as 'Yes, emphysematous changes are present in both lungs' unless that is the only valid content.
4. Do not use the phrase 'decrease in emphysematous density'.
5. If the source is a weak yes/no sample, rewrite toward more informative emphysema characterization when medically supported.
""".strip()
    if category == 'Medical material':
        return """
[MEDICAL-MATERIAL REQUIREMENTS]
1. Do not collapse the sample into a generic yes/no device-presence answer unless the source itself is truly only a yes/no presence task.
2. If the source names a concrete device, preserve the concrete device type and location.
3. Do not inject unrelated device names.
""".strip()
    if category == 'Coronary artery wall calcification':
        return """
[CORONARY CALCIFICATION REQUIREMENTS]
1. Keep the target focused on coronary artery wall calcification itself.
2. Do not broaden the answer into generic systemic atherosclerosis if the source is specifically coronary.
3. Preserve location/severity wording when the source supports it.
""".strip()
    if category == 'Cardiomegaly':
        return """
[CARDIOMEGALY REQUIREMENTS]
1. Do not return a one-word or ultra-short answer if a more image-grounded answer is possible.
2. Prefer answers that preserve the enlarged-cardiac-silhouette / heart-size finding explicitly.
""".strip()
    return ''


def _allowed_question_type_block(reasoning_type: str) -> str:
    values = ALLOWED_QUESTION_TYPES[reasoning_type]
    bullets = "\n".join(f'- {x}' for x in values)
    return (
        f"[FIXED REASONING TYPE]\n"
        f"You MUST use type = {reasoning_type}.\n\n"
        f"[ALLOWED QUESTION TYPES]\n"
        f"Choose question_type from this exact list only:\n{bullets}"
    )




def _reasoning_format_block(reasoning_type: str) -> str:
    n = EXPECTED_REASONING_LINES[reasoning_type]
    return (
        f"[STRICT REASONING ARRAY REQUIREMENT]\n"
        f"You MUST return reasoning as a JSON array with exactly {n} string items.\n"
        f"Item 1 must be exactly: This is a {reasoning_type} question about [chosen question_type].\n"
        f"Items 2-{n} must be concise image-grounded reasoning steps.\n"
        f"Do not return a paragraph. Do not return markdown. Do not return prose outside the JSON object."
    )

def build_open_prompt(task: SynthesisTask) -> str:
    reasoning_type = task.dimension or 'Rapid Triage'
    category_note = build_category_guardrail(task)
    allowed_block = _allowed_question_type_block(reasoning_type)
    reasoning_block = _reasoning_format_block(reasoning_type)
    return f"""
You are rewriting one open-ended medical imaging QA sample for model training.

{GLOBAL_PROMPT_CAUTION}

{allowed_block}

{reasoning_block}

{REJECT_FORMAT}

{OUTPUT_FORMAT}

[TASK]
1. Keep the question semantically aligned with the source sample.
2. Produce an answer that is concise but clinically informative.
3. Do not change the reasoning type.
4. Do not invent a new question_type.
5. Write structured reasoning that supports the final answer.
6. If the source question is weak, make it more specific while keeping the same clinical target.

{category_note}

[SOURCE QUESTION]
{task.question}

[SOURCE ANSWER]
{task.answer}

[SOURCE HINT REASONING]
{task.hint_reasoning}
""".strip()


def build_close_prompt(task: SynthesisTask) -> str:
    reasoning_type = task.dimension or 'Rapid Triage'
    category_note = build_category_guardrail(task)
    allowed_block = _allowed_question_type_block(reasoning_type)
    reasoning_block = _reasoning_format_block(reasoning_type)
    return f"""
You are rewriting one close-ended medical imaging QA sample for model training.

{GLOBAL_PROMPT_CAUTION}

{allowed_block}

{reasoning_block}

{REJECT_FORMAT}

{OUTPUT_FORMAT}

[TASK]
1. Keep the question and option structure semantically aligned with the source sample.
2. Keep exactly one correct option.
3. Keep the final answer normalized and consistent with the options.
4. Do not change the reasoning type.
5. Do not invent a new question_type.
6. Write structured reasoning that supports the final answer.
7. Avoid low-information duplicate yes/no style close-ended samples; rewrite the wording if needed while preserving the same correct option.

{category_note}

[SOURCE QUESTION]
{task.question}

[SOURCE ANSWER]
{task.answer}

[SOURCE HINT REASONING]
{task.hint_reasoning}
""".strip()


def build_report_prompt(task: SynthesisTask) -> str:
    reasoning_type = task.dimension or 'Comprehensive Diagnosis'
    allowed_block = _allowed_question_type_block(reasoning_type)
    reasoning_block = _reasoning_format_block(reasoning_type)
    return f"""
You are rewriting one report-style medical imaging sample for model training.

{GLOBAL_PROMPT_CAUTION}

{allowed_block}

{reasoning_block}

{REJECT_FORMAT}

{OUTPUT_FORMAT}

[TASK]
1. Preserve clinically grounded content.
2. Keep the output concise but informative.
3. Do not change the reasoning type.
4. Do not invent a new question_type.
5. Write structured reasoning that supports the final report-style answer.

[SOURCE QUESTION]
{task.question}

[SOURCE ANSWER]
{task.answer}

[SOURCE HINT REASONING]
{task.hint_reasoning}
""".strip()


def build_system_prompt(task: SynthesisTask) -> str:
    if task.task_type == 'mcq_vqa':
        return build_close_prompt(task)
    if task.task_type == 'report':
        return build_report_prompt(task)
    return build_open_prompt(task)


def build_user_content(task: SynthesisTask) -> str:
    return f"Question:\n{task.question}\n\nAnswer:\n{task.answer}\n\nHint reasoning:\n{task.hint_reasoning}"

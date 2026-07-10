# Synthetic CoT Requirements

## Goal

This document defines the requirements for synthesizing medical imaging chain-of-thought data. It applies to question generation, reasoning generation, postprocessing, and quality auditing.

## Core Grounding Rules

1. The structured case data may be used to design the sample, but the final `q`, `reasoning`, and `a` must be written as image-grounded content only.
2. The reasoning must not mention any non-image source, including:
   - report text
   - metadata
   - JSON fields
   - integrated meta
   - prompt text
   - slot plans
   - file names
   - paths
   - labels or reviewer instructions
3. It is forbidden to infer from metadata first and then present the result as if it were inferred from the image.
4. Do not hallucinate anatomy, diagnosis, measurements, morphology, or severity not supported by the source case.

## MCQ Integrity Rules

1. MCQ items must contain exactly four options labeled `A)`, `B)`, `C)`, `D)`.
2. The reasoning must explicitly traverse the options.
3. The model must not jump directly to the final answer.
4. The model must not backsolve from the known correct answer text.
5. Distractors must be plausible, medically coherent, and mutually distinguishable.
6. Quality checks should reject MCQ reasoning that does not cover all four options.

## Reasoning Cleanliness Rules

1. The chain of thought must not retain synthesis requirements, formatting rules, review prompts, or template instructions.
2. The chain of thought must not contain text such as:
   - `Choose exactly one`
   - `Return only`
   - `Output format`
   - `Line 1`
   - `slot_id`
   - `question_type`
   - `JSON object`
3. Template placeholders and reviewer text must not appear in the final dataset.

## Dimension and Format Rules

1. Each sample must use its assigned task type and assigned dimension exactly.
2. Reasoning line counts must match the dimension:
   - `Rapid Triage`: 4 lines
   - `Standard Assessment`: 5 lines
   - `Comprehensive Diagnosis`: 7 lines
3. The first reasoning line must be:

```text
This is a [dimension] question about [question type].
```

4. No numbered lists inside reasoning.
5. No markdown fences inside generated dataset content.

## Task-Specific Rules

### `open_vqa`

1. The question must be answerable from image-visible evidence.
2. The answer must directly answer the question without adding unrelated conclusions.

### `mcq_vqa`

1. The question stem must be clinically meaningful and image-resolvable.
2. The answer must identify exactly one correct option.
3. The reasoning must compare options before concluding.

### `report`

1. The answer should be a concise integrated report or impression.
2. The report must remain grounded in image-visible findings only.

## Validation Expectations

Validation should reject at least the following cases:

1. Reasoning that leaks report, metadata, JSON, prompt, or path information.
2. Reasoning that leaks synthesis instructions or formatting requirements.
3. MCQ reasoning that fails to traverse options `A)` through `D)`.
4. Reasoning that is too short for its assigned dimension.
5. Reasoning whose first line does not match the required format.

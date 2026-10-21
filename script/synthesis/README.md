# Synthesis Pipeline

## Goal
This folder contains a refactored synthesis pipeline for generating and
assembling training QA data with:

- source-specific task loading
- isolated filtering logic
- isolated prompt construction
- isolated model invocation
- isolated validation and serialization
- resumable orchestration
- final multi-source dataset assembly
- ratio control
- task4 category balancing
- versioned training-data folders

## High-Level Flow
1. Load raw tasks from source datasets.
2. Normalize tasks into a shared internal schema.
3. Apply source filters and category-specific quality rules.
4. Apply diversity controls for vulnerable categories such as `Emphysema`.
5. Build prompts using task-type-specific prompt contracts.
6. Run generation with a local LLM backend.
7. Parse structured outputs.
8. Validate the generated sample.
9. Write accepted samples, rejected samples, and audit logs.
10. Repeat until the requested target count is reached.
11. Assemble the final training set from three source datasets.
12. Enforce source/task-type ratios and task4 category minimums where possible.

## Why This Refactor
The old code mixed:

- task loading
- task filtering
- prompt construction
- model execution
- validation
- resume logic

inside one large script. That made it hard to change category-specific rules.

## Prompt Cautions
The current pipeline should avoid the following failure modes:

1. Do not overproduce near-duplicate yes/no QA for the same finding.
2. Do not preserve factually wrong supervision.
3. Do not mix irrelevant tasks into category-specific training subsets.
4. Do not force one rigid reasoning format for every task if it harms answer quality.

## Emphysema-Specific Issues To Address
Observed problems in the current dataset:

1. Too many semantically duplicated yes/no questions.
2. At least one sample contains factually wrong imaging wording.
3. Noisy cross-topic samples are mixed into the category.
4. Only a minority of samples contain useful subtype / distribution / severity information.

## Emphysema Policy In This Refactor
For `Emphysema`, the new pipeline uses the following policy:

1. Prefer diverse open QA over repeated yes/no phrasing.
2. Drop known noisy or off-topic samples.
3. Drop short generic answers when they do not add new information.
4. Prefer questions that capture:
   - subtype
   - location / distribution
   - severity
   - diagnostic interpretation

## Three-Source Final Dataset
The final dataset is assembled from three source groups:

- `ctrate_open`
- `m3d_mcq`
- `lidc_mixed`

The builder supports:

- source ratio control
- task-type ratio control
- task4 category minimum counts

Default ratios follow the historical mixed dataset:

- source counts
  - `ctrate_open = 854`
  - `m3d_mcq = 861`
  - `lidc_mixed = 710`
- task-type counts
  - `open_vqa = 1232`
  - `mcq_vqa = 981`
  - `report = 212`

## Versioned Training Data Layout
All new synthesis output should go under:

- `training_data/<source>/<version>/...`
- `training_data/mixed/<version>/...`

Example:

- `training_data/ctrate_open/v001/ctrate_open_cot.jsonl`
- `training_data/m3d_mcq/v001/m3d_mcq_cot.jsonl`
- `training_data/lidc_mixed/v001/medical_lidc_all_mixed_dataset_strict_cot_v2.jsonl`
- `training_data/mixed/v001/mixed_dataset.jsonl`

A single `version` groups the three source runs and the final mixed dataset.

## Deployment Policy
When synthesizing one source in one command, deploy the model once and keep it alive until that source finishes.
Do not intentionally loop with repeated model redeployments just because the count is slightly short.
For `lidc_mixed`, the wrapper uses a one-shot `case_limit` estimate that aims to slightly exceed the requested count, then the final mix trims to target.

## Main Entry Points
Generate only `ctrate_open` with the new open-task pipeline:

`python synthesize_dataset.py synthesize-ctrate-open ...`

Assemble a mixed dataset from existing source outputs:

`python synthesize_dataset.py build-mix ...`

Run versioned three-source synthesis and then build the mix:

`python synthesize_dataset.py run-version --version v001 --total-rows 600`

This command:
- computes per-source target counts from the old ratio
- creates `training_data/<source>/<version>` folders
- synthesizes only the needed amount for each source
- writes per-source manifests
- builds `training_data/mixed/<version>/mixed_dataset.jsonl`

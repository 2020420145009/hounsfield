# Dataset Synthesis Pipeline

This directory contains the **Observer-Synthesizer** pipeline for generating chain-of-thought (CoT) reasoning data from CT scans.

## Pipeline Overview

```
CT Scan → Observer → Medical Metadata → Synthesizer → CoT QA Pairs
```

### 1. Observer Module
Extracts structured medical metadata from CT scans:
- Nodule measurements (diameter, volume)
- Hounsfield Unit (HU) density
- Anatomical locations
- Scan parameters

### 2. Synthesizer Module  
Generates three types of QA pairs using LLM:
- **Open VQA**: Open-ended medical questions with detailed CoT reasoning
- **MCQ**: Multiple-choice questions with option-wise reasoning
- **Report**: Full diagnostic reports with structured analysis

## Usage

### Basic Synthesis

```bash
python synthesizer/synthesize_cot_dataset.py \
  --input_dir /path/to/ct_scans \
  --output_dir /path/to/output \
  --model_path /path/to/llm \
  --num_samples 1000
```

### Arguments

- `--input_dir`: Directory containing CT scan data with metadata
- `--output_dir`: Output directory for generated CoT dataset
- `--model_path`: Path to LLM for synthesis (e.g., DeepSeek-R1-Distill-Llama-70B)
- `--num_samples`: Number of QA pairs to generate per scan
- `--tensor_parallel_size`: Number of GPUs for model parallelism

### Output Format

The synthesizer generates JSONL files with the following structure:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion text"},
    {"from": "gpt", "value": "<think>\nChain-of-thought reasoning...\n</think>\nFinal answer"}
  ],
  "image": "path/to/ct_scan.nii.gz",
  "question_type": "open|mcq|report"
}
```

## Quality Control

The pipeline includes multiple quality checks:
- **Anatomical grounding**: All findings anchored to specific locations
- **Anti-robotic filters**: Removes template-like responses
- **Numerical precision**: Measurements rounded to 2 decimal places
- **Information density**: Ensures medical entity mentions in every sentence

## Examples

See `examples/` for sample inputs and outputs from the synthesis pipeline.

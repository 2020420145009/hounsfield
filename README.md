# Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/xxx)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/xxx)
[![Model](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/xxx)

This repository contains the official implementation of **Hounsfield-CoT**, a framework for advancing 3D medical visual question answering through chain-of-thought reasoning on CT scans.

## 📋 Overview

**Hounsfield-CoT** introduces:
- **Dataset**: 11.2k high-quality CoT reasoning instances synthesized from CT-RATE
- **Observer-Synthesizer Pipeline**: Automated pipeline for generating grounded medical reasoning
- **Hounsfield3D Model**: Fine-tuned vision-language model with enhanced 3D reasoning capabilities

## 🔥 Updates

- **2025.XX.XX**: Initial release of code and dataset

## 🗂️ Repository Structure

```
hounsfield/
├── dataset_synthesis/     # Observer-Synthesizer pipeline for CoT generation
│   ├── observer/         # Medical metadata extraction
│   ├── synthesizer/      # LLM-based CoT synthesis
│   └── utils/            # Data processing utilities
├── model/                # Hounsfield3D model implementation
│   ├── modeling/         # Model architecture
│   ├── training/         # Training scripts
│   └── inference/        # Inference utilities
├── benchmarks/           # Evaluation on 3D-RAD and other benchmarks
│   ├── 3D-RAD/          # 3D-RAD benchmark evaluation
│   └── evaluation/       # Evaluation scripts
├── docs/                 # Documentation
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/2020420145009/hounsfield.git
cd hounsfield

# Create conda environment
conda create -n hounsfield python=3.10
conda activate hounsfield

# Install dependencies
pip install -r requirements.txt
```

### Dataset Synthesis

Generate chain-of-thought reasoning data from CT scans:

```bash
cd dataset_synthesis
python synthesize_cot_dataset.py \
  --input_dir /path/to/ct_scans \
  --output_dir /path/to/output \
  --model_path /path/to/llm
```

### Model Training

Train Hounsfield3D on the synthesized CoT dataset:

```bash
cd model/training
bash train_hounsfield3d.sh \
  --data_path /path/to/cot_dataset \
  --model_path /path/to/base_model \
  --output_dir /path/to/checkpoints
```

### Evaluation

Evaluate on 3D-RAD benchmark:

```bash
cd benchmarks/3D-RAD
python evaluate.py \
  --model_path /path/to/hounsfield3d \
  --test_data /path/to/3drad_test \
  --output_dir /path/to/results
```

## 📊 Results

### Performance on 3D-RAD Benchmark

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Avg |
|-------|--------|--------|--------|--------|--------|--------|-----|
| Hounsfield3D | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |
| Hounsfield2D | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |
| M3D-RAD | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |

## 📦 Dataset

The Hounsfield-CoT dataset contains:
- **11.2k** training instances with detailed chain-of-thought reasoning
- Covers multiple question types: Open VQA, MCQ, Report Generation
- Grounded in anatomical metadata from CT-RATE

Download from [HuggingFace](https://huggingface.co/datasets/xxx).

## 🏗️ Model Architecture

Hounsfield3D extends Hounsfield2D with:
- 3D slice adapter for processing volumetric CT data
- Enhanced projector for 3D feature aggregation
- Fine-tuned on CoT reasoning data

## 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{hounsfield2026,
  title={Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning},
  author={},
  journal={arXiv preprint arXiv:xxx},
  year={2026}
}
```

## 🙏 Acknowledgments

- Dataset derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- Evaluated on [3D-RAD](https://github.com/Tang-xiaoxiao/3D-RAD) benchmark
- Built upon [InternVL](https://github.com/OpenGVLab/InternVL) architecture

## 📄 License

This project is released under the Apache 2.0 license. The CT-RATE source data is under CC-BY-NC-SA license.

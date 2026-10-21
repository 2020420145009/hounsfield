# Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning

[![Paper](https://img.shields.io/badge/Paper-BIBM%202024-red)](https://arxiv.org/abs/xxx)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/xxx)
[![Model](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/xxx)

> **Accepted at IEEE BIBM 2024** 🎉

This repository contains the official implementation of **Hounsfield-CoT**, a framework for advancing 3D medical visual question answering through chain-of-thought reasoning on CT scans.

## 📋 Overview

**Hounsfield-CoT** introduces:
- **Dataset**: 11.2k high-quality CoT reasoning instances synthesized from CT-RATE using LLM-based pipeline
- **Observer-Synthesizer Pipeline**: Automated framework combining medical metadata extraction with LLM reasoning synthesis
- **Hounsfield3D Model**: 3D-enhanced vision-language model achieving state-of-the-art performance on 3D-RAD benchmark

## 🔥 Updates

- **2024.12**: Paper accepted at IEEE BIBM 2024
- **2024.12**: Initial release of code and dataset

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

Our Hounsfield3D model achieves **state-of-the-art performance** on the 3D-RAD benchmark:

| Model | Task 1<br>(Plane) | Task 2<br>(Organ) | Task 3<br>(Abnormality) | Task 4<br>(Position) | Task 5<br>(Type) | Task 6<br>(Size) | **Average** |
|-------|--------|--------|--------|--------|--------|--------|-----|
| **Hounsfield3D** (Ours) | **99.24** | **77.59** | **59.04** | **95.19** | **83.85** | **82.59** | **82.92** |
| Hounsfield2D | 97.71 | 68.97 | 50.77 | 86.54 | 77.69 | 71.11 | 75.47 |
| M3D-LaMed | 94.66 | 70.69 | 53.85 | 84.62 | 73.08 | 67.41 | 74.05 |
| RadFM | 92.37 | 58.62 | 46.15 | 75.00 | 64.62 | 53.33 | 65.02 |

**Key Findings:**
- **+7.45 average improvement** over Hounsfield2D baseline through 3D enhancement
- **+17.87 average improvement** over previous SOTA (RadFM) 
- Particularly strong on spatial reasoning tasks (Task 4: Position, Task 6: Size)
- Benefits from CoT-enhanced training on 11.2k synthesized reasoning instances

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
@inproceedings{fang2024hounsfield,
  title={Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning},
  author={Fang, Ziyue and Wang, Shuyan and Miao, Bin and Zhong, Hui and Yu, Zhongtao},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024},
  organization={IEEE}
}
```

## 🙏 Acknowledgments

- Dataset derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- Evaluated on [3D-RAD](https://github.com/Tang-xiaoxiao/3D-RAD) benchmark
- Built upon [InternVL](https://github.com/OpenGVLab/InternVL) architecture

## 📄 License

This project is released under the Apache 2.0 license. The CT-RATE source data is under CC-BY-NC-SA license.

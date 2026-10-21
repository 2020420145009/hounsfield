# Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning

[![Paper](https://img.shields.io/badge/Paper-BIBM%202026-red)](https://arxiv.org/abs/xxx)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/xxx)
[![Model](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/xxx)

> **Accepted as Short Paper at IEEE BIBM 2026** 🎉

This repository contains the official implementation of **Hounsfield-CoT**, a framework for advancing 3D medical visual question answering through chain-of-thought reasoning on CT scans.

## 📋 Overview

**Hounsfield-CoT** introduces:
- **Dataset**: 11.2k high-quality CoT reasoning instances synthesized from CT-RATE using LLM-based pipeline
- **Observer-Synthesizer Pipeline**: Automated framework combining medical metadata extraction with LLM reasoning synthesis
- **Hounsfield3D Model**: 3D-enhanced vision-language model achieving state-of-the-art performance on 3D-RAD benchmark

## 🔥 Updates

- **2026.10**: Paper accepted as Short Paper at IEEE BIBM 2026
- **2026.10**: Initial release of code and dataset

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

### Performance on 3D-RAD and DeepChestVQA Benchmarks

Our **Hounsfield3D** model demonstrates substantial improvements after fine-tuning on the Hounsfield-CoT dataset:

#### 3D-RAD Benchmark (Open-Ended VQA)

| Model | T1 (Plane)<br>BERT-F1 / Acc | T2 (Organ)<br>BERT-F1 / Acc | T3 (Abnormality)<br>BERT-F1 / Acc | T4 (MCQ)<br>Acc |
|-------|--------|--------|--------|--------|
| M3D-RAD (Phi3-4B) | 0.922 / 46.0 | 0.907 / 41.9 | 0.948 / 15.4 | 82.3 |
| Qwen2.5-VL-7B | 0.650 / 24.7 | 0.624 / 18.3 | 0.444 / 12.3 | 38.3 |
| InternVL3-8B | 0.824 / 28.1 | 0.824 / 22.5 | 0.780 / 17.1 | 35.1 |
| Med3DVLM-Qwen-2.5-7B | 0.868 / 21.0 | 0.874 / 23.5 | 0.908 / 26.6 | 47.0 |
| HuatuoGPT-Vision-7B | 0.850 / 32.0 | 0.852 / 27.3 | 0.835 / 22.8 | 38.2 |
| **Hounsfield2D (Base)** | 0.841 / 7.2 | 0.841 / 7.7 | 0.846 / 3.2 | 77.8 |
| **Hounsfield3D (Ours, FT)** | **0.881 / 33.5** (+26.3↑) | **0.872 / 24.1** (+16.4↑) | **0.910 / 10.3** (+7.1↑) | **80.2** (+2.4↑) |

#### DeepChestVQA Zero-Shot Accuracy (%)

| Model | Recognition | Medical | Visual | Overall |
|-------|--------|--------|--------|--------|
| M3D-RAD (Phi3-4B) | 12.8 | 20.4 | 17.6 | 17.7 |
| Qwen2.5-VL-7B | 32.2 | 32.3 | 20.5 | 27.1 |
| Med3DVLM-Qwen-2.5-7B | 50.0 | 41.5 | 29.4 | 37.9 |
| HuatuoGPT-Vision-7B | 55.6 | 41.2 | 27.4 | 38.0 |
| **Hounsfield2D (Base)** | 51.1 | 39.7 | 26.6 | 36.4 |
| **Hounsfield3D (Ours, FT)** | **50.6** | **45.1** (+5.4↑) | **29.9** (+3.3↑) | **39.6** (+3.2↑) |

**Key Findings:**
- **Large accuracy gains** on 3D-RAD tasks: +26.3%, +16.4%, +7.1% on T1-T3, demonstrating the effectiveness of CoT reasoning
- **Improved zero-shot transfer** to DeepChestVQA: +3.2% overall accuracy, +5.4% on medical understanding
- **Strong 3D spatial reasoning** with enhanced feature aggregation from volumetric CT data
- Benefits from 11.2k high-quality CoT instances synthesized through Observer-Synthesizer pipeline

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
@inproceedings{fu2026hounsfield,
  title={Hounsfield-CoT: Advancing 3D Medical VQA with Chain-of-Thought Reasoning},
  author={Fu, Zhuoyuan and Li, Zeshang and Zhang, Yiqiong and Lin, Hangui and Shu, Yan and Li, Yan and Li, Binyang and Zhao, Yaru},
  booktitle={2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026},
  organization={IEEE},
  note={Short Paper}
}
```

## 🙏 Acknowledgments

- Dataset derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- Evaluated on [3D-RAD](https://github.com/Tang-xiaoxiao/3D-RAD) benchmark
- Built upon [InternVL](https://github.com/OpenGVLab/InternVL) architecture

## 📄 License

This project is released under the Apache 2.0 license. The CT-RATE source data is under CC-BY-NC-SA license.

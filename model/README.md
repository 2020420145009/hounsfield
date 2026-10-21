# Hounsfield3D Model

This directory contains the implementation of **Hounsfield3D**, a vision-language model specialized for 3D medical image understanding with chain-of-thought reasoning capabilities.

## Model Architecture

Hounsfield3D is built on top of Hounsfield2D (based on InternVL) and extends it with:

1. **3D Slice Adapter**: Processes multiple CT slices simultaneously
2. **Temporal Aggregation**: Combines information across slices
3. **Enhanced Projector**: Projects 3D visual features to language model space
4. **CoT Fine-tuning**: Trained on chain-of-thought reasoning data

### Architecture Diagram

```
CT Scan (32 slices)
    ↓
Vision Encoder (InternViT)
    ↓
3D Slice Adapter
    ↓
Projector (MLP)
    ↓
Language Model (Qwen)
    ↓
CoT Reasoning + Answer
```

## Components

### `modeling/`
- `modeling_fleming_3d.py`: Core model architecture
  - `Fleming3D_CoT_Model`: Main model class
  - Handles 3D CT input processing
  - Integrates vision encoder and language model

### `training/`
- `train_3d.py`: Training entry point
- `dataset_3d.py`: Dataset loader for CoT data
- `train_hounsfield3d.sh`: Training launcher script

### `inference/`
- Model inference utilities (to be added)
- Batch processing for evaluation
- Interactive demo scripts

## Training

### Quick Start

```bash
cd training
bash train_hounsfield3d.sh \
  /path/to/cot_dataset \
  /path/to/base_model \
  ./checkpoints
```

### Training Configuration

Key hyperparameters:
- **Learning rate**: 2e-5
- **Batch size**: 4 per GPU
- **Epochs**: 3
- **Max sequence length**: 2048
- **Precision**: bf16

### Multi-GPU Training

The training script supports distributed training:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 train_3d.py ...
```

## Model Checkpoint

Download the pre-trained Hounsfield3D model from [HuggingFace](https://huggingface.co/xxx).

## Inference

Example inference code:

```python
from modeling.modeling_fleming_3d import Fleming3D_CoT_Model

model = Fleming3D_CoT_Model(model_path="/path/to/hounsfield3d")
response = model.generate(ct_scan, question)
print(response)
```

## Performance

See the main README for benchmark results on 3D-RAD and other tasks.

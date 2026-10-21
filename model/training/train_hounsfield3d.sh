#!/bin/bash
# Training script for Hounsfield3D model

# Default arguments
DATA_PATH="${1:-/path/to/cot_dataset}"
MODEL_PATH="${2:-/path/to/base_model}"
OUTPUT_DIR="${3:-./checkpoints}"

# Training hyperparameters
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=2048

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

echo "=========================================="
echo "Training Hounsfield3D"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    train_3d.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_total_limit 3 \
    --bf16 \
    --tf32

echo "Training complete! Model saved to ${OUTPUT_DIR}"

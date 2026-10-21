# Benchmarks and Evaluation

This directory contains evaluation scripts for assessing Hounsfield3D on various 3D medical VQA benchmarks.

## 3D-RAD Benchmark

[3D-RAD](https://github.com/Tang-xiaoxiao/3D-RAD) is a comprehensive 3D radiology Med-VQA benchmark with six diverse tasks:

- **Task 1**: Anomaly Detection
- **Task 2**: Image Observation
- **Task 3**: Medical Computation
- **Task 4**: Existence Detection
- **Task 5**: Static Temporal Diagnosis
- **Task 6**: Longitudinal Temporal Diagnosis

### Running Evaluation

```bash
cd 3D-RAD
bash run_train_11_2k_judge.sh [conda_env] [judge_model] [tp_size] [max_num_seqs]
```

### LLM-as-a-Judge Evaluation

For open-ended questions, we use an LLM judge (DeepSeek-R1-Distill-Llama-70B) to evaluate response quality:

```bash
python llm_judge_open_results.py \
  --results-root /path/to/results \
  --tasks task1,task2,task3 \
  --judge-model /path/to/judge_model \
  --judge-backend vllm
```

### Metrics

- **Closed-ended tasks**: Accuracy, F1
- **Open-ended tasks**: Judge pass rate, judge score (0-100)
- **Medical computation**: Exact match, relative error

## Other Benchmarks

- **CT-SpatialVQA**: Spatial reasoning on CT scans
- **DeepChestVQA**: Chest X-ray and CT understanding  
- **FLARE26-MLLM-3D**: Multi-organ segmentation QA

## Adding New Benchmarks

To add a new benchmark:
1. Create a new directory under `benchmarks/`
2. Add dataset loading and evaluation scripts
3. Update this README with usage instructions

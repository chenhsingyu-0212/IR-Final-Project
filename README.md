# IR Final Project: Confidence-Based Retrieval-Augmented Generation

This project explores different approaches to Retrieval-Augmented Generation (RAG) with confidence-based retrieval decisions using the Mistral-7B-Instruct-v0.3 model.

## Overview

The project implements five experiments comparing various RAG strategies:

1. **Experiment 1**: Direct Answer Generation (Baseline)
2. **Experiment 2**: Fixed Retrieval (RAG)
3. **Experiment 3**: Binary Confidence-based Retrieval Decision (Yes/No)
4. **Experiment 4**: Calibrated Confidence-based Retrieval (Numerical Confidence)
5. **Experiment 5**: Fine-tuned Confidence Model for Retrieval Control

## Features

- **Model**: Mistral-7B-Instruct-v0.3
- **Retriever**: SentenceTransformer (all-MiniLM-L6-v2) + FAISS
- **Dataset**: Custom QA dataset (93,022 samples, split into 75% train / 25% test)
- **Evaluation**: Full test set evaluation for fair comparison

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd IR-FinalProject
```

2. Install dependencies:
```bash
pip install torch transformers sentence-transformers faiss-cpu tqdm
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

3. Download the dataset:
   - Place `single_answer_qa.json` in the project root
   - Run data splitting:
```bash
python split_dataset.py
```

## Dataset

- **Source**: `single_answer_qa.json` (93,022 QA pairs)
- **Train**: `train.json` (69,766 samples) - Used for retrieval corpus and fine-tuning
- **Test**: `test.json` (23,256 samples) - Used for evaluation across all experiments

## Experiments

### Experiment 1: Direct Generation
**File**: `experiment_1.py`
- Baseline: Direct answer generation without retrieval
- Output: `exp1_no_retrieval_custom_results.json`

### Experiment 2: Fixed Retrieval (RAG)
**File**: `experiment_2.py`
- Always performs retrieval before generation
- Uses top-3 similar documents from train set
- Output: `exp2_fixed_retrieval_custom_results.json`

### Experiment 3: Binary Confidence Retrieval
**File**: `experiment_3.py`
- Model outputs binary confidence (Yes/No)
- Retrieves only when confidence = "no"
- Output: `exp3_confidence_retrieval_custom_results.json`

### Experiment 4: Calibrated Confidence Retrieval
**File**: `experiment_4.py`
- Model outputs numerical confidence (0.0-1.0)
- Retrieves when confidence < 0.7
- Output: `exp4_calibrated_confidence_custom_results.json`

### Experiment 5: Fine-tuned Confidence Model
**Files**: `experiment_5_step1.py`, `experiment_5_step2.py`, `experiment_5_step3.py`

1. **Step 1**: Generate pseudo-labels from train set
   - Output: `exp5_confidence_train.jsonl`

2. **Step 2**: Fine-tune Mistral model with LoRA
   - Uses PEFT for efficient fine-tuning
   - Output: `exp5_finetuned_confidence/`

3. **Step 3**: Inference with fine-tuned model
   - Output: `exp5_finetuned_confidence_custom_results.json`

## Usage

### Running Individual Experiments

```bash
# Experiment 1
python experiment_1.py

# Experiment 2
python experiment_2.py

# Experiment 3
python experiment_3.py

# Experiment 4
python experiment_4.py

# Experiment 5
python experiment_5_step1.py  # Generate training data
python experiment_5_step2.py  # Fine-tune model
python experiment_5_step3.py  # Run inference
```

### Batch Run All Experiments

```bash
for exp in {1..4}; do python experiment_${exp}.py; done
python experiment_5_step1.py && python experiment_5_step2.py && python experiment_5_step3.py
```

## Results Format

All experiments output JSON files with unified structure:

```json
{
  "results": [
    {
      "question": "string",
      "gold_answer": "string",
      "predicted_answer": "string",
      "initial_answer": "string",  // null for exp1,2
      "confidence": "string|float|null",
      "used_retrieval": boolean,
      "retrieved_context": ["string"],
      "initial_time_sec": float,
      "rag_time_sec": float,
      "total_time_sec": float
    }
  ],
  "summary": {
    "total_samples": int,
    "total_time_sec": float,
    "average_time_sec": float,
    "average_initial_time_sec": float,
    "average_rag_time_sec": float
  }
}
```
## Inference

只計算並輸出到終端
```python evaluate_em_f1.py results.json```

計算並存成新檔
```python evaluate_em_f1.py results.json --output_file results_with_scores.json```


## Configuration

Key parameters in each experiment:

- `N_TRAIN`: Number of training samples for retrieval corpus (default: 500)
- `TOP_K`: Number of retrieved documents (default: 3)
- `CONF_THRESHOLD`: Confidence threshold for retrieval (exp3: 0.6, exp4: 0.7)
- `DEVICE`: "cuda" or "cpu"

## Dependencies

- `torch>=2.0.0`
- `transformers>=4.30.0`
- `sentence-transformers>=2.2.0`
- `faiss-cpu>=1.7.0` or `faiss-gpu`
- `tqdm`
- `peft` (for experiment 5)
- `accelerate` (for experiment 5)

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for Mistral-7B)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ for model weights and embeddings

## Troubleshooting

### CUDA Out of Memory
- Reduce `N_TRAIN` in experiments
- Use smaller batch sizes
- Switch to CPU mode: set `DEVICE = "cpu"`

### Model Loading Issues
- Ensure sufficient disk space for model cache (~14GB for Mistral-7B)
- Check internet connection for Hugging Face downloads

### FAISS Index Errors
- Ensure `faiss-cpu` or `faiss-gpu` is installed
- Check numpy version compatibility

## License

This project is for educational purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```
@misc{ir-final-project-2025,
  title={Confidence-Based Retrieval-Augmented Generation Experiments},
  author={Your Name},
  year={2025}
}
```
# IR Final Project: Confidence-Based Retrieval-Augmented Generation

This project explores different approaches to Retrieval-Augmented Generation (RAG) with confidence-based retrieval decisions using the Mistral-7B-Instruct-v0.3 model.

# IR Final Project: Confidence-Based Retrieval-Augmented Generation

This repository implements a set of experiments that explore confidence-driven Retrieval-Augmented Generation (RAG) strategies using the Mistral-7B-Instruct model family.

The codebase contains scripted experiments (1–6) that compare different retrieval triggers:

- Experiment 1: Direct generation (no retrieval)
- Experiment 2: Fixed RAG (always retrieve)
- Experiment 3: Logit-based confidence (use token probabilities / perplexity)
- Experiment 4: Calibrated confidence (model outputs a 0.0–1.0 confidence)
- Experiment 5: Fine-tuned logistic confidence (LoRA / PEFT)
- Experiment 6: Fine-tuned calibrated confidence (LoRA / PEFT)

## Repository layout

Top-level important files and folders:

- `1_experiment.py` … `6_step3_experiment.py` — experiment scripts
- `inference.py` — compute evaluation metrics / scores from results JSON
- `search.py` — DuckDuckGo-based retriever helpers
- `datasets/` — dataset files & utilities (`split_dataset.py`, `single_answer_qa.json`, `train.json`, `test.json`)
- `results/` — saved experiment outputs (e.g. `exp1.json`, `exp2.json`, `exp4.json`)
- `requirements.txt` — Python dependencies

## Quick features

- Model: `mistralai/Mistral-7B-Instruct-v0.3` (configurable in scripts)
- Retriever: DuckDuckGo full-page retrieval (via `search.py`)
- Dataset: Custom QA dataset in `datasets/` (split into `train.json` and `test.json`)

## Installation

Prerequisites:

- Python 3.8+
- (Optional) CUDA GPU for faster inference/training

Install dependencies (recommended to use a virtualenv):

```bash
pip install -r requirements.txt
```

If you need GPU PyTorch wheels, install the appropriate `torch` package from the PyTorch site first.

## Dataset preparation

Place `single_answer_qa.json` (original dataset) in `datasets/` and run the split utility:

```bash
python datasets/split_dataset.py
```

This produces `datasets/train.json` and `datasets/test.json` used by the experiments.

## Experiments — short descriptions & canonical filenames

- Experiment 1 — Direct Generation
   - File: `1_experiment.py`
   - Strategy: Pure generation (no retrieval)
   - Default output: `results/exp1.json`

- Experiment 2 — Fixed RAG
   - File: `2_experiment.py`
   - Strategy: Always retrieve (default top-k = 3)
   - Default output: `results/exp2.json`

- Experiment 3 — Logit-based Confidence
   - File: `3_experiment.py`
   - Strategy: Generate an initial answer, compute average token probability / perplexity from logits; retrieve if below threshold
   - Default output: `results/exp3.json`

- Experiment 4 — Calibrated Confidence
   - File: `4_experiment.py`
   - Strategy: Ask model to output a calibrated confidence (0.0–1.0); retrieve only when confidence < threshold (default 0.7)
   - Default output: `results/exp4.json`

- Experiment 5 — Fine-tuned Logistic Confidence (LoRA)
   - Files: `5_step1_experiment.py`, `5_step2_experiment.py`, `5_step3_experiment.py`
   - Workflow: generate pseudo-labels → LoRA fine-tune → inference using fine-tuned model
   - Outputs: training labels (`exp5_confidence_train.jsonl`), model checkpoints under `exp5_finetuned_confidence/`, inference outputs under `results/`

- Experiment 6 — Fine-tuned Calibrated Confidence (LoRA)
   - Files: `6_step1_experiment.py`, `6_step2_experiment.py`, `6_step3_experiment.py`
   - Workflow: similar to Exp 5 but target calibrated confidence values

Note: The README was cleaned up to match the actual script names and outputs present in the repository.

## Usage examples

Run individual experiments on a small sample (adjust `-n`):

```bash
python 1_experiment.py -n 50
python 2_experiment.py -n 50 -r 3
python 3_experiment.py -n 50 -c 0.7
python 4_experiment.py -n 50 -t 0.7 -r 3
```

Steps for fine-tuning experiments (Exp 5 / Exp 6):

```bash
# step1: generate pseudo-labels
python 5_step1_experiment.py -n 1000
# step2: fine-tune with LoRA (adjust epochs/batch as needed)
python 5_step2_experiment.py -e 1 -b 2
# step3: inference with the fine-tuned model
python 5_step3_experiment.py -n 50 -r 3
```

Batch run script:

```bash
# run_all_experiments.sh accepts a sample count (e.g. 50) or `all`
./run_all_experiments.sh 50
```

## Results format

All experiment outputs are JSON with a common structure. Example entry fields (may vary by experiment):

- `question`, `gold_answer`, `predicted_answer`
- `initial_answer` (present if experiment generates a first-pass answer)
- `confidence` or `confidence_logit`, `perplexity` (when applicable)
- `used_retrieval` (bool)
- `retrieved_context` (list or string)
- timing fields: `initial_time_sec`, `rag_time_sec`, `total_time_sec`

Top-level `summary` contains aggregated timing and counts.

## Inference / scoring

Quick terminal usage:

```bash
python inference.py results/exp4.json
# or save a scored copy
python inference.py results/exp4.json --output_file results/exp4_with_scores.json
```

## Configuration

Common parameters (exposed via CLI flags in scripts):

- `-n, --n_samples`: number of samples to run
- `-r`, `--num_results`: number of search results to retrieve
- `-t`, `--conf_threshold`: confidence threshold for retrieval
- `-o`, `--output_file`: output path for JSON

## Dependencies

See `requirements.txt` for the reproducible dependency list. Key packages used:

- torch, transformers, sentence-transformers (optional), faiss-cpu/faiss-gpu
- duckduckgo-search, beautifulsoup4, requests, tqdm
- datasets, accelerate, peft (for fine-tuning flows)

## Hardware guidance

- GPU recommended for model loading and inference (Mistral-7B is large)
- If GPU memory is limited, use smaller batch sizes and reduce `N_TRAIN` / sample counts

## Troubleshooting hints

- CUDA OOM: lower batch size, reduce `-n`, or use CPU
- model download failures: check internet and disk space for model cache
- FAISS issues: ensure `faiss-*` compatible with numpy

## Contributing

1. Fork
2. Create a feature branch
3. Make changes and test
4. Submit a PR

## License & Citation

This project is provided for educational use.

If you use this in research, cite:

```
@misc{ir-final-project-2025,
  title={Confidence-Based Retrieval-Augmented Generation Experiments},
  author={Your Name},
  year={2025}
}
```

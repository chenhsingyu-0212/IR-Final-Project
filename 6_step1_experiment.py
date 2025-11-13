#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6 Step 1: Fine-tuned Confidence (Calibrated)
Description: LoRA fine-tune model to predict calibrated confidence based on logits and answer correctness
Retrieval Trigger: Learned decision
- Step 1: Prepare fine-tuning dataset with calibrated confidence pseudo-labels (logit-based + correctness calibration)

Usage:
    python 6_step1_experiment.py                    # Process all train samples
    python 6_step1_experiment.py -n 1000           # Process first 1000 samples
    python 6_step1_experiment.py -o custom_train.jsonl  # Custom output file
"""

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse
import sys

def calculate_calibrated_confidence(question: str, answer: str, gold_answer: str) -> float:
    """Calculate calibrated confidence based on answer correctness and model uncertainty."""
    import torch.nn.functional as F

    # Create input text
    input_text = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

    # Calculate token probabilities
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    probs = F.softmax(shift_logits, dim=-1)
    token_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    avg_token_prob = token_probs.mean().item()

    # Calculate perplexity
    loss = outputs.loss.item() if outputs.loss is not None else 0.0
    perplexity = torch.exp(torch.tensor(loss)).item()

    # Check if answer is correct
    is_correct = gold_answer.lower().strip() in answer.lower().strip()

    # Calibrated confidence: higher if correct, adjusted by uncertainty
    base_confidence = avg_token_prob
    if is_correct:
        # Boost confidence for correct answers
        calibrated_conf = min(1.0, base_confidence + 0.3)
    else:
        # Reduce confidence for incorrect answers
        calibrated_conf = max(0.0, base_confidence - 0.4)

    # Further adjust by perplexity (lower perplexity = higher confidence)
    perplexity_factor = max(0.0, 1.0 - perplexity / 10.0)  # Normalize perplexity
    final_confidence = calibrated_conf * perplexity_factor

    return max(0.0, min(1.0, final_confidence))

def main():
    parser = argparse.ArgumentParser(description="Experiment 6 Step 1: Prepare calibrated confidence fine-tuning dataset")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp6_calibrated_confidence_train.jsonl",
                       help="Output file name")
    parser.add_argument("-d", "--data_file", type=str,
                       default="datasets/train.json",
                       help="Input data file name")

    args = parser.parse_args()

    # ========== CONFIG ==========
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    SRC_FILE = args.data_file
    OUT_FILE = args.output_file
    N_SAMPLES = args.n_samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 6 Step 1 with:")
    print(f"  Data file: {SRC_FILE}")
    print(f"  Output file: {OUT_FILE}")
    print(f"  Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD MODEL ==========
    print("Loading Mistral model...")
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # ========== LOAD TRAIN DATA ==========
    print(f"Loading data from {SRC_FILE}...")
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if N_SAMPLES:
        data = data[:N_SAMPLES]

    print(f"Processing {len(data)} samples...")

    count = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in data:
            q = s["question"]
            gold = str(s["answer"]).lower().strip()

            # Generate answer
            prompt = f"Answer the following question concisely:\n\nQuestion: {q}\nAnswer:"
            resp = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9, return_full_text=False)[0]["generated_text"]
            ans_raw = resp.strip()

            # Calculate calibrated confidence
            calibrated_conf = calculate_calibrated_confidence(q, ans_raw, gold)

            record = {
                "instruction": (
                    "Given a question and your generated answer, "
                    "predict your calibrated confidence (0.0–1.0) in correctness, "
                    "considering both answer quality and model uncertainty."
                ),
                "input": f"Question: {q}\nAnswer: {ans_raw}",
                "output": f"{calibrated_conf:.3f}",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"Processed {count} samples...")

    print(f"✅ Saved {count} calibrated confidence training samples → {OUT_FILE}")

if __name__ == "__main__":
    main()
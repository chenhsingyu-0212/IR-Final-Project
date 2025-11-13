#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Direct Answer
Description: Pure generation (no retrieval)
Retrieval Trigger: Never
Model: mistralai/Mistral-7B-Instruct-v0.3
Dataset: test.json

Usage:
    python 1_experiment.py                    # Run on full test set
    python 1_experiment.py -n 100            # Run on first 100 samples
    python 1_experiment.py -o custom_output.json  # Custom output file
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import json
import time
import statistics
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Direct Answer Generation")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp1.json",
                       help="Output file name")
    parser.add_argument("-d", "--data_file", type=str,
                       default="datasets/test.json",
                       help="Input data file name")

    args = parser.parse_args()

    # ========== CONFIG ==========
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    DATA_FILE = args.data_file
    OUTPUT_FILE = args.output_file
    N_SAMPLES = args.n_samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 1 with:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD MODEL ==========
    print("Loading Mistral model...")
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

    # ========== LOAD CUSTOM DATA ==========
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if N_SAMPLES:
        dataset = dataset[:N_SAMPLES]
        print(f"Using first {N_SAMPLES} samples")
    else:
        print(f"Using all {len(dataset)} samples")

    # ========== RUN EXPERIMENT ==========
    results = []
    times = []

    for sample in tqdm(dataset, desc="Generating answers"):
        question = sample["question"]
        gold_answer = sample["answer"]
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"

        start_time = time.perf_counter()
        output = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(elapsed)

        answer = output.split("Answer:")[-1].strip()

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": answer,
            "initial_answer": None,
            "confidence": None,
            "used_retrieval": False,
            "retrieved_context": [],
            "initial_time_sec": elapsed,
            "rag_time_sec": 0.0,
            "total_time_sec": elapsed,
        })

    # ========== 統計時間 ==========
    total_time = sum(times)
    avg_time = statistics.mean(times) if times else 0

    summary = {
        "total_samples": len(results),
        "total_time_sec": total_time,
        "average_time_sec": avg_time,
    }

    # ========== SAVE RESULTS ==========
    output_data = {
        "results": results,
        "summary": summary,
    }

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} results to {OUTPUT_FILE}")
    print(f"Total time: {total_time:.2f} sec | Average per sample: {avg_time:.2f} sec")


if __name__ == "__main__":
    main()

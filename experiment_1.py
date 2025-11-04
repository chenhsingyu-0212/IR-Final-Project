#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Direct Answer Generation (Custom Dataset)
Model: mistralai/Mistral-7B-Instruct-v0.3
Dataset: single_answer_qa.json
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import json
import time
import statistics

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_FILE = "test.json"
OUTPUT_FILE = "exp1_no_retrieval_custom_results.json"
N_SAMPLES = 100  # 可調整
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD MODEL ==========
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
with open(DATA_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Run on full test set

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

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} results to {OUTPUT_FILE}")
print(f"Total time: {total_time:.2f} sec | Average per sample: {avg_time:.2f} sec")

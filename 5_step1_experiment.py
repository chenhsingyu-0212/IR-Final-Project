#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5 Step 1: Fine-tuned Confidence (Logistic)
Description: LoRA-fine-tuned model to predict logit-based confidence from average token probability and perplexity
Retrieval Trigger: Learned decision
- Step 1: Prepare fine-tuning dataset with logit-based confidence pseudo-labels (matching exp3)

Usage:
    python 5_step1_experiment.py                    # Process all train samples
    python 5_step1_experiment.py -n 1000           # Process first 1000 samples
    python 5_step1_experiment.py -o custom_train.jsonl  # Custom output file
"""

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torch.nn.functional as F
import argparse
import sys

def compute_logit_confidence(model, tokenizer, prompt: str, answer: str, device="cuda"):
    """根據 logits 計算生成答案的平均 token 機率與困惑度 (與實驗三相同)"""
    full_input = prompt + " " + answer
    inputs = tokenizer(full_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["input_ids"]

    prompt_len = len(tokenizer(prompt)["input_ids"])
    answer_logits = logits[:, prompt_len - 1:-1, :]
    answer_labels = labels[:, prompt_len:]

    probs = F.softmax(answer_logits, dim=-1)
    token_probs = probs.gather(2, answer_labels.unsqueeze(-1)).squeeze(-1)

    avg_prob = token_probs.mean().item()
    avg_logprob = token_probs.log().mean().item()
    perplexity = torch.exp(-torch.tensor(avg_logprob)).item()

    return {
        "avg_prob": avg_prob,
        "avg_logprob": avg_logprob,
        "perplexity": perplexity,
    }

def main():
    parser = argparse.ArgumentParser(description="Experiment 5 Step 1: Prepare fine-tuning dataset")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp5_confidence_train.jsonl",
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

    print(f"Running Experiment 5 Step 1 with:")
    print(f"  Data file: {SRC_FILE}")
    print(f"  Output file: {OUT_FILE}")
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

    # ========== LOAD TRAIN DATA ==========
    print(f"Loading data from {SRC_FILE}...")
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if N_SAMPLES:
        data = data[:N_SAMPLES]
        print(f"Using first {N_SAMPLES} samples")
    else:
        print(f"Using all {len(data)} samples")

    count = 0

    print(f"Generating pseudo-labels with logit-based confidence...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in data:
            q = s["question"]
            gold = str(s["answer"]).lower().strip()

            # Generate answer (same prompt as exp3)
            prompt = (
                f"You are a precise reasoning assistant.\n"
                f"Answer the question clearly and concisely.\n\n"
                f"Question: {q}\nAnswer:"
            )

            resp = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
            ans_raw = resp.split("Answer:")[-1].strip()

            # Calculate logit-based confidence (same as exp3)
            logit_conf = compute_logit_confidence(model, tokenizer, prompt, ans_raw, device=DEVICE)
            conf = logit_conf["avg_prob"]

            # Check if answer is correct for pseudo-labeling
            pred = str(ans_raw).lower().strip()
            is_correct = gold in pred if gold else False

            # Use logit confidence as the target (regression task)
            # Higher confidence should correlate with correctness
            target_conf = conf if is_correct else max(0.0, conf - 0.3)  # Penalize wrong answers

            record = {
                "instruction": (
                    "Given a question and your generated answer, "
                    "predict your logit-based confidence (0.0–1.0) in correctness "
                    "based on token probabilities and perplexity."
                ),
                "input": f"Question: {q}\nAnswer: {ans_raw}",
                "output": f"{target_conf:.3f}",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"Processed {count} samples...")

    print(f"✅ Saved {count} logit-based confidence training samples → {OUT_FILE}")


if __name__ == "__main__":
    main()

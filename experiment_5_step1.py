#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare fine-tuning dataset for Exp5
Generate pseudo-labels from train.json using base model
Output: exp5_confidence_train.jsonl
"""

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SRC_FILE = "train.json"
OUT_FILE = "exp5_confidence_train.jsonl"
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
    device=0 if DEVICE == "cuda" else -1,
)

# ========== LOAD TRAIN DATA ==========
with open(SRC_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in data:
        q = s["question"]
        gold = str(s["answer"]).lower().strip()

        # Generate answer and confidence (similar to exp4)
        prompt = (
            f"You are a precise reasoning assistant.\n"
            f"Answer the question and provide your confidence score (0.0–1.0) "
            f"representing the probability that your answer is correct.\n\n"
            f"Question: {q}\n"
            f"Output format strictly as:\n"
            f"Answer: <text>\nConfidence: <float>\n"
        )

        resp = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]

        ans_raw = resp.split("Answer:")[-1].split("Confidence")[0].strip()
        pred = str(ans_raw).lower().strip()

        # Extract confidence (simple)
        conf_match = re.search(r"Confidence[:：]\s*([\d\.]+)", resp)
        conf = float(conf_match.group(1)) if conf_match else 0.5

        # Pseudo-label: 1.0 if gold in pred, else 0.0
        label = 1.0 if gold and gold in pred else 0.0

        record = {
            "instruction": (
                "Given a question and your generated answer, "
                "predict your confidence (0.0–1.0) in correctness."
            ),
            "input": f"Question: {q}\nAnswer: {ans_raw}",
            "output": str(label),
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Saved {count} pseudo-labeled samples → {OUT_FILE}")

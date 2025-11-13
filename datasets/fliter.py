#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter QA datasets for single-answer questions only.
Datasets: NaturalQuestions, WebQuestions, HotpotQA
"""

from datasets import load_dataset
import json

OUTPUT_FILE = "single_answer_qa.json"
MAX_SAMPLES = None  # set to e.g. 500 for testing

def extract_web_questions():
    ds = load_dataset("stanfordnlp/web_questions", split="train")
    items = []
    for row in ds:
        if isinstance(row["answers"], list) and len(row["answers"]) == 1:
            items.append({"source": "web_questions", "question": row["question"], "answer": row["answers"][0]})
    return items

def extract_hotpotqa():
    ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    items = []
    for row in ds:
        ans = row.get("answer")
        if isinstance(ans, str) and ans.strip():
            items.append({"source": "hotpotqa", "question": row["question"], "answer": ans.strip()})
    return items

def extract_nq():
    ds = load_dataset("natural_questions", split="train", trust_remote_code=True)
    items = []
    for row in ds:
        ans = row.get("annotations", {}).get("short_answers") if isinstance(row.get("annotations"), dict) else None
        if ans and isinstance(ans, list) and len(ans) == 1:
            text = ans[0].get("text")
            if isinstance(text, str):
                items.append({"source": "natural_questions", "question": row["question"]["text"], "answer": text})
    return items

if __name__ == "__main__":
    all_data = []
    all_data.extend(extract_web_questions())
    all_data.extend(extract_hotpotqa())
    try:
        all_data.extend(extract_nq())
    except Exception as e:
        print("NaturalQuestions skipped:", e)

    if MAX_SAMPLES:
        all_data = all_data[:MAX_SAMPLES]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_data)} single-answer QA pairs to {OUTPUT_FILE}")

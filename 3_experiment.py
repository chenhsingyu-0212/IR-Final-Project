#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3: Logit-based Confidence Retrieval + DuckDuckGo Search
Model: mistralai/Mistral-7B-Instruct-v0.3
Retriever: DuckDuckGo full-page retrieval
Dataset: test.json

Usage:
    python 3_experiment.py                    # Run on full test set
    python 3_experiment.py -n 100            # Run on first 100 samples
    python 3_experiment.py -o custom_output.json  # Custom output file
    python 3_experiment.py -c 0.8            # Set confidence threshold to 0.8
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import statistics
import random
import time
import argparse
import sys
from search import search_and_retrieve_fulltext, combine_documents

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Logit-based Confidence Retrieval + DuckDuckGo")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp3.json",
                       help="Output file name")
    parser.add_argument("-d", "--data_file", type=str,
                       default="datasets/test.json",
                       help="Input test data file name")
    parser.add_argument("-t", "--train_file", type=str,
                       default="datasets/train.json",
                       help="Input train data file name")
    parser.add_argument("-c", "--conf_threshold", type=float,
                       default=0.7,
                       help="Confidence threshold for retrieval (default: 0.7)")
    parser.add_argument("-nt", "--n_train", type=int,
                       default=500,
                       help="Number of training samples to use (default: 500)")

    args = parser.parse_args()

    # ========== CONFIG ==========
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    DATA_FILE = args.data_file
    TRAIN_FILE = args.train_file
    OUTPUT_FILE = args.output_file
    CONF_THRESHOLD = args.conf_threshold
    N_TRAIN = args.n_train
    N_SAMPLES = args.n_samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 3 with:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Train file: {TRAIN_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  Training samples: {N_TRAIN}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD DATA ==========
    print(f"Loading train data from {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loading test data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    trainset = train_data[:N_TRAIN]
    evalset = test_data

    if N_SAMPLES:
        evalset = evalset[:N_SAMPLES]
        print(f"Using first {N_SAMPLES} test samples")
    else:
        print(f"Using all {len(evalset)} test samples")

    # ========== 載入模型 ==========
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

    # ========== HELPER FUNCTIONS ==========
    def compute_logit_confidence(model, tokenizer, prompt: str, answer: str, device="cuda"):
        """根據 logits 計算生成答案的平均 token 機率與困惑度"""
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

    # ========== MAIN EXPERIMENT ==========
    results = []
    times_total = []
    times_initial = []
    times_rag = []

    for s in tqdm(evalset, desc="Logit-based Confidence + DuckDuckGo RAG"):
        q = s["question"]
        gold_answer = s["answer"]

        # Step 1: 初步生成答案（不要求信心分數）
        prompt = (
            f"You are a precise reasoning assistant.\n"
            f"Answer the question clearly and concisely.\n\n"
            f"Question: {q}\nAnswer:"
        )

        t0 = time.perf_counter()
        resp1 = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
        t1 = time.perf_counter()
        elapsed_initial = t1 - t0

        # 解析生成的答案
        ans_raw = resp1.split("Answer:")[-1].strip()

        # 計算 logit-based confidence
        logit_conf = compute_logit_confidence(model, tokenizer, prompt, ans_raw, device=DEVICE)
        conf = logit_conf["avg_prob"]

        used_retrieval = False
        final_answer = ans_raw
        elapsed_rag = 0.0
        retrieved_context = ""

        # Step 2: 若 logit-based confidence < 閾值 → 啟用 DuckDuckGo 檢索
        if conf < CONF_THRESHOLD:
            used_retrieval = True

            # 🔍 檢索英文全文
            retrieved_docs = search_and_retrieve_fulltext(q, num_results=5)
            retrieved_context = combine_documents(retrieved_docs)

            prompt_rag = (
                f"Your previous answer had low confidence ({conf:.2f}). "
                f"Use the retrieved information to improve your response.\n\n"
                f"Retrieved context:\n{retrieved_context}\n\n"
                f"Question: {q}\nAnswer:"
            )

            t2 = time.perf_counter()
            resp2 = generator(prompt_rag, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
            t3 = time.perf_counter()
            elapsed_rag = t3 - t2

            final_answer = resp2.split("Answer:")[-1].strip()

        total_time = elapsed_initial + elapsed_rag
        times_total.append(total_time)
        times_initial.append(elapsed_initial)
        times_rag.append(elapsed_rag)

        results.append({
            "question": q,
            "gold_answer": gold_answer,
            "initial_answer": ans_raw,
            "predicted_answer": final_answer,
            "confidence_logit": float(conf),
            "perplexity": float(logit_conf["perplexity"]),
            "used_retrieval": used_retrieval,
            "retrieved_context": retrieved_context[:1000],  # 儲存前段文字避免檔案過大
            "initial_time_sec": elapsed_initial,
            "rag_time_sec": elapsed_rag,
            "total_time_sec": total_time,
        })

    # ========== 統計 ==========
    summary = {
        "total_samples": len(results),
        "total_time_sec": sum(times_total),
        "average_time_sec": statistics.mean(times_total),
        "average_initial_time_sec": statistics.mean(times_initial),
        "average_rag_time_sec": statistics.mean([t for t in times_rag if t > 0]) if any(times_rag) else 0.0,
        "avg_confidence_logit": statistics.mean([r["confidence_logit"] for r in results]),
        "conf_threshold": CONF_THRESHOLD,
    }

    # ========== SAVE ==========
    output_data = {
        "results": results,
        "summary": summary,
    }

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} results to {OUTPUT_FILE}")
    print(
        f"Total time: {summary['total_time_sec']:.2f} sec | "
        f"Avg total: {summary['average_time_sec']:.2f} sec | "
        f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
        f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec | "
        f"Avg logit-conf: {summary['avg_confidence_logit']:.3f} | "
        f"Conf threshold: {CONF_THRESHOLD}"
    )


if __name__ == "__main__":
    main()
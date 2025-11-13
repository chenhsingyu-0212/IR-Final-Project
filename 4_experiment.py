#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4: Calibrated Confidence
Description: Output numeric confidence [0–1]; retrieve if < 0.7
Retrieval Trigger: Conditional (threshold)
Model: mistralai/Mistral-7B-Instruct-v0.3
Retriever: DuckDuckGo full-page retrieval
Dataset: test.json

Usage:
    python 4_experiment.py                    # Run on full test set
    python 4_experiment.py -n 100            # Run on first 100 samples
    python 4_experiment.py -t 0.7            # Set confidence threshold to 0.7 (default)
    python 4_experiment.py -r 3              # Use 3 search results
    python 4_experiment.py -o custom_output.json  # Custom output file
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
import re
import time
import statistics
import random
import argparse
import sys
from search import search_and_retrieve_fulltext, combine_documents

def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Calibrated Confidence-based Retrieval")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-t", "--conf_threshold", type=float, default=0.7,
                       help="Confidence threshold for retrieval (default: 0.7)")
    parser.add_argument("-r", "--num_results", type=int, default=5,
                       help="Number of DuckDuckGo search results (default: 5)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp4.json",
                       help="Output file name")
    parser.add_argument("-d", "--data_file", type=str,
                       default="datasets/test.json",
                       help="Input data file name")

    args = parser.parse_args()

    # ========== CONFIG ==========
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    DATA_FILE = args.data_file
    OUTPUT_FILE = args.output_file
    NUM_RESULTS = args.num_results
    CONF_THRESHOLD = args.conf_threshold
    N_SAMPLES = args.n_samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 4 with:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  Search results: {NUM_RESULTS}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD DATA ==========
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 直接使用原始順序以確保重現性
    evalset = test_data[:N_SAMPLES] if N_SAMPLES else test_data
    print(f"Using {len(evalset)} samples")

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

    # ========== HELPER ==========
    def extract_confidence(text: str) -> float:
        """Extract normalized confidence score (0.0–1.0) from model output."""
        patterns = [
            r"Confidence[:=]\s*([\d\.]+)",
            r"([\d\.]+)\s*%",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                val = float(m.group(1))
                if val > 1:
                    val /= 100
                return max(0.0, min(1.0, val))
        return 0.0

    # ========== EXPERIMENT ==========
    results = []
    times_total = []
    times_initial = []
    times_rag = []

    for s in tqdm(evalset, desc="Calibrated Confidence RAG"):
        q = s["question"]
        gold_answer = s["answer"]

        # Step 1: 初步回答 + 校準信心
        prompt = (
            f"You are a precise reasoning assistant.\n"
            f"Answer the question and provide your confidence score (0.0–1.0) "
            f"representing the probability that your answer is correct.\n\n"
            f"Question: {q}\n"
            f"Output format strictly as:\n"
            f"Answer: <text>\nConfidence: <float>\n"
        )

        t0 = time.perf_counter()
        resp1 = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
        t1 = time.perf_counter()

        ans_raw = resp1.split("Answer:")[-1].split("Confidence")[0].strip()
        conf = extract_confidence(resp1)
        elapsed_initial = t1 - t0

        used_retrieval = False
        final_answer = ans_raw
        elapsed_rag = 0.0
        retrieved_context = []

        # Step 2: 若信心低於閾值 → 檢索輔助
        if conf < CONF_THRESHOLD:
            used_retrieval = True
            retrieved_docs = search_and_retrieve_fulltext(q, num_results=NUM_RESULTS)
            context = combine_documents(retrieved_docs)

            prompt_rag = (
                f"You were only {conf:.2f} confident in your previous answer.\n"
                f"Use the retrieved context to improve your response.\n\n"
                f"Context:\n{context}\n\nQuestion: {q}\n"
                f"Answer:"
            )

            t2 = time.perf_counter()
            resp2 = generator(prompt_rag, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
            t3 = time.perf_counter()
            elapsed_rag = t3 - t2

            final_answer = resp2.split("Answer:")[-1].strip()
            retrieved_context = [f"{d['title']}: {d['content'][:500]}..." for d in retrieved_docs]

        total_time = elapsed_initial + elapsed_rag
        times_total.append(total_time)
        times_initial.append(elapsed_initial)
        times_rag.append(elapsed_rag)

        results.append({
            "question": q,
            "gold_answer": gold_answer,
            "predicted_answer": final_answer,
            "initial_answer": ans_raw,
            "confidence": float(conf),
            "used_retrieval": used_retrieval,
            "retrieved_context": retrieved_context,
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
        f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
    )


if __name__ == "__main__":
    main()

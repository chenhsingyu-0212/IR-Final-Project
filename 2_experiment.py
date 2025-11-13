#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: Fixed RAG
Description: Always retrieve top-3 docs
Retrieval Trigger: Always
Model: mistralai/Mistral-7B-Instruct-v0.3
Retriever: DuckDuckGo full-page retrieval
Dataset: test.json

Usage:
    python 2_experiment.py                    # Run on full test set
    python 2_experiment.py -n 100            # Run on first 100 samples
    python 2_experiment.py -r 3              # Use 3 search results (default)
    python 2_experiment.py -o custom_output.json  # Custom output file
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
import time
import statistics
import random
import argparse
import sys
from search import search_and_retrieve_fulltext, combine_documents

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Fixed Retrieval (RAG) with DuckDuckGo")
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("-r", "--num_results", type=int, default=3,
                       help="Number of DuckDuckGo search results (default: 3)")
    parser.add_argument("-o", "--output_file", type=str,
                       default="results/exp2.json",
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
    N_SAMPLES = args.n_samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 2 with:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  Search results: {NUM_RESULTS}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD DATA ==========
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 直接使用原始順序以確保重現性
    evalset = test_data[:N_SAMPLES] if N_SAMPLES else test_data
    print(f"Using {len(evalset)} samples")

    # ========== LOAD LLM ==========
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

    # ========== RAG GENERATION ==========
    results = []
    times = []

    for sample in tqdm(evalset, desc="RAG generation with DuckDuckGo"):
        q = sample["question"]
        gold_answer = sample["answer"]

        # 檢索階段計時
        retrieval_start = time.perf_counter()
        retrieved_docs = search_and_retrieve_fulltext(q, num_results=NUM_RESULTS)
        retrieval_end = time.perf_counter()
        retrieval_time = retrieval_end - retrieval_start

        # 組合檢索到的內容
        context = combine_documents(retrieved_docs)

        # 組合 Prompt
        prompt = (
            f"Use the following retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\nAnswer:"
        )

        # 推論階段計時
        start_time = time.perf_counter()
        output = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(retrieval_time + elapsed)

        answer = output.split("Answer:")[-1].strip()

        results.append({
            "question": q,
            "gold_answer": gold_answer,
            "predicted_answer": answer,
            "initial_answer": None,
            "confidence": None,
            "used_retrieval": True,
            "retrieved_context": [f"{d['title']}: {d['content'][:500]}..." for d in retrieved_docs],
            "initial_time_sec": retrieval_time,
            "rag_time_sec": elapsed,
            "total_time_sec": retrieval_time + elapsed,
        })

    # ========== 統計時間 ==========
    total_time = sum(times)
    avg_time = statistics.mean(times)

    # 計算檢索與生成時間
    retrieval_times = [r["initial_time_sec"] for r in results]
    rag_times = [r["rag_time_sec"] for r in results]

    summary = {
        "total_samples": len(results),
        "total_time_sec": total_time,
        "average_time_sec": avg_time,
        "average_initial_time_sec": statistics.mean(retrieval_times),
        "average_rag_time_sec": statistics.mean(rag_times),
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

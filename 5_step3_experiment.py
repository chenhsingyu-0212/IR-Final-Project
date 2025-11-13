#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5 Step 3: Fine-tuned Confidence (Logistic)
Description: LoRA-fine-tuned model to predict confidence from average token probability and perplexity
Retrieval Trigger: Learned decision
- Step 3: Run inference with fine-tuned model

Usage:
    python 5_step3_experiment.py                    # Run inference with default settings
    python 5_step3_experiment.py -n 100            # Test on first 100 samples
    python 5_step3_experiment.py -c 0.8            # Set confidence threshold to 0.8
    python 5_step3_experiment.py -o custom_results.json  # Custom output file
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, torch, re, time, statistics, random
from tqdm import tqdm
from search import search_and_retrieve_fulltext, combine_documents

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned confidence model and DuckDuckGo retrieval")
    parser.add_argument("-m", "--model_dir", type=str, default="results/exp5_finetuned_confidence",
                       help="Directory of fine-tuned model")
    parser.add_argument("-d", "--data_file", type=str, default="datasets/test.json",
                       help="Path to test data JSON file")
    parser.add_argument("-t", "--train_file", type=str, default="datasets/train.json",
                       help="Path to train data JSON file")
    parser.add_argument("-o", "--output_file", type=str, default="results/exp5.json",
                       help="Output file for results")
    parser.add_argument("-r", "--num_results", type=int, default=5,
                       help="Number of DuckDuckGo search results")
    parser.add_argument("-c", "--conf_threshold", type=float, default=0.7,
                       help="Confidence threshold for retrieval (0.0-1.0)")
    parser.add_argument("-n", "--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== LOAD DATA ==========
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(args.data_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 直接使用原始順序以確保重現性
    evalset = test_data[:args.num_samples] if args.num_samples else test_data

    # ========== LOAD MODEL ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

    # ========== HELPER ==========
    def parse_conf(t: str) -> float:
        """Extract confidence score from text (0.0–1.0 range)."""
        m = re.search(r"([\d\.]+)", t)
        if not m:
            return 0.0
        v = float(m.group(1))
        if v > 1:
            v /= 100
        return max(0.0, min(1.0, v))

    # ========== EXPERIMENT ==========
    results = []
    times_total, times_initial, times_rag = [], [], []

    for s in tqdm(evalset, desc="Exp5 Inference (custom)"):
        q = s["question"]
        gold_answer = s["answer"]

        # Step 1: 產生初步回答與信心分數
        p = f"Answer the question and output your confidence (0.0–1.0).\nQuestion: {q}\nAnswer:"

        t0 = time.perf_counter()
        out = gen(p, max_new_tokens=128, temperature=0.2)[0]["generated_text"]
        t1 = time.perf_counter()
        elapsed_initial = t1 - t0

        initial_ans = out.split("Answer:")[-1].split("Confidence")[0].strip()
        conf = parse_conf(out)
        used = False
        elapsed_rag = 0.0
        retrieved_context = []

        # Step 2: 若信心不足則進行檢索
        ans = initial_ans  # default: use initial answer when no retrieval
        if conf < args.conf_threshold:
            used = True
            retrieved_docs = search_and_retrieve_fulltext(q, num_results=args.num_results)
            ctx = combine_documents(retrieved_docs)

            p2 = f"Use the retrieved context to refine your answer.\nContext:\n{ctx}\nQuestion: {q}\nAnswer:"
            t2 = time.perf_counter()
            out2 = gen(p2, max_new_tokens=128, temperature=0.2)[0]["generated_text"]
            t3 = time.perf_counter()
            elapsed_rag = t3 - t2

            ans = out2.split("Answer:")[-1].strip()
            retrieved_context = [f"{d['title']}: {d['content'][:500]}..." for d in retrieved_docs]

        total_time = elapsed_initial + elapsed_rag
        times_initial.append(elapsed_initial)
        times_rag.append(elapsed_rag)
        times_total.append(total_time)

        results.append({
            "question": q,
            "gold_answer": gold_answer,
            "predicted_answer": ans,
            "initial_answer": initial_ans,
            "confidence": float(conf),
            "used_retrieval": used,
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
    output_data = {"results": results, "summary": summary}

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {args.output_file}")
    print(
        f"Total time: {summary['total_time_sec']:.2f} sec | "
        f"Avg total: {summary['average_time_sec']:.2f} sec | "
        f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
        f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
    )

if __name__ == "__main__":
    main()

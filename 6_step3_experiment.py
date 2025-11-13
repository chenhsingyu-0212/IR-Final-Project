#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6 Step 3: Fine-tuned Confidence (Calibrated)
Description: LoRA fine-tune model to predict calibrated confidence based on logits and answer correctness
Retrieval Trigger: Learned decision
- Step 3: Run inference with fine-tuned calibrated confidence model

Usage:
    python 6_step3_experiment.py                    # Run inference with default settings
    python 6_step3_experiment.py -n 100            # Test on first 100 samples
    python 6_step3_experiment.py -c 0.7            # Set confidence threshold to 0.7
    python 6_step3_experiment.py -o custom_results.json  # Custom output file
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, torch, re, time, statistics
from tqdm import tqdm
from search import search_and_retrieve_fulltext, combine_documents

def extract_calibrated_confidence(text: str) -> float:
    """Extract calibrated confidence score from model output."""
    patterns = [
        r"Confidence[:：]\s*([\d\.]+)",
        r"([\d\.]+)\s*$",  # Last number in response
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                continue
    return 0.5  # Default confidence

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned calibrated confidence model")
    parser.add_argument("-m", "--model_dir", type=str, default="results/exp6_finetuned_calibrated_confidence",
                       help="Directory of fine-tuned model")
    parser.add_argument("-d", "--data_file", type=str, default="datasets/test.json",
                       help="Path to test data JSON file")
    parser.add_argument("-o", "--output_file", type=str, default="results/exp6.json",
                       help="Output file for results")
    parser.add_argument("-r", "--num_results", type=int, default=3,
                       help="Number of search results")
    parser.add_argument("-c", "--conf_threshold", type=float, default=0.7,
                       help="Confidence threshold for retrieval (0.0-1.0)")
    parser.add_argument("-n", "--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Experiment 6 Step 3 with:")
    print(f"  Model: {args.model_dir}")
    print(f"  Data: {args.data_file}")
    print(f"  Output: {args.output_file}")
    print(f"  Samples: {args.num_samples if args.num_samples else 'all'}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Device: {DEVICE}")

    # ========== LOAD DATA ==========
    with open(args.data_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 直接使用原始順序以確保重現性
    evalset = test_data[:args.num_samples] if args.num_samples else test_data
    print(f"Using {len(evalset)} samples")

    # ========== LOAD MODEL ==========
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    
    if DEVICE == "cuda":
        model = model.to("cuda")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # ========== EXPERIMENT LOOP ==========
    results = []
    times_total = []
    times_initial = []
    times_rag = []

    for s in tqdm(evalset, desc="Fine-tuned Calibrated Confidence RAG"):
        q = s["question"]
        gold_answer = s["answer"]

        # Step 1: Generate answer and calibrated confidence
        prompt = (
            f"Given a question and your generated answer, "
            f"predict your calibrated confidence (0.0–1.0) in correctness, "
            f"considering both answer quality and model uncertainty.\n\n"
            f"Question: {q}\nAnswer:"
        )

        t0 = time.perf_counter()
        resp1 = generator(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9, return_full_text=False)[0]["generated_text"]
        t1 = time.perf_counter()

        # Extract answer and confidence
        full_response = resp1.strip()
        if "Answer:" in full_response:
            ans_raw = full_response.split("Answer:")[-1].split("Confidence:")[0].strip()
        else:
            ans_raw = full_response

        calibrated_conf = extract_calibrated_confidence(full_response)
        elapsed_initial = t1 - t0

        used_retrieval = False
        final_answer = ans_raw
        elapsed_rag = 0.0
        retrieved_context = []

        # Step 2: Retrieve if calibrated confidence below threshold
        if calibrated_conf < args.conf_threshold:
            used_retrieval = True
            retrieved_docs = search_and_retrieve_fulltext(q, num_results=args.num_results)
            context = combine_documents(retrieved_docs)
            retrieved_context = [f"{d['title']}: {d['content'][:500]}..." for d in retrieved_docs]

            prompt_rag = (
                f"Your calibrated confidence was {calibrated_conf:.3f}. "
                f"Use the retrieved context to improve your answer.\n\n"
                f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
            )

            t2 = time.perf_counter()
            resp2 = generator(prompt_rag, max_new_tokens=128, temperature=0.2, top_p=0.9, return_full_text=False)[0]["generated_text"]
            t3 = time.perf_counter()

            final_answer = resp2.strip()
            elapsed_rag = t3 - t2

        total_time = elapsed_initial + elapsed_rag
        times_total.append(total_time)
        times_initial.append(elapsed_initial)
        times_rag.append(elapsed_rag)

        results.append({
            "question": q,
            "gold_answer": gold_answer,
            "predicted_answer": final_answer,
            "initial_answer": ans_raw,
            "confidence": calibrated_conf,
            "used_retrieval": used_retrieval,
            "retrieved_context": retrieved_context,
            "initial_time_sec": elapsed_initial,
            "rag_time_sec": elapsed_rag,
            "total_time_sec": total_time,
        })

    # ========== STATISTICS ==========
    summary = {
        "total_samples": len(results),
        "total_time_sec": sum(times_total),
        "average_time_sec": statistics.mean(times_total),
        "average_initial_time_sec": statistics.mean(times_initial),
        "average_rag_time_sec": statistics.mean([t for t in times_rag if t > 0]) if any(times_rag) else 0.0,
        "retrieval_rate": sum(1 for r in results if r["used_retrieval"]) / len(results),
    }

    # ========== SAVE RESULTS ==========
    output_data = {
        "results": results,
        "summary": summary,
    }

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} results to {args.output_file}")
    print(f"Total time: {summary['total_time_sec']:.2f} sec | "
          f"Avg total: {summary['average_time_sec']:.2f} sec | "
          f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
          f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec | "
          f"Retrieval rate: {summary['retrieval_rate']:.2%}")

if __name__ == "__main__":
    main()
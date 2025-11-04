#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4: Calibrated Confidence-based Retrieval (Numerical Confidence)
Model: mistralai/Mistral-7B-Instruct-v0.3
Retriever: SentenceTransformer + FAISS
Dataset: single_answer_qa.json
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import json
from tqdm import tqdm
import re
import time
import statistics
import random

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = "test.json"
TRAIN_FILE = "train.json"
OUTPUT_FILE = "exp4_calibrated_confidence_custom_results.json"
TOP_K = 3
CONF_THRESHOLD = 0.7  # calibrated threshold
N_TRAIN = 500
N_EVAL = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD DATA ==========
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

random.shuffle(train_data)
random.shuffle(test_data)

trainset = train_data[:N_TRAIN]
evalset = test_data  # Run on full test set

# ========== 建立檢索知識庫 ==========
corpus_texts = [f"Q: {x['question']}\nA: {x['answer']}" for x in trainset]
embedder = SentenceTransformer(EMBED_MODEL)
corpus_embeds = embedder.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(corpus_embeds)
index = faiss.IndexFlatIP(corpus_embeds.shape[1])
index.add(corpus_embeds)

# ========== 載入模型 ==========
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
        q_embed = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_embed)
        D, I = index.search(q_embed, TOP_K)
        retrieved_context = [corpus_texts[i] for i in I[0]]
        context = "\n\n".join(retrieved_context)

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

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} results to {OUTPUT_FILE}")
print(
    f"Total time: {summary['total_time_sec']:.2f} sec | "
    f"Avg total: {summary['average_time_sec']:.2f} sec | "
    f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
    f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
)

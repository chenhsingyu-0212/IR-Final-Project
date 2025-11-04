#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: Fixed Retrieval (RAG)
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
import time
import statistics
import random

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = "test.json"
TRAIN_FILE = "train.json"
OUTPUT_FILE = "exp2_fixed_retrieval_custom_results.json"
TOP_K = 3
N_TRAIN = 500  # 用來建立檢索庫的樣本數
N_EVAL = 50    # 測試樣本數
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

# 建立檢索知識庫（以 QA 拼接）
corpus_texts = [f"Q: {x['question']}\nA: {x['answer']}" for x in trainset]
corpus_ids = list(range(len(corpus_texts)))

# ========== BUILD FAISS INDEX ==========
print("Building FAISS index...")
embedder = SentenceTransformer(EMBED_MODEL)
corpus_embeds = embedder.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)

faiss.normalize_L2(corpus_embeds)
index = faiss.IndexFlatIP(corpus_embeds.shape[1])
index.add(corpus_embeds)

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
    device=0 if DEVICE == "cuda" else -1,
)

# ========== RAG GENERATION ==========
results = []
times = []

for sample in tqdm(evalset, desc="RAG generation"):
    q = sample["question"]
    gold_answer = sample["answer"]

    # 檢索階段計時
    retrieval_start = time.perf_counter()
    q_embed = embedder.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_embed)
    D, I = index.search(q_embed, TOP_K)
    retrieved_docs = [corpus_texts[i] for i in I[0]]
    retrieval_end = time.perf_counter()
    retrieval_time = retrieval_end - retrieval_start

    # 組合 Prompt
    context = "\n\n".join(retrieved_docs)
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
        "retrieved_context": retrieved_docs,
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

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} results to {OUTPUT_FILE}")
print(
    f"Total time: {summary['total_time_sec']:.2f} sec | "
    f"Avg total: {summary['average_time_sec']:.2f} sec | "
    f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
    f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
)

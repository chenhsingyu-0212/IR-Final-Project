#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3: Binary Confidence-based Retrieval Decision (Yes/No)
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
OUTPUT_FILE = "exp3_confidence_retrieval_custom_results.json"
TOP_K = 3
CONF_THRESHOLD = 0.6  # 信心分數低於此值才進行檢索
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

# ========== 建立知識庫 ==========
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
def parse_confidence(text: str) -> str:
    """擷取 'Confidence: Yes/No'（或變體），回傳字串 "yes" 或 "no"。

    支援的文字輸出包括：yes/no, y/n, true/false（大小寫皆可）。
    若找不到文字標記，會嘗試解析數值（如 0.8 或 80%）並以 0.5 做二元判斷。
    若皆無法解析則回傳 "no"（表示不自信，需要檢索）。
    """
    txt = text.lower()

    # 先找 yes/no 類型
    m = re.search(r"confidence[:：]\s*(yes|no|y|n|true|false)\b", txt)
    if m:
        val = m.group(1)
        if val in ("yes", "y", "true"):
            return "yes"
        return "no"

    # 若為小數或百分比（例如 0.8 或 80%），嘗試解析並以 0.5 作為二元界值
    m2 = re.search(r"confidence[:：]\s*([\d\.]+)\s*%?", txt)
    if m2:
        try:
            v = float(m2.group(1))
            if v > 1:
                v = v / 100.0
            return "yes" if v >= 0.5 else "no"
        except Exception:
            return "no"

    # fallback
    return "no"

# ========== EXPERIMENT LOOP ==========
results = []
times_total = []
times_initial = []
times_rag = []

for s in tqdm(evalset, desc="Confidence-based RAG"):
    q = s["question"]
    gold_answer = s["answer"]

    # Step 1: 初步回答 + 二元信心（Yes 或 No）
    # 要求模型嚴格輸出格式：
    # Answer: <text>\nConfidence: <Yes or No>    (Yes 表示有信心，No 表示不自信)
    prompt_conf = (
        f"You are an intelligent assistant.\n"
        f"Answer the following question directly. Then explicitly state whether you are confident with your answer by writing 'Yes' or 'No'.\n\n"
        f"Output format strictly as:\nAnswer: <your answer>\nConfidence: <Yes or No>\n\n"
        f"Question: {q}\n"
    )

    t0 = time.perf_counter()
    resp1 = generator(prompt_conf, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
    t1 = time.perf_counter()

    # 把 Answer 與後續的 Confidence 分開
    ans_raw = resp1.split("Answer:")[-1].split("Confidence")[0].strip()
    conf = parse_confidence(resp1)

    used_retrieval = False
    final_answer = ans_raw
    elapsed_initial = t1 - t0
    elapsed_rag = 0.0
    retrieved_context = []

    # Step 2: 若信心為 "no" → 檢索 + 再回答
    if conf == "no":
        used_retrieval = True
        q_embed = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_embed)
        D, I = index.search(q_embed, TOP_K)
        retrieved_context = [corpus_texts[i] for i in I[0]]
        context = "\n\n".join(retrieved_context)

        prompt_rag = (
            f"Your confidence was low. Use the retrieved context below to improve your answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        )

        t2 = time.perf_counter()
        resp2 = generator(prompt_rag, max_new_tokens=128, temperature=0.2, top_p=0.9)[0]["generated_text"]
        t3 = time.perf_counter()

        final_answer = resp2.split("Answer:")[-1].strip()
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
        "confidence": conf,
        "used_retrieval": used_retrieval,
        "retrieved_context": retrieved_context,
        "initial_time_sec": elapsed_initial,
        "rag_time_sec": elapsed_rag,
        "total_time_sec": total_time,
    })

# ========== 統計時間 ==========
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
    f"Average per sample: {summary['average_time_sec']:.2f} sec | "
    f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
    f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
)

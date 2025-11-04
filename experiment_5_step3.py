#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp5 Inference: Fine-tuned confidence model controlling retrieval
Dataset: single_answer_qa.json
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, torch, re, time, statistics, random
from tqdm import tqdm

# ========== CONFIG ==========
MODEL_DIR = "exp5_finetuned_confidence"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = "test.json"
TRAIN_FILE = "train.json"
OUTPUT_FILE = "exp5_finetuned_confidence_custom_results.json"
TOP_K = 3
CONF_THRESHOLD = 0.7
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

# 建立檢索庫
corpus = [f"Q: {x['question']}\nA: {x['answer']}" for x in trainset]
embedder = SentenceTransformer(EMBED_MODEL)
emb = embedder.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# ========== LOAD MODEL ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
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
    if conf < CONF_THRESHOLD:
        used = True
        qe = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(qe)
        D, I = index.search(qe, TOP_K)
        retrieved_context = [corpus[i] for i in I[0]]
        ctx = "\n\n".join(retrieved_context)

        p2 = f"Use the retrieved context to refine your answer.\nContext:\n{ctx}\nQuestion: {q}\nAnswer:"
        t2 = time.perf_counter()
        out2 = gen(p2, max_new_tokens=128, temperature=0.2)[0]["generated_text"]
        t3 = time.perf_counter()
        elapsed_rag = t3 - t2

        ans = out2.split("Answer:")[-1].strip()

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

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} results to {OUTPUT_FILE}")
print(
    f"Total time: {summary['total_time_sec']:.2f} sec | "
    f"Avg total: {summary['average_time_sec']:.2f} sec | "
    f"Avg initial: {summary['average_initial_time_sec']:.2f} sec | "
    f"Avg RAG: {summary['average_rag_time_sec']:.2f} sec"
)

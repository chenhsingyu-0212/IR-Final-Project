# filename: evaluate_em_f1.py

import json
import re
from collections import Counter
import argparse

# -------------------------
# Helper functions
# -------------------------
def normalize_answer(s):
    """簡單標準化：小寫、去掉標點與多餘空格"""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # 去掉標點
    s = re.sub(r"\s+", " ", s).strip()  # 多空格合併
    return s

def compute_exact_match(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))

def compute_f1(gold, pred):
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute EM and F1 for a prediction JSON file with 'results' and 'summary'")
    parser.add_argument("input_file", type=str, help="Path to JSON file containing results")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: save results with EM/F1 scores")
    args = parser.parse_args()

    # 讀 JSON
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    summary = data.get("summary", {})

    total_em = 0
    total_f1 = 0

    for r in results:
        gold = r.get("gold_answer", "")
        pred = r.get("predicted_answer", "")
        em = compute_exact_match(gold, pred)
        f1 = compute_f1(gold, pred)
        r["exact_match"] = em
        r["f1_score"] = f1
        total_em += em
        total_f1 += f1

    num_samples = len(results)
    summary["exact_match"] = total_em / num_samples * 100 if num_samples else 0
    summary["f1_score"] = total_f1 / num_samples * 100 if num_samples else 0

    print(f"Samples evaluated: {num_samples}")
    print(f"Exact Match: {summary['exact_match']:.2f}%")
    print(f"F1 Score: {summary['f1_score']:.2f}%")

    # 可選：將結果存檔
    if args.output_file:
        # 更新 data dict 的 summary 和 results
        data["results"] = results
        data["summary"] = summary
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()

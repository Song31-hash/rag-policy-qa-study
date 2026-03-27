import json
import os
import re
import pandas as pd

BASE_DIR = "experiments"
results = []

pattern = re.compile(r"^overlap_(\d+)_topk_(\d+)$")

for exp_name in os.listdir(BASE_DIR):
    exp_path = os.path.join(BASE_DIR, exp_name)
    pred_path = os.path.join(exp_path, "predictions.json")

    if not os.path.isdir(exp_path):
        continue
    if not os.path.exists(pred_path):
        continue

    m = pattern.match(exp_name)
    if not m:
        # joint 실험만 사용
        continue

    expected_overlap = int(m.group(1))
    expected_topk = int(m.group(2))

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {})
    top_k = config.get("top_k")
    overlap = config.get("chunk_overlap")

    # 폴더명과 json 내부값 불일치 검사
    if top_k != expected_topk or overlap != expected_overlap:
        print(
            f"[WARNING] mismatch: {exp_name} "
            f"(folder: overlap={expected_overlap}, top_k={expected_topk} / "
            f"json: overlap={overlap}, top_k={top_k})"
        )

    gold_path = os.path.join("data", "dataset", "gold_answers.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_map = {
        item["id"]: item["expected_decision"]
        for item in gold_data["questions"]
    }

    total = 0
    correct = 0

    for item in data.get("results", []):
        qid = item["question_id"]
        answer = item["answer"].lower().strip()
        gold = gold_map[qid]

        pred = None
        if "selection_required" in answer or "상생형" in answer or "tops형" in answer:
            pred = "selection_required"
        elif "신청이 불가능" in answer or "불가능" in answer or "자격이 없습니다" in answer or "no" == answer:
            pred = "no"
        elif "신청이 가능" in answer or "가능" in answer or "자격이 됩니다" in answer or "yes" == answer:
            pred = "yes"
        else:
            # 보수적으로 fallback
            if gold == "selection_required" and ("상생형" in answer or "tops형" in answer):
                pred = "selection_required"
            elif "불가" in answer:
                pred = "no"
            else:
                pred = "unknown"

        total += 1
        if pred == gold:
            correct += 1

    accuracy = correct / total if total else 0.0

    results.append({
        "experiment": exp_name,
        "overlap": overlap,
        "top_k": top_k,
        "accuracy": accuracy,
    })

df = pd.DataFrame(results)
df = df.sort_values(by=["overlap", "top_k"]).drop_duplicates(subset=["overlap", "top_k"])

os.makedirs("results/tables", exist_ok=True)
df.to_csv("results/tables/summary.csv", index=False, encoding="utf-8-sig")

print(df)
print("\nSaved: results/tables/summary.csv")
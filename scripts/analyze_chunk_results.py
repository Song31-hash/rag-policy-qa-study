import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = {
    "256": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk256" / "evaluation.json",
    "512": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk512" / "evaluation.json",
    "1024": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk1024" / "evaluation.json",
}

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    summary_rows = []
    per_question = {}

    for chunk_size, eval_path in EXPERIMENTS.items():
        if not eval_path.exists():
            print(f"Missing evaluation file: {eval_path}")
            continue

        data = load_json(eval_path)

        summary_rows.append({
            "chunk_size": chunk_size,
            "accuracy": data.get("accuracy", ""),
            "rule_retrieval_rate": data.get("rule_retrieval_rate", ""),
            "num_questions": data.get("num_questions", ""),
        })

        for item in data.get("per_item", []):
            qid = item.get("question_id", "")
            if qid not in per_question:
                per_question[qid] = {"question_id": qid}

            per_question[qid][f"gold_{chunk_size}"] = item.get("gold_answer", "")
            per_question[qid][f"pred_{chunk_size}"] = item.get("normalized_answer", "")
            per_question[qid][f"acc_{chunk_size}"] = item.get("accuracy", False)
            per_question[qid][f"error_{chunk_size}"] = item.get("error_type", "")
            per_question[qid][f"retrieval_{chunk_size}"] = item.get("retrieved_contains_correct_rule", False)

    # 1) chunk summary
    summary_path = RESULTS_DIR / "chunk_size_comparison.csv"
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_size", "accuracy", "rule_retrieval_rate", "num_questions"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # 2) per-question comparison
    per_question_path = RESULTS_DIR / "chunk_size_per_question.csv"
    fieldnames = [
        "question_id",
        "gold_256", "pred_256", "acc_256", "error_256", "retrieval_256",
        "gold_512", "pred_512", "acc_512", "error_512", "retrieval_512",
        "gold_1024", "pred_1024", "acc_1024", "error_1024", "retrieval_1024",
    ]
    rows = [per_question[qid] for qid in sorted(per_question.keys())]

    with open(per_question_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")
    print(f"Saved: {per_question_path}")


if __name__ == "__main__":
    main()
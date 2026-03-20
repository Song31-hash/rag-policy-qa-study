import json
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_decision(text: str) -> str:
    if not text:
        return "uncertain"

    t = text.strip().lower()

    selection_keywords = [
        "상생형", "tops형", "트랙", "유형", "선택", "어느 트랙", "어떤 트랙"
    ]
    if any(k.lower() in t for k in selection_keywords):
        return "selection_required"

    no_keywords = [
        "아니요",
        "불가",
        "불가능",
        "제외",
        "지원 대상이 아닙니다",
        "신청 자격이 없습니다",
        "자격이 없습니다",
        "신청이 불가능",
        "해당하지 않습니다",
        "지원이 어렵",
        "지원 대상이 되지 않습니다",
    ]
    if any(k.lower() in t for k in no_keywords):
        return "no"

    yes_keywords = [
        "예",
        "네",
        "가능",
        "신청 가능합니다",
        "신청이 가능합니다",
        "자격이 있습니다",
        "지원 대상입니다",
        "신청할 수 있습니다",
        "지원이 가능합니다",
        "해당합니다",
    ]
    if any(k.lower() in t for k in yes_keywords):
        return "yes"

    uncertain_keywords = [
        "판단할 수 없습니다",
        "명확하지 않습니다",
        "확인이 필요합니다",
        "가능성이 높습니다",
        "해당할 수 있습니다",
        "정보가 없습니다",
        "명시되어 있지 않습니다",
    ]
    if any(k.lower() in t for k in uncertain_keywords):
        return "uncertain"

    return "uncertain"


def load_gold_map(gold_path: Path) -> dict:
    gold_data = load_json(gold_path)
    questions = gold_data.get("questions", [])

    gold_map = {}
    for item in questions:
        qid = str(item.get("id"))
        gold_map[qid] = {
            "question": item.get("question", ""),
            "gold_decision": item.get("expected_decision", ""),
            "category": item.get("category", ""),
            "reasoning_point": item.get("reasoning_point", ""),
            "target_eligibility": item.get("target_eligibility", ""),
        }
    return gold_map


def load_prediction_map(pred_path: Path) -> dict:
    data = load_json(pred_path)
    results = data.get("results", [])
    pred_map = {}

    for item in results:
        qid = str(item.get("question_id"))
        answer = item.get("answer", "")
        retrieved_chunks = item.get("retrieved_chunks", [])

        pred_map[qid] = {
            "question": item.get("question", ""),
            "answer_raw": answer,
            "decision": normalize_decision(answer),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
            "num_retrieved_chunks": len(retrieved_chunks),
        }
    return pred_map


def compute_improvement(gold: str, vanilla: str, rag: str) -> str:
    vanilla_correct = vanilla == gold
    rag_correct = rag == gold

    if not vanilla_correct and rag_correct:
        return "RAG fixed"
    if vanilla_correct and not rag_correct:
        return "RAG worse"
    if vanilla_correct and rag_correct:
        return "-"
    return "none"


def infer_error_type(gold: str, vanilla: str, rag: str, rag_num_chunks: int) -> str:
    vanilla_correct = vanilla == gold
    rag_correct = rag == gold

    if vanilla_correct and rag_correct:
        return "correct"

    if not vanilla_correct and rag_correct:
        return "vanilla_error"

    if vanilla_correct and not rag_correct:
        if rag_num_chunks == 0:
            return "retrieval_failure"
        return "reasoning_error"

    # 둘 다 틀림
    if rag == "uncertain":
        if rag_num_chunks == 0:
            return "retrieval_failure"
        return "source_gap_or_retrieval_insufficiency"

    if rag_num_chunks == 0:
        return "retrieval_failure"

    return "reasoning_or_source_gap"


def build_notes(row: dict) -> str:
    error_type = row["error_type"]

    if error_type == "correct":
        return "Both models matched gold decision."
    if error_type == "vanilla_error":
        return "Vanilla failed but RAG matched gold."
    if error_type == "retrieval_failure":
        return "RAG failed without sufficient retrieved evidence."
    if error_type == "reasoning_error":
        return "RAG retrieved evidence but still produced wrong decision."
    if error_type == "source_gap_or_retrieval_insufficiency":
        return "RAG responded uncertain despite retrieved chunks; source completeness should be checked."
    if error_type == "reasoning_or_source_gap":
        return "Both models wrong; inspect policy.docx completeness and rule application."
    return ""


def main():
    gold_path = ROOT / "data" / "dataset" / "gold_answers.json"
    vanilla_path = ROOT / "experiments" / "vanilla_llm" / "run1" / "predictions.json"
    rag_path = ROOT / "experiments" / "rag_baseline" / "run1" / "predictions.json"

    out_dir = ROOT / "results" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_map = load_gold_map(gold_path)
    vanilla_map = load_prediction_map(vanilla_path)
    rag_map = load_prediction_map(rag_path)

    rows = []
    for qid, gold_item in gold_map.items():
        vanilla_item = vanilla_map.get(qid, {})
        rag_item = rag_map.get(qid, {})

        row = {
            "QID": qid,
            "Question": gold_item.get("question", ""),
            "Category": gold_item.get("category", ""),
            "Gold": gold_item.get("gold_decision", ""),
            "Vanilla": vanilla_item.get("decision", "missing"),
            "RAG": rag_item.get("decision", "missing"),
            "Improvement": compute_improvement(
                gold_item.get("gold_decision", ""),
                vanilla_item.get("decision", "missing"),
                rag_item.get("decision", "missing"),
            ),
            "ErrorType": infer_error_type(
                gold_item.get("gold_decision", ""),
                vanilla_item.get("decision", "missing"),
                rag_item.get("decision", "missing"),
                rag_item.get("num_retrieved_chunks", 0),
            ),
            "RAGRetrievedChunkIDs": ", ".join(
                str(x) for x in rag_item.get("retrieved_chunk_ids", [])
            ),
            "ReasoningPoint": gold_item.get("reasoning_point", ""),
            "VanillaAnswerRaw": vanilla_item.get("answer_raw", ""),
            "RAGAnswerRaw": rag_item.get("answer_raw", ""),
            "Notes": "",
            "EvidenceInPolicyDocManual": "",
            "RetrievedCorrectRuleManual": "",
            "FinalErrorTypeManual": "",
        }
        row["Notes"] = build_notes({
            "error_type": row["ErrorType"]
        })
        rows.append(row)

    csv_path = out_dir / "error_analysis_full.csv"
    md_path = out_dir / "error_analysis_full.md"

    fieldnames = list(rows[0].keys()) if rows else []

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis Table\n\n")
        f.write("| QID | Gold | Vanilla | RAG | Improvement | ErrorType | Notes |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['QID']} | {row['Gold']} | {row['Vanilla']} | {row['RAG']} | "
                f"{row['Improvement']} | {row['ErrorType']} | {row['Notes']} |\n"
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
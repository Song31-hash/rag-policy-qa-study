"""평가 오케스트레이션: answer_eval + retrieval_eval + error_analysis 통합."""
from typing import Any

from .answer_eval import exact_match, evaluate_answer, answer_accuracy
from .retrieval_eval import check_chunks_contain_rule, retrieval_recall_at_k, evaluate_retrieval_single
from .error_analysis import classify_error_type, aggregate_error_types


def normalize_decision(text: str) -> str:
    """
    자연어 답변을 yes / no / selection_required 로 정규화.
    """
    if not text:
        return ""

    t = text.strip().lower()

    # selection_required / track choice
    selection_keywords = [
        "상생형", "tops형", "트랙", "유형", "선택", "어느 쪽", "어떤 트랙"
    ]
    if any(k.lower() in t for k in selection_keywords):
        return "selection_required"

    # no
    no_keywords = [
        "아니요",
        "불가",
        "불가능",
        "제외",
        "지원 대상이 아닙니다",
        "신청이 불가능",
        "신청 자격이 없습니다",
        "자격이 없습니다",
        "지원이 어렵",
        "해당하지 않습니다",
    ]
    if any(k.lower() in t for k in no_keywords):
        return "no"

    # yes
    yes_keywords = [
        "예",
        "가능",
        "신청 가능합니다",
        "지원 대상입니다",
        "자격이 있습니다",
        "신청할 수 있습니다",
        "지원이 가능합니다",
        "해당합니다",
    ]
    if any(k.lower() in t for k in yes_keywords):
        return "yes"

    return t


def evaluate_single(
    question_id: str,
    model_answer: str,
    retrieved_chunks: list[dict],
    gold: dict,
    error_type_manual: str | None = None,
) -> dict[str, Any]:
    """단일 질문 평가 (answer + retrieval + error_type)."""
    gold_answer = gold.get("answer", gold.get("expected_decision", ""))
    normalized_answer = normalize_decision(model_answer)

    ans_ev = evaluate_answer(normalized_answer, gold_answer)
    ret_ev = evaluate_retrieval_single(retrieved_chunks, gold)
    error_type = classify_error_type(
        normalized_answer, gold_answer, retrieved_chunks, gold, error_type_manual or gold.get("error_type")
    )

    return {
        "question_id": question_id,
        "model_answer": model_answer,
        "normalized_answer": normalized_answer,
        "gold_answer": gold_answer,
        "accuracy": ans_ev["accuracy"],
        "exact_match": ans_ev["exact_match"],
        "token_f1": ans_ev["token_f1"],
        "retrieved_contains_correct_rule": ret_ev["retrieved_contains_correct_rule"],
        "num_retrieved": ret_ev["num_retrieved"],
        "error_type": error_type,
    }


def run_evaluation(
    results: list[dict],
    gold_map: dict[str, dict],
) -> dict[str, Any]:
    """
    results: [{ question_id, answer, retrieved_chunks }, ...]
    gold_map: { question_id: { answer, correct_rule_chunk_id?, error_type? } }
    """
    per_item = []
    for r in results:
        qid = str(r.get("question_id") or r.get("id", ""))
        gold = gold_map.get(qid, {})
        ev = evaluate_single(
            question_id=qid,
            model_answer=r.get("answer", ""),
            retrieved_chunks=r.get("retrieved_chunks", []),
            gold=gold,
            error_type_manual=r.get("error_type"),
        )
        per_item.append(ev)

    return {
        "per_item": per_item,
        "accuracy": answer_accuracy(per_item),
        "rule_retrieval_rate": retrieval_recall_at_k(per_item),
        "num_questions": len(per_item),
        "error_type_counts": aggregate_error_types(per_item),
    }
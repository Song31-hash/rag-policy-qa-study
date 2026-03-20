"""답변 평가: accuracy, exact match, F1 (토큰/문자 단위)."""
from typing import Any


def exact_match(model_answer: str, gold_answer: str) -> bool:
    """문자열 완전 일치 또는 포함."""
    ga = (gold_answer or "").strip().lower()
    ma = (model_answer or "").strip().lower()
    if not ga:
        return False
    return ga == ma or ga in ma or ma in ga


def token_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    """토큰 단위 F1 (공백 분리)."""
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    prec = overlap / len(pred_set) if pred_set else 0.0
    rec = overlap / len(gold_set)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def answer_accuracy(per_item_results: list[dict]) -> float:
    """전체 정확도 (exact match 비율)."""
    if not per_item_results:
        return 0.0
    return sum(1 for r in per_item_results if r.get("accuracy")) / len(per_item_results)


def evaluate_answer(model_answer: str, gold_answer: str) -> dict[str, Any]:
    """단일 답변에 대한 메트릭."""
    em = exact_match(model_answer, gold_answer)
    pred_tokens = (model_answer or "").split()
    gold_tokens = (gold_answer or "").split()
    f1 = token_f1(pred_tokens, gold_tokens)
    return {"accuracy": em, "exact_match": em, "token_f1": f1}

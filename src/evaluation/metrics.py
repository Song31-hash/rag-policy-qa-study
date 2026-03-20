"""평가 지표: 정확도, 규칙 검색 여부."""
from typing import Any


def compare_answer(model_answer: str, gold_answer: str) -> bool:
    """문자열 일치/포함 여부."""
    ga = (gold_answer or "").strip().lower()
    ma = (model_answer or "").strip().lower()
    if not ga:
        return False
    return ga in ma or ma in ga or ga == ma


def check_chunks_contain_rule(
    retrieved_chunks: list[dict[str, Any]],
    correct_rule_chunk_id: int | None,
) -> bool:
    """검색된 청크에 정답 규칙 청크가 포함되었는지."""
    if correct_rule_chunk_id is None:
        return False
    chunk_ids = {c.get("chunk_id") for c in retrieved_chunks}
    return correct_rule_chunk_id in chunk_ids


def compute_accuracy(per_item_results: list[dict]) -> float:
    """전체 정확도 (맞은 개수 / 전체)."""
    if not per_item_results:
        return 0.0
    return sum(1 for r in per_item_results if r.get("accuracy")) / len(per_item_results)


def compute_rule_retrieval_rate(per_item_results: list[dict]) -> float:
    """retrieved_contains_correct_rule 비율."""
    if not per_item_results:
        return 0.0
    return sum(1 for r in per_item_results if r.get("retrieved_contains_correct_rule")) / len(per_item_results)

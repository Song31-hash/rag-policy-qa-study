"""Retrieval 평가: recall, precision (correct_rule_chunk 포함 여부)."""
from typing import Any


def check_chunks_contain_rule(
    retrieved_chunks: list[dict[str, Any]],
    correct_rule_chunk_id: int | None,
) -> bool:
    """검색된 청크에 정답 규칙 청크가 포함되었는지 (retrieval recall @k)."""
    if correct_rule_chunk_id is None:
        return False
    chunk_ids = {c.get("chunk_id") for c in retrieved_chunks}
    return correct_rule_chunk_id in chunk_ids


def retrieval_recall_at_k(per_item_results: list[dict], k: int | None = None) -> float:
    """correct_rule이 검색된 비율 (rule_retrieval_rate)."""
    if not per_item_results:
        return 0.0
    return sum(1 for r in per_item_results if r.get("retrieved_contains_correct_rule")) / len(per_item_results)


def retrieval_precision_at_k(retrieved_chunks: list[dict], relevant_chunk_ids: set[int]) -> float:
    """검색된 청크 중 relevant 비율 (relevant가 1개일 때 0 or 1/k)."""
    if not retrieved_chunks:
        return 0.0
    retrieved_ids = {c.get("chunk_id") for c in retrieved_chunks}
    hits = len(retrieved_ids & relevant_chunk_ids)
    return hits / len(retrieved_chunks) if retrieved_chunks else 0.0


def evaluate_retrieval_single(
    retrieved_chunks: list[dict],
    gold: dict,
) -> dict[str, Any]:
    """단일 질문에 대한 retrieval 메트릭."""
    correct_id = gold.get("correct_rule_chunk_id")
    contains_rule = check_chunks_contain_rule(retrieved_chunks, correct_id)
    return {
        "retrieved_contains_correct_rule": contains_rule,
        "num_retrieved": len(retrieved_chunks),
    }

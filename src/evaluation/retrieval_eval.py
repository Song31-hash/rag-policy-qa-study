"""Retrieval 평가: recall, precision (correct_rule_chunk 포함 여부)."""
from typing import Any


def check_chunks_contain_rule(
    retrieved_chunks: list[dict[str, Any]],
    correct_rule_chunk_id: int | None,
) -> bool | None:
    """
    검색된 청크에 정답 규칙 청크가 포함되었는지 확인.
    반환:
      - True / False : correct_rule_chunk_id가 있는 경우
      - None         : gold에 chunk-level annotation이 없는 경우
    """
    if correct_rule_chunk_id is None:
        return None

    chunk_ids = {c.get("chunk_id") for c in retrieved_chunks}
    return correct_rule_chunk_id in chunk_ids


def retrieval_recall_at_k(
    per_item_results: list[dict],
    k: int | None = None,
) -> float | None:
    """
    correct_rule이 검색된 비율 (rule_retrieval_rate).
    주의:
      - retrieved_contains_correct_rule가 None인 항목은 제외
      - 계산 가능한 항목이 하나도 없으면 None 반환
    """
    if not per_item_results:
        return None

    valid = [
        r.get("retrieved_contains_correct_rule")
        for r in per_item_results
        if r.get("retrieved_contains_correct_rule") is not None
    ]

    if not valid:
        return None

    return sum(1 for v in valid if v) / len(valid)


def retrieval_precision_at_k(
    retrieved_chunks: list[dict],
    relevant_chunk_ids: set[int] | None,
) -> float | None:
    """
    검색된 청크 중 relevant 비율.
    relevant_chunk_ids가 없으면 계산 불가이므로 None 반환.
    """
    if relevant_chunk_ids is None:
        return None

    if not retrieved_chunks:
        return 0.0

    retrieved_ids = {c.get("chunk_id") for c in retrieved_chunks}
    hits = len(retrieved_ids & relevant_chunk_ids)
    return hits / len(retrieved_chunks)


def evaluate_retrieval_single(
    retrieved_chunks: list[dict],
    gold: dict,
) -> dict[str, Any]:
    """
    단일 질문에 대한 retrieval 메트릭.
    gold에 correct_rule_chunk_id가 없으면 retrieval success/failure를 판정하지 않음.
    """
    correct_id = gold.get("correct_rule_chunk_id")
    contains_rule = check_chunks_contain_rule(retrieved_chunks, correct_id)

    return {
        "retrieved_contains_correct_rule": contains_rule,
        "num_retrieved": len(retrieved_chunks),
        "has_chunk_level_gold": correct_id is not None,
    }
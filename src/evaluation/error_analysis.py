"""에러 분석: retrieval_failure, reasoning_failure, hallucination 등 수동/자동 분류."""
from typing import Any

from .answer_eval import exact_match
from .retrieval_eval import check_chunks_contain_rule


# 수동 라벨용 에러 타입 (논문 분석)
ERROR_TYPES = [
    "retrieval_failure",   # 정답 청크 미검색
    "reasoning_failure",   # 검색됐으나 잘못 추론
    "hallucination",      # 컨텍스트에 없는 내용 생성
    "no_answer",           # 답 없음/회피
    "other",
]


def classify_error_type(
    model_answer: str,
    gold_answer: str,
    retrieved_chunks: list[dict],
    gold: dict,
    manual_label: str | None = None,
) -> str | None:
    """
    manual_label 이 있으면 사용.
    없으면 휴리스틱: correct_rule 미검색 -> retrieval_failure,
    검색됐는데 틀림 -> reasoning_failure 등.
    """
    if manual_label and manual_label in ERROR_TYPES:
        return manual_label
    if exact_match(model_answer, gold_answer):
        return None  # 정답
    correct_id = gold.get("correct_rule_chunk_id")
    has_rule = check_chunks_contain_rule(retrieved_chunks, correct_id)
    if not has_rule:
        return "retrieval_failure"
    if not (model_answer or "").strip():
        return "no_answer"
    return "reasoning_failure"  # 기본 추정


def aggregate_error_types(per_item_results: list[dict]) -> dict[str, int]:
    """per_item 의 error_type 집계."""
    counts: dict[str, int] = {}
    for r in per_item_results:
        et = r.get("error_type") or "other"
        counts[et] = counts.get(et, 0) + 1
    return counts

"""Error analysis utilities for policy QA evaluation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

ALLOWED_DECISIONS = {"yes", "no", "selection_required"}

# 정책 QA에서 자주 등장하는 핵심 규칙 cue
DOMAIN_CUES = [
    "10명 미만",
    "5명 미만",
    "30일 이상",
    "10일 이상",
    "4회",
    "1회",
    "최근 3개월",
    "징수유예",
    "체납처분유예",
    "징수특례",
    "시설자금",
    "공유오피스",
    "공유주방",
    "부동산업",
    "약국",
    "특별재난지역",
    "비영리",
    "외국법인",
    "지점",
    "조합",
    "부채비율",
    "700%",
    "업력 7년 이하",
    "상생형",
    "tops형",
    "tops 프로그램",
    "2단계",
    "플랫폼 판매촉진",
    "온라인 플랫폼",
    "운수업",
    "건설업",
    "제조업",
    "음식점",
]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _normalize_text(text: str) -> str:
    """
    rule matching용 느슨한 정규화.
    - 소문자화
    - 공백 축소
    - 일부 구두점 제거
    """
    text = _safe_str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[\"'“”‘’\(\)\[\]\{\},:;]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _join_retrieved_texts(retrieved_chunks: list[dict]) -> str:
    return "\n".join(_safe_str(c.get("text", "")) for c in (retrieved_chunks or []))


def _extract_domain_cues(text: str) -> list[str]:
    """
    gold reasoning / question에서 정책 판단에 중요한 cue를 추출.
    """
    norm = _normalize_text(text)
    found = []

    for cue in DOMAIN_CUES:
        if _normalize_text(cue) in norm:
            found.append(cue)

    numeric_patterns = [
        r"\d+\s*명\s*미만",
        r"\d+\s*일\s*이상",
        r"\d+\s*회",
        r"\d+\s*%",
        r"\d+\s*단계",
        r"\d+\s*년",
    ]
    for pattern in numeric_patterns:
        for match in re.findall(pattern, norm):
            found.append(match)

    dedup = []
    seen = set()
    for x in found:
        key = _normalize_text(x)
        if key and key not in seen:
            dedup.append(x)
            seen.add(key)

    return dedup


def infer_retrieval_status(
    retrieved_chunks: list[dict],
    gold: dict,
) -> str:
    """
    retrieval 상태를 success / partial / failure / unknown 으로 판정.
    """
    if not retrieved_chunks:
        return "not_applicable"

    retrieved_text = _normalize_text(_join_retrieved_texts(retrieved_chunks))
    reasoning = _safe_str(gold.get("reasoning_point", ""))
    question = _safe_str(gold.get("question", ""))
    gold_answer = _safe_str(gold.get("answer", gold.get("expected_decision", "")))

    cue_source = " ".join([reasoning, question, gold_answer])
    cues = _extract_domain_cues(cue_source)

    if not cues:
        return "unknown"

    matched = []
    for cue in cues:
        if _normalize_text(cue) in retrieved_text:
            matched.append(cue)

    n_total = len(cues)
    n_matched = len(matched)

    if n_total <= 2:
        if n_matched >= 1:
            return "success"
        return "failure"

    ratio = n_matched / n_total

    if n_matched >= 2 and ratio >= 0.5:
        return "success"
    if n_matched >= 1:
        return "partial"
    return "failure"


def _is_unclassified_decision(normalized_answer: str) -> bool:
    return _safe_str(normalized_answer) not in ALLOWED_DECISIONS


def classify_error_type(
    normalized_answer: str,
    gold_answer: str,
    retrieved_chunks: list[dict],
    gold: dict,
    error_type_manual: str | None = None,
) -> str | None:
    """
    오류 유형 분류.
    반환값:
    - retrieval_failure
    - partial_retrieval
    - reasoning_failure
    - normalization_failure
    - no_retrieval_setting
    - other
    - None (정답이면)
    """
    if error_type_manual is not None:
        manual = _safe_str(error_type_manual).strip()
        if manual and manual.lower() != "nan":
            return manual

    normalized_answer = _safe_str(normalized_answer)
    gold_answer = _safe_str(gold_answer)

    if normalized_answer == gold_answer:
        return None

    # 1) 먼저 정규화 실패 확인
    if _is_unclassified_decision(normalized_answer):
        return "normalization_failure"

    retrieval_status = infer_retrieval_status(retrieved_chunks, gold)

    # 2) retrieval 자체가 없는 설정 (e.g. vanilla)
    if retrieval_status == "not_applicable":
        return "reasoning_failure"

    # 3) retrieval 실패
    if retrieval_status == "failure":
        return "retrieval_failure"

    # 4) 일부만 가져온 경우
    if retrieval_status == "partial":
        return "partial_retrieval"

    # 5) retrieval은 성공했는데 decision 판단을 잘못함
    if retrieval_status in {"success", "unknown"} and normalized_answer in ALLOWED_DECISIONS:
        return "reasoning_failure"

    return "other"


def aggregate_error_types(per_item: list[dict[str, Any]]) -> dict[str, int]:
    """
    per_item 평가 결과에서 error_type 빈도 집계.
    None은 정답으로 간주하여 별도 success로 집계.
    """
    counter = Counter()

    for item in per_item:
        err = item.get("error_type", None)
        if err is None:
            counter["success"] += 1
        else:
            counter[_safe_str(err)] += 1

    return dict(counter)
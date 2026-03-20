"""데이터셋 로더: questions, gold_answers, metadata, 정책 문서."""
from pathlib import Path
from typing import Any

from ..utils.io import load_json


def load_docx_text(path: Path) -> str:
    """policy.docx에서 전체 텍스트 추출."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx required: pip install python-docx")
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_questions(path: Path | None = None, dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """questions.json 로드. dataset_dir 이면 dataset_dir/questions.json 사용."""
    from ..utils.io import load_json
    if path is None and dataset_dir is not None:
        path = dataset_dir / "questions.json"
    if path is None:
        raise ValueError("path or dataset_dir required")
    data = load_json(path)
    if isinstance(data, list):
        return data
    if "questions" in data:
        return data["questions"]
    return [data]


def load_gold_answers(path: Path | None = None, dataset_dir: Path | None = None) -> dict[str, Any]:
    """gold_answers.json 로드. { question_id: { answer, correct_rule_chunk_id?, error_type? } }"""
    if path is None and dataset_dir is not None:
        path = dataset_dir / "gold_answers.json"
    if path is None:
        raise ValueError("path or dataset_dir required")
    return load_json(path)


def load_metadata(path: Path | None = None, dataset_dir: Path | None = None) -> dict[str, Any]:
    """metadata.json 로드. { question_id: { difficulty?, type? } }"""
    if path is None and dataset_dir is not None:
        path = dataset_dir / "metadata.json"
    if path is None or not path.exists():
        return {}
    return load_json(path)


# 하위 호환
def load_gold(path: Path | None = None, dataset_dir: Path | None = None) -> dict[str, Any]:
    """load_gold_answers 별칭 (benchmark gold.json 호환)."""
    if path is None and dataset_dir is not None:
        path = dataset_dir / "gold_answers.json"
    if path and path.exists():
        return load_json(path)
    if dataset_dir and (dataset_dir / "gold_answers.json").exists():
        return load_json(dataset_dir / "gold_answers.json")
    return {}

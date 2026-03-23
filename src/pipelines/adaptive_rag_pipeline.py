"""Adaptive RAG pipeline: question type routing + chunk-specific retrieval."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..dataset.dataset_loader import load_questions, load_gold_answers
from ..retrieval.retriever import load_chunks_and_index, retrieve
from ..generation.generator import generate
from ..evaluation.evaluator import run_evaluation
from ..utils.io import save_json, save_yaml, load_config_with_base, resolve_paths, get_project_root


TYPE_TO_CHUNK = {
    "threshold_boundary": "256",
    "multi_criteria": "512",
    "temporal_condition": "1024",
    "conditional_exception": "512",
    "financial_restriction": "1024",
    "industry_restriction": "512",
    "entity_type": "256",
    "program_track": "512",
    "track_matching": "512",
    "default": "512",
}


def classify_question_type(question: str) -> str:
    """
    Rule-based question type classifier for adaptive chunk routing.
    Gold category를 직접 쓰지 않고, 질문 텍스트만 보고 분류.
    """
    q = (question or "").lower().strip()

    # track matching
    if "어떤 트랙" in q or "무슨 트랙" in q or "어느 트랙" in q:
        return "track_matching"

    # multi-criteria: threshold + extra business condition
    if ("부업" in q or "병행" in q or "동시에" in q) and ("직원" in q or "상시 근로자" in q):
        return "multi_criteria"

    # threshold boundary
    if "직원" in q or "상시 근로자" in q or "근로자" in q:
        return "threshold_boundary"

    # temporal condition
    if "연체" in q or "최근 3개월" in q or "2개월 전" in q or "몇 개월 전" in q:
        return "temporal_condition"

    # conditional exception
    if "징수유예" in q or "예외" in q or "시설자금" in q or "특별재난지역" in q:
        return "conditional_exception"

    # financial restriction
    if "부채비율" in q or "체납" in q:
        return "financial_restriction"

    # industry restriction
    if "유흥주점" in q or "약국" in q or "공유오피스" in q or "부동산업" in q:
        return "industry_restriction"

    # entity type
    if "비영리" in q or "외국법인" in q or "조합" in q or "지점" in q:
        return "entity_type"

    # program track
    if "tops" in q or "플랫폼" in q or "입점" in q:
        return "program_track"

    return "default"


def _build_gold_map(gold_data: Any) -> dict[str, dict]:
    """
    gold_answers.json을 evaluator가 기대하는 형식으로 변환.
    """
    gold_map: dict[str, dict] = {}

    # case 1: {"Q-01": "no", ...} or {"Q-01": {"answer": "no"}, ...}
    if isinstance(gold_data, dict) and "questions" not in gold_data:
        for qid, value in gold_data.items():
            if isinstance(value, dict):
                item = dict(value)
                if "answer" not in item and "expected_decision" in item:
                    item["answer"] = item["expected_decision"]
                gold_map[str(qid)] = item
            else:
                gold_map[str(qid)] = {"answer": value}
        return gold_map

    # case 2: {"questions": [...]}
    if isinstance(gold_data, dict) and "questions" in gold_data and isinstance(gold_data["questions"], list):
        for item in gold_data["questions"]:
            qid = item.get("id") or item.get("question_id")
            if not qid:
                continue
            mapped = dict(item)
            mapped["answer"] = item.get("expected_decision", item.get("answer", ""))
            gold_map[str(qid)] = mapped
        return gold_map

    # case 3: [ ... ]
    if isinstance(gold_data, list):
        for item in gold_data:
            if not isinstance(item, dict):
                continue
            qid = item.get("id") or item.get("question_id")
            if not qid:
                continue
            mapped = dict(item)
            mapped["answer"] = item.get("expected_decision", item.get("answer", ""))
            gold_map[str(qid)] = mapped
        return gold_map

    raise ValueError(f"Unsupported gold data format: {type(gold_data)}")


def _load_adaptive_resources(config: dict) -> dict[str, tuple[list[dict], Any]]:
    """
    미리 생성해둔 256/512/1024 chunks + index를 모두 로드.
    config['adaptive_paths'] 사용.
    """
    adaptive_paths = config.get("adaptive_paths", {})
    resources = {}

    required = ["256", "512", "1024"]
    for key in required:
        if key not in adaptive_paths:
            raise KeyError(f"adaptive_paths['{key}'] not found in config")

        chunk_cfg = adaptive_paths[key]
        chunks_path = Path(chunk_cfg["chunks_path"])
        index_dir = Path(chunk_cfg["index_dir"])
        chunks, index = load_chunks_and_index(chunks_path, index_dir)
        resources[key] = (chunks, index)

    return resources


def run(
    config_path: Path,
    experiment_dir: Path,
    question_path: Path | None = None,
    openai_api_key: str | None = None,
    run_gold_eval: bool = True,
    gold_path: Path | None = None,
) -> list[dict[str, Any]]:
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    raw = load_config_with_base(config_path, get_project_root())
    project_root = get_project_root()
    config = resolve_paths(raw, project_root)
    paths = config["paths"]

    q_path = Path(question_path) if question_path else Path(paths["questions_path"])
    final_gold_path = Path(gold_path) if gold_path else Path(paths["gold_path"])

    embedding_cfg = config.get("embedding", {})
    retrieval_cfg = config.get("retrieval", {})
    generation_cfg = config.get("generation", {})

    embed_model = embedding_cfg.get("model", config.get("embedding_model", "text-embedding-3-small"))
    embedding_dimension = embedding_cfg.get("dimension", config.get("embedding_dimension"))
    top_k = retrieval_cfg.get("top_k", config.get("top_k", 5))
    llm_model = generation_cfg.get("model", config.get("llm_model", "gpt-4o-mini"))
    temperature = generation_cfg.get("temperature", config.get("temperature", 0))

    questions = load_questions(path=q_path)
    resources = _load_adaptive_resources(config)

    results = []
    for q in questions:
        qid = q.get("id", q.get("question_id", len(results)))
        question_text = q.get("question", q.get("text", ""))
        if not question_text:
            continue

        predicted_type = classify_question_type(question_text)
        selected_chunk = TYPE_TO_CHUNK.get(predicted_type, "512")
        chunks, index = resources[selected_chunk]

        retrieved = retrieve(
            query=question_text,
            chunks=chunks,
            index=index,
            embedding_model=embed_model,
            top_k=top_k,
            api_key=api_key,
            embedding_dimension=embedding_dimension,
        )

        context = "\n\n---\n\n".join(c["text"] for c in retrieved)
        answer = generate(question_text, context, llm_model, temperature, api_key)

        results.append(
            {
                "question_id": qid,
                "question": question_text,
                "predicted_question_type": predicted_type,
                "selected_chunk_size": selected_chunk,
                "retrieved_chunks": retrieved,
                "answer": answer,
            }
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(raw, experiment_dir / "config.yaml")

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": _config_for_log(config),
        "num_questions": len(results),
        "results": results,
    }
    save_json(payload, experiment_dir / "predictions.json")

    if run_gold_eval and final_gold_path.exists():
        gold_data = load_gold_answers(path=final_gold_path)
        gold_map = _build_gold_map(gold_data)

        eval_result = run_evaluation(results, gold_map)
        eval_result["config"] = _config_for_log(config)
        eval_result["timestamp"] = datetime.utcnow().isoformat() + "Z"
        save_json(eval_result, experiment_dir / "evaluation.json")

    return results


def _config_for_log(config: dict) -> dict:
    out = dict(config)
    if "paths" in out:
        out["paths"] = {k: str(v) for k, v in out["paths"].items()}
    if "adaptive_paths" in out:
        out["adaptive_paths"] = {
            k: {kk: str(vv) for kk, vv in v.items()}
            for k, v in out["adaptive_paths"].items()
        }
    return out
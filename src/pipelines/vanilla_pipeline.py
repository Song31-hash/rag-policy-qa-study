"""Vanilla LLM 파이프라인: retrieval 없이 전체 문서 또는 빈 컨텍스트로 생성."""
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..dataset.dataset_loader import load_questions, load_docx_text, load_gold_answers
from ..generation.generator import generate
from ..evaluation.evaluator import run_evaluation
from ..utils.io import load_config_with_base, resolve_paths, get_project_root, save_json, save_yaml


def run(
    config_path: Path,
    experiment_dir: Path,
    question_path: Path | None = None,
    dataset_dir: Path | None = None,
    policy_path: Path | None = None,
    use_full_doc: bool = True,
    openai_api_key: str | None = None,
    run_gold_eval: bool = True,
    gold_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    결과: config.yaml, predictions.json. run_gold_eval 이면 evaluation.json 도 생성.
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    raw = load_config_with_base(config_path, get_project_root())
    project_root = get_project_root()
    config = resolve_paths(raw, project_root)
    paths = config["paths"]
    ds_dir = dataset_dir or paths["dataset"]
    raw_dir = paths["raw"]
    q_path = question_path or ds_dir / "questions.json"
    doc_path = policy_path or raw_dir / "policy.docx"

    questions = load_questions(path=q_path)
    if use_full_doc and doc_path.exists():
        full_text = load_docx_text(doc_path)
    else:
        full_text = ""

    llm_model = config.get("llm_model", "gpt-4o-mini")
    temperature = config.get("temperature", 0)

    results = []
    for q in questions:
        qid = q.get("id", q.get("question_id", len(results)))
        question_text = q.get("question", q.get("text", ""))
        if not question_text:
            continue
        answer = generate(question_text, full_text, llm_model, temperature, api_key)
        results.append({
            "question_id": qid,
            "question": question_text,
            "retrieved_chunks": [],
            "answer": answer,
        })

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(raw, experiment_dir / "config.yaml")
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": _config_for_log(config),
        "num_questions": len(results),
        "results": results,
    }
    save_json(payload, experiment_dir / "predictions.json")

    if run_gold_eval:
        gold_path = gold_path or ds_dir / "gold_answers.json"
        if gold_path.exists():
            gold_data = load_gold_answers(path=gold_path)
            gold_map = {str(k): v if isinstance(v, dict) else {"answer": v} for k, v in gold_data.items()}
            eval_result = run_evaluation(results, gold_map)
            eval_result["config"] = _config_for_log(config)
            eval_result["timestamp"] = datetime.utcnow().isoformat() + "Z"
            save_json(eval_result, experiment_dir / "evaluation.json")

    return results


def _config_for_log(config: dict) -> dict:
    out = dict(config)
    if "paths" in out:
        out["paths"] = {k: str(v) for k, v in out["paths"].items()}
    return out

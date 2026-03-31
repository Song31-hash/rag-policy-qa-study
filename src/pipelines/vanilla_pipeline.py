"""Vanilla LLM 파이프라인: retrieval 없이 전체 문서 또는 빈 컨텍스트로 생성."""
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..dataset.dataset_loader import load_questions, load_docx_text, load_gold_answers
from ..generation.generator import generate
from ..evaluation.evaluator import run_evaluation
from ..utils.io import (
    load_config_with_base,
    resolve_paths,
    get_project_root,
    save_json,
    save_yaml,
)


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
    결과:
      - config.yaml
      - predictions.json
      - evaluation.json (gold가 존재하고 평가가 가능할 때)

    목표:
      - 일부 질문에서 API 오류가 나도 전체 run이 날아가지 않도록 함
      - 가능한 한 predictions.json은 항상 남기도록 함
      - gold_answers.json 형식이 달라도 robust 하게 처리
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

    llm_model = config.get("llm_model", "gpt-4o-mini")
    temperature = config.get("temperature", 0)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(raw, experiment_dir / "config.yaml")

    meta: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": _config_for_log(config),
        "question_path": str(q_path),
        "policy_path": str(doc_path),
        "use_full_doc": use_full_doc,
        "llm_model": llm_model,
        "temperature": temperature,
    }

    if not api_key:
        payload = {
            **meta,
            "num_questions": 0,
            "num_success": 0,
            "num_failed": 0,
            "fatal_error": "OPENAI_API_KEY is not set",
            "results": [],
        }
        save_json(payload, experiment_dir / "predictions.json")
        raise ValueError("OPENAI_API_KEY is not set")

    try:
        questions = load_questions(path=q_path)
    except Exception as e:
        payload = {
            **meta,
            "num_questions": 0,
            "num_success": 0,
            "num_failed": 0,
            "fatal_error": f"Failed to load questions: {type(e).__name__}: {e}",
            "results": [],
        }
        save_json(payload, experiment_dir / "predictions.json")
        raise

    full_text = ""
    doc_error = None
    if use_full_doc:
        try:
            if doc_path.exists():
                full_text = load_docx_text(doc_path)
            else:
                doc_error = f"Policy file not found: {doc_path}"
        except Exception as e:
            doc_error = f"Failed to load policy document: {type(e).__name__}: {e}"

    results: list[dict[str, Any]] = []
    num_success = 0
    num_failed = 0

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", q.get("question_id", idx))
        question_text = q.get("question", q.get("text", ""))

        if not question_text:
            results.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "retrieved_chunks": [],
                    "answer": "",
                    "status": "skipped",
                    "error": "Empty question text",
                }
            )
            num_failed += 1
            continue

        try:
            answer = generate(
                question=question_text,
                context=full_text,
                llm_model=llm_model,
                temperature=temperature,
                openai_api_key=api_key,
            )
            results.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "retrieved_chunks": [],
                    "answer": answer,
                    "status": "success",
                    "error": None,
                }
            )
            num_success += 1

        except Exception as e:
            results.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "retrieved_chunks": [],
                    "answer": "",
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            num_failed += 1

    payload = {
        **meta,
        "doc_error": doc_error,
        "context_char_length": len(full_text),
        "num_questions": len(results),
        "num_success": num_success,
        "num_failed": num_failed,
        "results": results,
    }
    save_json(payload, experiment_dir / "predictions.json")

    if run_gold_eval:
        try:
            gold_path = gold_path or ds_dir / "gold_answers.json"
            if gold_path.exists():
                gold_data = load_gold_answers(path=gold_path)
                gold_map = _build_gold_map(gold_data)

                eval_input = [
                    {
                        "question_id": item["question_id"],
                        "answer": item["answer"],
                        "retrieved_chunks": item.get("retrieved_chunks", []),
                    }
                    for item in results
                    if item.get("status") == "success"
                ]

                eval_result = run_evaluation(eval_input, gold_map)
                eval_result["config"] = _config_for_log(config)
                eval_result["timestamp"] = datetime.utcnow().isoformat() + "Z"
                eval_result["num_total_questions"] = len(results)
                eval_result["num_evaluated_questions"] = len(eval_input)
                eval_result["num_skipped_due_to_generation_error"] = (
                    len(results) - len(eval_input)
                )
                save_json(eval_result, experiment_dir / "evaluation.json")
        except Exception as e:
            eval_error_payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config": _config_for_log(config),
                "error": f"Evaluation failed: {type(e).__name__}: {e}",
                "num_total_questions": len(results),
                "num_success": num_success,
                "num_failed": num_failed,
            }
            save_json(eval_error_payload, experiment_dir / "evaluation_error.json")

    return results


def _build_gold_map(gold_data: Any) -> dict[str, dict[str, Any]]:
    """
    gold JSON을 evaluator가 기대하는 형태로 변환.
    지원 형식:
      1) { "Q-01": {"answer": "no"}, ... }
      2) { "Q-01": "no", ... }
      3) { "questions": [ { "id": "Q-01", "expected_decision": "no", ... }, ... ] }
    """
    if not isinstance(gold_data, dict):
        raise ValueError("gold data must be a dict")

    # 형식 3: benchmark style
    if "questions" in gold_data and isinstance(gold_data["questions"], list):
        gold_map: dict[str, dict[str, Any]] = {}
        for item in gold_data["questions"]:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("id", "")).strip()
            if not qid:
                continue
            gold_map[qid] = {
                "expected_decision": item.get("expected_decision", ""),
                "reasoning_point": item.get("reasoning_point", ""),
                "metadata": item.get("metadata", {}),
                "category": item.get("category", ""),
                "target_eligibility": item.get("target_eligibility"),
            }
        return gold_map

    # 형식 1, 2: direct map style
    gold_map = {}
    for k, v in gold_data.items():
        qid = str(k)
        if isinstance(v, dict):
            gold_map[qid] = v
        else:
            gold_map[qid] = {"answer": v}
    return gold_map


def _config_for_log(config: dict) -> dict:
    out = dict(config)
    if "paths" in out:
        out["paths"] = {k: str(v) for k, v in out["paths"].items()}
    return out
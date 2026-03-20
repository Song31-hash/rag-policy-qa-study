"""RAG 파이프라인: retrieval + generation, 실험 디렉터리에 config / predictions / evaluation 저장."""
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..dataset.dataset_loader import load_questions, load_gold_answers
from ..retrieval.retriever import load_chunks_and_index, retrieve
from ..generation.generator import generate
from ..evaluation.evaluator import run_evaluation
from ..utils.io import save_json, save_yaml, load_config_with_base, resolve_paths, get_project_root


def _build_gold_map(gold_data: Any) -> dict[str, dict]:
    """
    gold_answers.json을 evaluator가 기대하는 형식으로 변환.
    최종 형식:
      {
        "Q-01": {
          "answer": "no",
          "category": "...",
          "target_eligibility": "...",
          "reasoning_point": "..."
        },
        ...
      }
    """
    gold_map: dict[str, dict] = {}

    # case 1: {"Q-01": "no", ...}
    if isinstance(gold_data, dict) and all(
        isinstance(k, str) and not isinstance(v, list)
        for k, v in gold_data.items()
    ) and "questions" not in gold_data:
        for qid, value in gold_data.items():
            if isinstance(value, dict):
                if "answer" in value:
                    gold_map[str(qid)] = value
                elif "expected_decision" in value:
                    item = dict(value)
                    item["answer"] = item["expected_decision"]
                    gold_map[str(qid)] = item
                else:
                    gold_map[str(qid)] = {"answer": ""}
            else:
                gold_map[str(qid)] = {"answer": value}
        return gold_map

    # case 2: {"test_suite_metadata": ..., "questions": [...]}
    if isinstance(gold_data, dict) and "questions" in gold_data and isinstance(gold_data["questions"], list):
        for item in gold_data["questions"]:
            qid = item.get("id") or item.get("question_id")
            if not qid:
                continue
            mapped = dict(item)
            mapped["answer"] = item.get("expected_decision", item.get("answer", ""))
            gold_map[str(qid)] = mapped
        return gold_map

    # case 3: [{"id": "...", "expected_decision": "..."}]
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


def run(
    config_path: Path,
    experiment_dir: Path,
    question_path: Path | None = None,
    dataset_dir: Path | None = None,
    openai_api_key: str | None = None,
    run_gold_eval: bool = True,
    gold_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    experiment_dir: experiments/<exp_name>/run1 등.
    결과: config.yaml, predictions.json. run_gold_eval 이면 evaluation.json 도 생성.
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    raw = load_config_with_base(config_path, get_project_root())
    project_root = get_project_root()
    config = resolve_paths(raw, project_root)
    paths = config["paths"]

    chunks_path = Path(paths["chunks_path"])
    index_dir = Path(paths["index_dir"])
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
    chunks, index = load_chunks_and_index(chunks_path, index_dir)

    results = []
    for q in questions:
        qid = q.get("id", q.get("question_id", len(results)))
        question_text = q.get("question", q.get("text", ""))
        if not question_text:
            continue

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
    return out
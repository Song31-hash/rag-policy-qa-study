"""
실험 결과 평가: predictions.json + gold_answers.json -> evaluation.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.dataset_loader import load_gold_answers
from src.evaluation.evaluator import run_evaluation
from src.utils.io import load_json, load_yaml, save_json


def build_gold_map(gold_data: dict) -> dict:
    """
    gold_answers.json 구조를 evaluator가 읽을 수 있는 gold_map으로 변환.
    기대 출력:
    {
        "Q-01": {
            "answer": "no",
            "reasoning_point": "...",
            "target_eligibility": "...",
            "category": "..."
        },
        ...
    }
    """
    gold_map = {}

    # case 1: 현재 사용자 gold 구조
    # {
    #   "test_suite_metadata": {...},
    #   "questions": [
    #       {"id": "Q-01", "expected_decision": "no", ...},
    #       ...
    #   ]
    # }
    if isinstance(gold_data, dict) and "questions" in gold_data and isinstance(gold_data["questions"], list):
        for item in gold_data["questions"]:
            qid = str(item.get("id"))
            if not qid or qid == "None":
                continue

            gold_map[qid] = {
                "answer": item.get("expected_decision"),
                "reasoning_point": item.get("reasoning_point"),
                "target_eligibility": item.get("target_eligibility"),
                "category": item.get("category"),
            }

            # retrieval 평가용 정답 chunk id가 있다면 함께 전달
            if "correct_rule_chunk_id" in item:
                gold_map[qid]["correct_rule_chunk_id"] = item["correct_rule_chunk_id"]

        return gold_map

    # case 2: 이미 evaluator용 맵 구조인 경우
    # {
    #   "Q-01": {"answer": "no", ...},
    #   "Q-02": {"answer": "yes", ...}
    # }
    if isinstance(gold_data, dict):
        for k, v in gold_data.items():
            if isinstance(v, dict):
                gold_map[str(k)] = v
            else:
                gold_map[str(k)] = {"answer": v}
        return gold_map

    # case 3: list 구조인 경우
    # [
    #   {"id": "Q-01", "answer": "no"},
    #   ...
    # ]
    if isinstance(gold_data, list):
        for item in gold_data:
            qid = str(item.get("id") or item.get("question_id"))
            if not qid or qid == "None":
                continue
            gold_map[qid] = {
                "answer": item.get("answer") or item.get("expected_decision")
            }
        return gold_map

    return gold_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="e.g. vanilla_llm/run1 or rag_baseline/run1")
    parser.add_argument("--gold", type=str, default=None, help="gold_answers.json path (default: data/dataset/gold_answers.json)")
    parser.add_argument("--predictions", type=str, default=None, help="predictions.json path")
    args = parser.parse_args()

    experiment_path = Path(ROOT, "experiments", *args.experiment.split("/"))
    predictions_path = Path(args.predictions) if args.predictions else experiment_path / "predictions.json"
    if not predictions_path.is_absolute():
        predictions_path = ROOT / predictions_path

    gold_path = Path(args.gold) if args.gold else ROOT / "data" / "dataset" / "gold_answers.json"
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path

    if not predictions_path.exists():
        print(f"Predictions not found: {predictions_path}")
        sys.exit(1)

    if not gold_path.exists():
        print(f"Gold not found: {gold_path}")
        sys.exit(1)

    data = load_json(predictions_path)
    results = data.get("results", [])

    gold_data = load_gold_answers(path=gold_path)
    gold_map = build_gold_map(gold_data)

    print(f"Loaded predictions: {len(results)}")
    print(f"Loaded gold items: {len(gold_map)}")

    eval_result = run_evaluation(results, gold_map)
    eval_result["experiment"] = args.experiment

    config_path = experiment_path / "config.yaml"
    if config_path.exists():
        eval_result["config"] = load_yaml(config_path)

    save_json(eval_result, experiment_path / "evaluation.json")

    print(f"Accuracy: {eval_result['accuracy']:.4f}, Rule retrieval: {eval_result['rule_retrieval_rate']:.4f}")
    print(f"Saved: {experiment_path / 'evaluation.json'}")


if __name__ == "__main__":
    main()

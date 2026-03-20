"""
통합 실험 실행: config에 따라 RAG 또는 Vanilla 파이프라인 실행.
결과: experiments/<experiment>/<run_name>/config.yaml, predictions.json, evaluation.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, get_project_root
from src.pipelines.rag_pipeline import run as run_rag
from src.pipelines.vanilla_pipeline import run as run_vanilla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="configs/rag.yaml | configs/vanilla.yaml | configs/experiment/*.yaml")
    parser.add_argument("--experiment", type=str, required=True, help="e.g. vanilla, rag_baseline, chunk_size, topk")
    parser.add_argument("--run-name", type=str, default="run1", help="e.g. run1, run2, chunk_200")
    parser.add_argument("--no-eval", action="store_true", help="Skip gold evaluation")
    args = parser.parse_args()

    root = ROOT
    config_path = root / args.config
    if not config_path.exists():
        config_path = get_project_root() / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    raw = load_yaml(config_path)
    use_retrieval = raw.get("use_retrieval", True)
    experiment_dir = root / "experiments" / args.experiment / args.run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if use_retrieval:
        run_rag(
            config_path,
            experiment_dir,
            run_gold_eval=not args.no_eval,
        )
    else:
        run_vanilla(
            config_path,
            experiment_dir,
            run_gold_eval=not args.no_eval,
        )
    print(f"Results: {experiment_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()

"""Vanilla LLM 파이프라인 실행 (retrieval 없음)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.vanilla_pipeline import run
from src.utils.io import get_project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vanilla.yaml")
    parser.add_argument("--experiment", type=str, required=True, help="e.g. vanilla, rag_baseline")
    parser.add_argument("--run-name", type=str, default="run1")
    parser.add_argument("--questions", type=str, default=None)
    parser.add_argument("--no-doc", action="store_true", help="Do not use full policy doc as context")
    args = parser.parse_args()

    root = ROOT
    config_path = root / args.config
    experiment_dir = root / "experiments" / args.experiment / args.run_name
    question_path = Path(args.questions) if args.questions else None
    if question_path and not question_path.is_absolute():
        question_path = root / question_path

    run(
        config_path,
        experiment_dir,
        question_path=question_path,
        use_full_doc=not args.no_doc,
    )
    print(f"Results saved to {experiment_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()

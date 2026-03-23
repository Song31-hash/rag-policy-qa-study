"""Adaptive RAG pipeline 실행."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.adaptive_rag_pipeline import run
from src.utils.io import load_config_with_base, resolve_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/adaptive.yaml")
    parser.add_argument("--questions", type=str, default=None, help="Optional override for questions path")
    args = parser.parse_args()

    root = ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    config = load_config_with_base(config_path, root)
    config = resolve_paths(config, root)

    output_dir = Path(config["paths"]["output_dir"])
    question_path = Path(args.questions) if args.questions else Path(config["paths"]["questions_path"])

    if not question_path.is_absolute():
        question_path = (root / question_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    run(config_path, output_dir, question_path=question_path)
    print(f"Adaptive RAG results saved to {output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
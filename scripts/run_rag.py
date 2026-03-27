"""RAG 파이프라인 실행."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.rag_pipeline import run
from src.utils.io import load_config_with_base, resolve_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag.yaml")
    parser.add_argument("--questions", type=str, default=None)
    args = parser.parse_args()

    root = ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    # ----------------------------
    # config 로드
    # ----------------------------
    config = load_config_with_base(config_path, root)
    config = resolve_paths(config, root)

    # ----------------------------
    # 🔥 핵심: config 자체 수정
    # ----------------------------
    if "paths" not in config:
        config["paths"] = {}

    paths = config["paths"]

    # chunks_path 강제 생성
    if "chunks_path" not in paths:
        processed_dir = paths.get("processed", "data/processed")
        paths["chunks_path"] = str(Path(processed_dir) / "chunks.json")

    # questions_path 강제 생성
    if "questions_path" not in paths:
        dataset_dir = paths.get("dataset", "data/dataset")
        paths["questions_path"] = str(Path(dataset_dir) / "questions.json")

    # output_dir 설정
    if "output_dir" not in paths:
        paths["output_dir"] = f"experiments/topk_{config.get('top_k', 'default')}"

    output_dir = Path(paths["output_dir"])

    # 질문 경로
    if args.questions:
        question_path = Path(args.questions)
    else:
        question_path = Path(paths["questions_path"])

    if not question_path.is_absolute():
        question_path = (root / question_path).resolve()

    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 실행
    # ----------------------------
    run(config_path, output_dir, question_path=question_path)

    print(f"Results saved to {output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
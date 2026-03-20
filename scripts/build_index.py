"""
기존 data/processed/chunks.json 기준으로 FAISS 인덱스만 재생성.
(청크는 그대로 두고 embedding 모델/설정만 바꿀 때 사용)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_json, load_yaml, resolve_paths, get_project_root
from src.retrieval.embedder import embed_texts
from src.retrieval.faiss_index import build_index, save_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag.yaml")
    args = parser.parse_args()

    project_root = ROOT
    config_path = ROOT / args.config
    if not config_path.exists():
        config_path = get_project_root() / args.config
    config = load_yaml(config_path)
    config = resolve_paths(config, project_root)
    paths = config["paths"]
    processed_dir = paths["processed"]
    indexes_dir = paths.get("indexes") or (processed_dir / "faiss_index")
    chunks_path = processed_dir / "chunks.json"
    if not chunks_path.exists():
        print("chunks.json not found. Run ingest_policy.py first.")
        sys.exit(1)

    chunks = load_json(chunks_path)
    texts = [c["text"] for c in chunks]
    embed_model = config.get("embedding_model", "text-embedding-3-small")
    dim = config.get("embedding_dimension")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = embed_texts(client, texts, embed_model, dim)
    index, emb_norm = build_index(embeddings)
    indexes_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, emb_norm, indexes_dir)
    print(f"Index built at {indexes_dir}")


if __name__ == "__main__":
    main()

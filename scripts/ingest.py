"""
정책 문서 청크 분할 및 FAISS 인덱스 생성.
출력: config에 지정된 chunks_path, chunk_meta_path, index_dir
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.dataset_loader import load_docx_text
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import embed_texts
from src.retrieval.faiss_index import build_index, save_index
from src.utils.io import load_config_with_base, resolve_paths, get_project_root, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Config YAML path")
    parser.add_argument("--policy", type=str, default=None, help="Optional override for policy.docx path")
    args = parser.parse_args()

    project_root = ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    config = load_config_with_base(config_path, get_project_root())
    config = resolve_paths(config, project_root)
    paths = config["paths"]

    raw_dir = Path(paths["raw"])
    processed_dir = Path(paths["processed"])
    chunks_path = Path(paths["chunks_path"])
    chunk_meta_path = Path(paths["chunk_meta_path"])
    index_dir = Path(paths["index_dir"])

    policy_path = Path(args.policy) if args.policy else raw_dir / "policy.docx"
    if not policy_path.is_absolute():
        policy_path = (project_root / policy_path).resolve()

    if not policy_path.exists():
        print(f"Policy not found: {policy_path}")
        sys.exit(1)

    chunk_cfg = config.get("chunking", {})
    embedding_cfg = config.get("embedding", {})

    chunk_size = chunk_cfg.get("chunk_size", config.get("chunk_size", 512))
    chunk_overlap = chunk_cfg.get("chunk_overlap", config.get("chunk_overlap", 64))
    embed_model = embedding_cfg.get("model", config.get("embedding_model", "text-embedding-3-small"))
    embedding_dimension = embedding_cfg.get("dimension", config.get("embedding_dimension"))

    text = load_docx_text(policy_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_meta_path.parent.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    save_json(chunks, chunks_path)
    save_json(
        {
            "policy_path": str(policy_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embed_model,
            "embedding_dimension": embedding_dimension,
            "num_chunks": len(chunks),
        },
        chunk_meta_path,
    )

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    embeddings = embed_texts(client, [c["text"] for c in chunks], embed_model, embedding_dimension)
    index, emb_norm = build_index(embeddings)
    save_index(index, emb_norm, index_dir)

    print("Ingestion done.")
    print(f"chunks_path: {chunks_path}")
    print(f"chunk_meta_path: {chunk_meta_path}")
    print(f"index_dir: {index_dir}")


if __name__ == "__main__":
    main()
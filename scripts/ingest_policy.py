"""
정책 문서 청크 분할 + FAISS 인덱스 생성.
출력: data/processed/chunks.json, chunk_meta.json / data/indexes/index.faiss, embeddings.npy
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
from src.utils.io import load_yaml, resolve_paths, get_project_root, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag.yaml")
    parser.add_argument("--policy", type=str, default=None, help="policy.docx path (default: data/raw/policy.docx)")
    args = parser.parse_args()

    project_root = ROOT
    config_path = ROOT / args.config
    if not config_path.exists():
        config_path = get_project_root() / args.config
    config = load_yaml(config_path)
    config = resolve_paths(config, project_root)
    paths = config["paths"]
    raw_dir = paths["raw"]
    processed_dir = paths["processed"]
    indexes_dir = paths.get("indexes") or (processed_dir / "faiss_index")
    policy_path = Path(args.policy) if args.policy else (raw_dir / "policy.docx")
    if not policy_path.is_absolute():
        policy_path = project_root / policy_path
    if not policy_path.exists():
        print(f"Policy not found: {policy_path}")
        sys.exit(1)

    chunk_size = config.get("chunk_size", 512)
    overlap = config.get("chunk_overlap", 64)
    embed_model = config.get("embedding_model", "text-embedding-3-small")
    dim = config.get("embedding_dimension")

    text = load_docx_text(policy_path)
    print("text length:", len(text))
    print("first 200 chars:", text[:200])
    chunks = chunk_text(text, chunk_size, overlap)
    print("chunk count:", len(chunks))
    processed_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    save_json(chunks, processed_dir / "chunks.json")
    meta = {
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "embedding_model": embed_model,
        "num_chunks": len(chunks),
    }
    save_json(meta, processed_dir / "chunk_meta.json")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = embed_texts(client, [c["text"] for c in chunks], embed_model, dim)
    index, emb_norm = build_index(embeddings)
    save_index(index, emb_norm, indexes_dir)
    print(f"Ingestion done. processed: {processed_dir}, indexes: {indexes_dir}")


if __name__ == "__main__":
    main()

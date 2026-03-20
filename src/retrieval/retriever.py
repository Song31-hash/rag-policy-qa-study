"""FAISS 기반 retriever. top_k 조절 가능."""
from pathlib import Path
from typing import Any

from openai import OpenAI

from .embedder import embed_query
from .faiss_index import load_index, search


def load_chunks_and_index(chunks_path: Path, index_dir: Path):
    """지정된 chunks_path 와 index_dir 에서 청크와 FAISS 인덱스를 로드."""
    from ..utils.io import load_json

    chunks_path = Path(chunks_path)
    index_dir = Path(index_dir)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run ingest first.")
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS index not found at {index_dir}. Run ingest first.")

    chunks = load_json(chunks_path)
    index, _ = load_index(index_dir)
    return chunks, index


def retrieve(
    query: str,
    chunks: list[dict],
    index: Any,
    embedding_model: str,
    top_k: int,
    api_key: str | None = None,
    embedding_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """
    top_k 개 청크 검색.
    반환: [{ chunk_id, text, score, start_char, end_char }, ...]
    """
    client = OpenAI(api_key=api_key)
    q_emb = embed_query(client, query, embedding_model, embedding_dimension)

    import numpy as np

    q_emb = np.asarray(q_emb, dtype=np.float32)
    norm = np.linalg.norm(q_emb)
    if norm > 0:
        q_emb = q_emb / norm

    scores, indices = search(index, q_emb, top_k)

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append(
            {
                "chunk_id": c["id"],
                "text": c["text"],
                "score": float(score),
                "start_char": c["start_char"],
                "end_char": c["end_char"],
            }
        )
    return results
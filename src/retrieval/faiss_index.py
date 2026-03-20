"""FAISS 인덱스 빌드/저장/로드 및 검색."""
from pathlib import Path
from typing import Any

import numpy as np
from faiss import IndexFlatIP, read_index, write_index


def build_index(embeddings: np.ndarray) -> Any:
    """L2 정규화 후 IndexFlatIP (cosine) 생성."""
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norm
    d = emb_norm.shape[1]
    index = IndexFlatIP(d)
    index.add(emb_norm)
    return index, emb_norm


def save_index(index: Any, embeddings: np.ndarray, dir_path: Path) -> None:
    """인덱스와 정규화된 임베딩 저장."""
    dir_path.mkdir(parents=True, exist_ok=True)
    write_index(index, str(dir_path / "index.faiss"))
    np.save(dir_path / "embeddings.npy", embeddings)


def load_index(dir_path: Path) -> tuple[Any, np.ndarray]:
    """인덱스와 임베딩 로드. 반환: (index, embeddings)."""
    index = read_index(str(dir_path / "index.faiss"))
    embeddings = np.load(dir_path / "embeddings.npy")
    return index, embeddings


def search(index: Any, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """검색. query_embedding: (dim,) 정규화됨. 반환: (scores, indices)."""
    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores, indices = index.search(q, top_k)
    return scores[0], indices[0]

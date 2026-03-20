"""OpenAI embedding (고정 모델: text-embedding-3-small)."""
from typing import Any

import numpy as np
from openai import OpenAI


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimension: int | None = None,
) -> np.ndarray:
    """여러 텍스트 임베딩. 반환: (N, dim) float32."""
    kwargs: dict[str, Any] = {"model": model, "input": texts}
    if dimension is not None:
        kwargs["dimensions"] = dimension
    resp = client.embeddings.create(**kwargs)
    order = sorted(resp.data, key=lambda x: x.index)
    return np.array([d.embedding for d in order], dtype=np.float32)


def embed_query(
    client: OpenAI,
    query: str,
    model: str,
    dimension: int | None = None,
) -> np.ndarray:
    """단일 질의 임베딩. 반환: (dim,) float32."""
    arr = embed_texts(client, [query], model, dimension)
    return arr[0]

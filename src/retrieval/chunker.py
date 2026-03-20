"""정책 문서 청크 분할 (size / overlap 설정 가능)."""
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """
    텍스트를 chunk_size / overlap 으로 분할.
    각 청크: id, text, start_char, end_char.
    """
    chunks = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        slice_text = text[start:end]

        chunks.append({
            "id": chunk_id,
            "text": slice_text,
            "start_char": start,
            "end_char": end,
        })

        chunk_id += 1

        # 마지막 chunk이면 종료
        if end == text_len:
            break

        start = end - overlap

    return chunks

"""프롬프트 템플릿 (실험 재현·논문 기술용)."""
RAG_SYSTEM = (
    "Answer based only on the provided context. "
    "If no context or no answer in context, say so. Be concise."
)
VANILLA_SYSTEM = (
    "Answer based only on the provided context. "
    "If no context or no answer in context, say so. Be concise."
)


def format_rag_user(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def format_vanilla_user(context: str, question: str) -> str:
    if (context or "").strip():
        return f"Context:\n{context}\n\nQuestion: {question}"
    return question

"""LLM 생성. temperature=0 고정 (deterministic)."""
from openai import OpenAI


def generate(
    question: str,
    context: str,
    llm_model: str,
    temperature: float = 0.0,
    openai_api_key: str | None = None,
) -> str:
    """
    질문과 컨텍스트로 답변 생성.
    context: RAG면 검색된 청크 결합 문자열, vanilla면 전체 문서 또는 빈 문자열.
    """
    client = OpenAI(api_key=openai_api_key)
    if context.strip():
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "Answer based only on the provided context. If no context or no answer in context, say so. Be concise.",
            },
            {"role": "user", "content": user_content},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

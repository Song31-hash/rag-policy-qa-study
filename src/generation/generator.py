"""LLM 생성. temperature=0 고정 (deterministic)."""
from openai import OpenAI


def generate(
    question: str,
    context: str,
    llm_model: str,
    temperature: float = 0.0,
    openai_api_key: str | None = None,
    prompt_mode: str = "baseline",
) -> str:
    """
    질문과 컨텍스트로 답변 생성.
    context: RAG면 검색된 청크 결합 문자열, vanilla면 전체 문서 또는 빈 문자열.

    prompt_mode:
        - "baseline": 기존 방식과 유사하게 간단 응답
        - "cot": 단계적 판단을 유도하는 프롬프트
    """
    client = OpenAI(api_key=openai_api_key)

    if prompt_mode == "cot":
        system_prompt = (
            "You are a policy eligibility decision assistant.\n"
            "Answer strictly based only on the provided context.\n"
            "Do not use outside knowledge.\n\n"
            "Follow these steps internally:\n"
            "1. Extract the key conditions from the question.\n"
            "2. Find the relevant rule(s) in the context.\n"
            "3. Check whether any exception applies.\n"
            "4. Decide the final result.\n\n"
            "Output format:\n"
            "Reasoning: <brief reasoning based only on context>\n"
            "Final decision: yes / no / selection_required\n\n"
            "Important rules:\n"
            "- Apply boundary conditions exactly, such as 'less than 10' or 'less than 5'.\n"
            "- If the context does not support an answer, say so clearly.\n"
            "- The final decision must be one of: yes, no, selection_required."
        )

        if context.strip():
            user_content = (
                f"[Context]\n{context}\n\n"
                f"[Question]\n{question}"
            )
        else:
            user_content = (
                f"[Context]\n(no context)\n\n"
                f"[Question]\n{question}"
            )

    else:
        # 기존 실험 영향 최소화를 위한 baseline 모드
        system_prompt = (
            "Answer based only on the provided context. "
            "If no context or no answer in context, say so. Be concise."
        )

        if context.strip():
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            user_content = question

    resp = client.chat.completions.create(
        model=llm_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return (resp.choices[0].message.content or "").strip()
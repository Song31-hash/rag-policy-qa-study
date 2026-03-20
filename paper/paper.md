# Policy RAG: 실험 보고서

## 개요

본 프로젝트는 정책 문서 기반 RAG(Retrieval-Augmented Generation) 파이프라인의 평가를 목적으로 합니다.

## 실험 설정

- **Embedding**: text-embedding-3-small (고정)
- **Vector store**: FAISS
- **LLM**: GPT-4o-mini, temperature=0 (고정)
- **데이터셋**: data/dataset (questions.json, gold_answers.json, metadata.json)

## 실험 구성

| 실험명 | 설명 |
|--------|------|
| vanilla | Retrieval 없이 전체 문서/빈 컨텍스트로 생성 |
| rag_baseline | RAG 기본 (chunk_size=512, top_k=5) |
| chunk_size | chunk 200 / 400 / 800 변인 |
| topk | k=1, 3, 5 변인 |
| embedding | embedding 모델 변인 (openai_small, bge_base 등) |

## 메트릭

- **Answer**: accuracy, exact match, token F1
- **Retrieval**: retrieval recall (correct_rule 포함 여부)
- **Error analysis**: retrieval_failure, reasoning_failure, hallucination 등

## 결과

결과 테이블은 `results/tables/`, Figure는 `results/figures/` 에 저장합니다.

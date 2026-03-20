# Policy RAG Study

## Overview

This project investigates whether Retrieval-Augmented Generation (RAG) improves the accuracy of large language models in policy eligibility question answering.

A benchmark dataset of 20 policy QA questions was constructed based on a Korean government policy document. The study compares a vanilla LLM with a RAG-based system under controlled experimental conditions.

The results show that RAG improves accuracy from 0.55 to 0.75, demonstrating the importance of retrieval for rule-based reasoning tasks.

---

## Research Questions

This study investigates two key questions:

1. Does retrieval improve the accuracy of LLMs in policy-based decision tasks?
2. What types of errors can retrieval mitigate compared to a vanilla LLM?

---

## Dataset

The dataset consists of 20 manually constructed policy eligibility questions.

Each question is designed to evaluate specific reasoning challenges:

- Threshold boundary conditions
- Exception rules
- Industry restrictions
- Financial eligibility conditions
- Program selection logic

Each item includes:

- question_id
- question text
- expected decision (yes / no / selection_required)
- reasoning reference

Ground truth answers were manually verified against the policy document.

---

## Method

Two systems are evaluated:

### Vanilla LLM
The model directly answers questions without document retrieval.

Question → LLM → Answer

### RAG-based LLM
The system retrieves relevant document chunks and uses them as context.

Question → Embedding → FAISS Retrieval → Top-k Chunks → LLM → Answer

The RAG pipeline includes:

- Document ingestion (.docx)
- Text chunking (chunk size: 512, overlap: 64)
- Embedding generation (text-embedding-3-small)
- FAISS vector index
- Top-k retrieval (k=5)
- Answer generation (gpt-4o-mini)

All experiments are conducted under fixed conditions to ensure reproducibility.

---

## Experiment Setup

- Number of questions: 20
- Evaluation metric: Accuracy (decision correctness)
- Same dataset used for both Vanilla and RAG

---

## Results

| Model        | Accuracy |
|-------------|--------|
| Vanilla LLM | 0.55   |
| RAG-based LLM | 0.75 |

RAG improves accuracy by 20 percentage points.

This improvement suggests that many errors in the vanilla LLM originate from missing or incorrectly recalled policy rules, which can be mitigated through retrieval.

---

## Error Analysis

Representative cases:

| QID | Gold | Vanilla | RAG | Improvement | Error Type | Notes |
|-----|------|--------|-----|------------|------------|------|
| Q-08 | yes | no | yes | RAG fixed | vanilla hallucination | Incorrect rejection by vanilla |
| Q-09 | yes | no | yes | RAG fixed | vanilla hallucination | Multi-rule reasoning required |
| Q-10 | no | yes | no | RAG fixed | vanilla hallucination | Misinterpreted condition |
| Q-14 | yes | no | yes | RAG fixed | vanilla hallucination | Exception rule applied |
| Q-11 | yes | no | no | none | reasoning error | Boundary misinterpretation |
| Q-12 | yes | no | uncertain | none | source gap | Missing or unclear evidence |

Key findings:

- Vanilla LLM frequently hallucinates or misapplies policy rules
- RAG reduces such errors by grounding answers in retrieved documents
- Remaining errors are due to reasoning limitations and incomplete document coverage

---

## Project Structure

policy-rag-study/

configs/        # experiment configurations  
data/           # dataset and policy document  
src/            # core RAG pipeline implementation  
scripts/        # experiment execution  
experiments/    # experiment outputs  
results/        # analysis outputs  

---

## Reproducibility

- All model settings (embedding, LLM, temperature) are fixed via config
- Each experiment run saves config.yaml, predictions.json, and evaluation.json
- Experiments can be reproduced under identical conditions

---

## Usage

1. Install dependencies
   pip install -r requirements.txt

2. Set environment variables
   Set OPENAI_API_KEY in .env

3. Build index
   python scripts/ingest_policy.py --config configs/rag.yaml

4. Run experiments
   python scripts/run_experiment.py --config configs/vanilla.yaml --experiment vanilla --run-name run1
   python scripts/run_experiment.py --config configs/rag.yaml --experiment rag_baseline --run-name run1

5. Evaluate
   python scripts/evaluate.py --experiment rag_baseline/run1

---

## Future Work

- Investigating the impact of chunk size on retrieval performance
- Improving reasoning through structured rule extraction
- Expanding the dataset for more robust evaluation

---

## Author

Independent research project for AI graduate school application.
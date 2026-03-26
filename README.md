# Policy RAG Study

A research project investigating the limitations of Retrieval-Augmented Generation (RAG) for policy QA, showing that improving retrieval alone is insufficient without robust rule-based reasoning.

## Overview

This project investigates the effectiveness and limitations of Retrieval-Augmented Generation (RAG) for policy-based question answering (QA), focusing on rule-based decision tasks.

Unlike general QA, policy QA requires strict application of rules, including threshold conditions, exception handling, and multi-step reasoning.

This study goes beyond simple RAG evaluation and analyzes how retrieval configurations (chunk size, top-k, and overlap) and reasoning limitations affect performance.

- Vanilla LLM: 0.55 accuracy  
- RAG (baseline): 0.70 accuracy  
- Best configuration (overlap=64, top-k=5): **0.80 accuracy**

**Key Insight:**  
Improving retrieval alone is insufficient for policy QA; the primary bottleneck lies in rule-based reasoning rather than information access.

---

## Research Questions

This study investigates:

1. Does RAG improve policy QA accuracy compared to a vanilla LLM?
2. How do retrieval configurations (chunk size, top-k, overlap) affect performance?
3. What are the main sources of error in policy QA systems?
4. How do retrieval and reasoning interact in policy QA tasks?

---

## Dataset

- Source: Korean government policy document (소상공인 정책자금)
- Size: 20 manually constructed QA pairs
- Labels: `yes`, `no`, `selection_required`

Each question evaluates:

- Threshold boundary conditions
- Exception rules
- Industry restrictions
- Financial eligibility conditions
- Multi-condition reasoning

Ground truth answers were manually verified.

---

## Method

### Vanilla LLM

`Question → LLM → Answer`

### RAG-based LLM

`Question → Embedding → FAISS → Top-k Chunks → LLM → Answer`

Pipeline components:

- Document ingestion (.docx)
- Chunking (chunk size / overlap)
- Embedding (text-embedding-3-small)
- FAISS retrieval
- Top-k selection
- Answer generation (gpt-4o-mini)

---

## Experiments

We systematically evaluate key RAG components:

### 1. Chunk Size
- 256 / 512 / 1024

### 2. Top-k Retrieval
- 1 / 3 / 5 / 10

### 3. Chunk Overlap
- 0 / 64 / 128

### 4. Chain-of-Thought (CoT)
- reasoning-enhanced prompting

---

## Results

### Vanilla vs RAG

| Model | Accuracy |
|------|--------|
| Vanilla LLM | 0.55 |
| RAG (512) | 0.70 |

---

### Top-k Effect

| Top-k | Accuracy |
|------|--------|
| 1 | 0.65 |
| 3 | 0.70 |
| 5 | 0.80 |
| 10 | 0.80 |

→ Increasing top-k improves recall but saturates after k=5.

---

### Overlap Effect

| Overlap | Accuracy |
|--------|--------|
| 0 | 0.65 |
| 64 | 0.80 |
| 128 | 0.80 |

→ Overlap reduces rule fragmentation and significantly improves performance.

---

### Joint Effect (Top-k + Overlap)

| Overlap \ Top-k | 3 | 5 | 10 |
|----------------|---|---|----|
| 0 | 0.60 | 0.65 | 0.65 |
| 64 | 0.70 | **0.80** | 0.80 |
| 128 | 0.70 | 0.80 | 0.75 |

→ Optimal configuration: **overlap=64, top-k=5**

---

### CoT Effect

| Setting | Accuracy |
|--------|--------|
| Baseline | 0.80 |
| + CoT | ~0.82 |

→ CoT provides minor improvements but does not fully resolve reasoning errors.

---

## Error Analysis

Key error types:

- **Retrieval Failure**
- **Partial Retrieval**
- **Reasoning Failure**
- **Normalization Failure**

Findings:

- RAG reduces hallucination errors significantly
- Remaining errors are dominated by reasoning failures
- Boundary conditions and exception rules are the hardest cases

---

## Key Findings

- Retrieval improves performance, but is not sufficient
- Chunk size, overlap, and top-k critically affect performance
- There exists an optimal retrieval configuration
- Policy QA is fundamentally a **rule-based reasoning problem**

---

## Project Structure


policy-rag-study/
├── configs/
├── data/
├── src/
├── scripts/
├── experiments/
├── results/


---

## Reproducibility

- All experiments use fixed configurations
- Each run saves:
  - config
  - predictions
  - evaluation results

---

## Usage

1. Install dependencies

pip install -r requirements.txt


2. Set API key

export OPENAI_API_KEY=...


3. Build index

python scripts/ingest_policy.py --config configs/rag.yaml


4. Run experiments

python scripts/run_rag.py --config configs/rag.yaml


---

## Future Work

- Rule-aware retrieval methods
- Structured reasoning for policy rules
- Larger-scale policy QA benchmarks

---

## Author

Independent research project for AI graduate school application
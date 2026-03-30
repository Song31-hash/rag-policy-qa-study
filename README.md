# Policy RAG Study

A research project investigating the limitations of Retrieval-Augmented Generation (RAG) for policy QA, showing that improving retrieval alone is insufficient without robust rule-based reasoning.

---

## Overview

This project evaluates the effectiveness of Retrieval-Augmented Generation (RAG) for policy-based question answering (QA), with a focus on rule-based decision tasks.

Unlike general QA, policy QA requires strict application of rules, including threshold conditions, exception handling, and multi-step reasoning.

This study analyzes how retrieval configurations (chunk size, top-k, and overlap) affect performance and identifies the primary sources of error.

### Key Results

- Vanilla LLM: **0.60 accuracy**
- RAG (baseline): **0.75 accuracy**
- RAG + CoT: **up to 0.85 accuracy**

### Key Insight

Improving retrieval alone is not sufficient for policy QA.  
The primary bottleneck lies in **rule-based reasoning and answer normalization**, rather than information access.

---

## Research Questions

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
- Multi-condition reasoning
- Program track selection

Ground truth answers were manually verified.

---

## Method

### Vanilla LLM

Question → LLM → Answer

### RAG-based LLM

Question → Embedding → FAISS → Top-k Chunks → LLM → Answer

Pipeline components:

- Document ingestion (.docx)
- Chunking (chunk size / overlap)
- Embedding (text-embedding-3-small)
- FAISS retrieval
- Top-k selection
- Answer generation (gpt-4o-mini)

---

## Experiments

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
| Vanilla LLM | 0.60 |
| RAG (512) | 0.75 |

→ RAG significantly improves performance on policy QA.

---

### Top-k Effect

- Performance is highly sensitive to retrieval depth
- **Top-k=5 provides the most stable results**
- Larger top-k introduces noise without further gains

---

### Overlap Effect

- Overlap improves performance when combined with appropriate top-k
- Best results observed at **overlap=64–128 with top-k=5**
- Effect is **not monotonic** and depends on retrieval depth

---

### Joint Effect (Top-k + Overlap)

| Overlap \ Top-k | 3 | 5 | 10 |
|----------------|---|---|----|
| 0 | 0.75 | 0.70 | 0.65 |
| 64 | 0.55 | 0.75 | 0.65 |
| 128 | 0.60 | 0.75 | 0.65 |

→ High performance is consistently observed around **top-k=5**

---

### CoT Effect

| Setting | Accuracy |
|--------|--------|
| Baseline (top-k=5, overlap=64) | 0.75 |
| + CoT | 0.85 |

→ Chain-of-Thought significantly improves performance  
→ However, it **does not fully resolve reasoning errors**

---

## Error Analysis

Main error types:

- **Reasoning Failure**
- **Normalization Failure**

Key findings:

- Errors are not primarily caused by retrieval failure
- Even with correct context, models often fail to apply rules correctly
- Boundary conditions and exception rules are the most challenging cases

---

## Key Findings

- RAG improves performance but is not sufficient
- Retrieval configuration (top-k, overlap) significantly affects results
- **Top-k is a more stable factor than overlap**
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

### 1. Install dependencies

pip install -r requirements.txt

### 2. Set API key

export OPENAI_API_KEY=...

### 3. Build index

python scripts/ingest_policy.py --config configs/rag.yaml

### 4. Run experiments

python scripts/run_rag.py --config configs/rag.yaml

---

## Future Work

- Rule-aware retrieval methods
- Structured reasoning for policy rules
- Larger-scale policy QA benchmarks

---

## Author

Independent research project for AI graduate school application
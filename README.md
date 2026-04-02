# Policy RAG Study

### Retrieval vs Reasoning in Policy QA Systems

---

## Overview

This project investigates whether Retrieval-Augmented Generation (RAG) improves policy-based question answering (QA) and identifies reasoning as the primary bottleneck.

Unlike prior RAG studies that focus on missing information, this work examines a different failure mode: cases where the correct information is retrieved, but the model fails to apply it correctly.

Policy QA requires strict rule-based decision making (e.g., thresholds, exceptions, multi-condition logic), making it fundamentally different from semantic QA tasks.

---

## Key Results

* Vanilla LLM: **0.60 accuracy**
* RAG (baseline): **0.75 accuracy**
* RAG + CoT: **0.85 accuracy**

**Key Insight:**
Retrieval improves performance but is not sufficient.
The primary bottleneck lies in **rule-based reasoning**, not information access.

---

## Research Questions

1. Does retrieval improve policy QA accuracy?
2. How do retrieval configurations (top-k, overlap) affect performance?
3. Why do errors persist even when correct information is retrieved?

---

## Method

### Pipeline

Vanilla:
Question → LLM → Answer

RAG:
Question → Embedding → Retrieval → LLM → Answer

### Key Components

* Embedding: `text-embedding-3-small`
* Retrieval: FAISS
* LLM: `gpt-4o-mini`
* Controlled variables: top-k, overlap

---

## Experimental Findings

### Top-k

* Performance peaks at **k=5**
* Larger k introduces noise without consistent gains

### Overlap

* Improves performance when combined with appropriate top-k
* Best result at **overlap=128, top-k=5 (0.80)**
* Not strictly monotonic across configurations

### Interaction

Performance is determined by the interaction between retrieval depth and chunk overlap.

---

## Error Analysis

Main error types:

* **Reasoning Failure**
* **Normalization Failure**

Even when correct rules are retrieved, models often fail to consistently apply deterministic rules.

This indicates that errors are primarily caused by reasoning limitations rather than retrieval failure.

---

## Key Finding

Policy QA performance is fundamentally constrained not by information access, but by the ability to apply structured rules.

**Policy QA is a reasoning problem, not a retrieval problem.**

---

## Reproducibility

The project is fully reproducible through a config-driven pipeline.

### Example Experiment

```bash
python scripts/ingest_policy.py --config configs/experiment/overlap_64_topk_5.yaml
python scripts/run_rag.py --config configs/experiment/overlap_64_topk_5.yaml
python scripts/evaluate.py --experiment overlap_64_topk_5
```

All experiments store:

* config
* predictions
* evaluation results

---

## Repository

https://github.com/your-repo-link

The repository provides a fully reproducible experimental pipeline,
including all configurations, predictions, and evaluation results used in this study.

---

## Contribution

**Technical**

* Config-driven, reproducible RAG pipeline
* Systematic analysis of retrieval parameters

**Empirical**

* Retrieval improves performance but is insufficient
* Reasoning is the dominant bottleneck in policy QA

---

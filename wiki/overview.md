---
title: "Project overview"
type: overview
tags: [cs541, rag, meta]
created: 2026-10-11
updated: 2026-10-28
status: active
---

# FinSight-AI — Project overview

> Living synthesis. Read first in every session. Update after every milestone.

## Current status

**Phase:** proposal finalized → E1 experimentation (Week 10 tasks).
**Done:** proposal ([PDF](../raw_sources/references/CS541_RAG_Project_Proposal_Final.pdf)); repo scaffolded; M1 metrics, M2 data loading, M3 embedding cache, merge I1 (PR #1); M7 PCA merged via I3 (PR #2); M4 TorchKNN + M6 FAISS Flat/IVF + bonus KMeansIndex pushed on `h/e1-retrieval` (PR I2, pending J's review); M5 SciFact smoke run passed 2026-10-28 (Recall@10 0.7833, debug only).
**Next:** PR I2 (PR #3) review by J · E2 PCA on top of the chosen ivf nprobe 16 config (J) · E1 corpus-size sweep 10k/25k/57,638 (H) · M8 manual check of the unanswerable set (shared) · E3 generation runs (J) · Streamlit demo · final report and 10-min video (see [guide.md](../agents/workflow/guide.md) milestones).

## Problem

RAG answers are only as good as retrieval. We study, on financial QA (FiQA), how retrieval choices trade
accuracy against latency and memory, and when the system should refuse to answer.

## Research questions

- **RQ1 — Search algorithm:** exact (PyTorch brute-force KNN, FAISS Flat) vs. approximate (FAISS IVF):
  retrieval quality, latency, memory.
- **RQ2 — Dimensionality:** PCA 384 → 128 / 64 / 32: quality vs. efficiency.
- **RQ3 — Reliability:** top-k and similarity threshold τ vs. answer quality and correct refusal of unanswerable questions.

## Fixed setup

| Item | Choice |
|---|---|
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, L2-normalized |
| Similarity | inner product on unit vectors (= cosine) |
| Retrieval unit | whole document (qrels are document-level); chunking only in the demo |
| LLM | Groq `openai/gpt-oss-120b`, temperature 0, reasoning effort low → [decisions/llm_choice.md](decisions/llm_choice.md) |
| Seed | 42 |

## Experiments

| Exp | Owner | Summary page | Status |
|---|---|---|---|
| E1 retrieval method | H | [e1_summary](experiments/e1_summary.md) | grid complete on FiQA dev (6 runs + bonus); trade-off chosen: ivf nprobe 16 (2.82 pt for 6.7×); latency re-measured at natural thread counts |
| E2 PCA | J | [e2_summary](experiments/e2_summary.md) | not started |
| E3 reliability | J | [e3_summary](experiments/e3_summary.md) | not started |
| Bonus K-means index | H | — | optional |

## Metrics

Retrieval: Recall@5, Recall@10, MRR@10, nDCG@10 · Efficiency: mean / p95 latency (ms, single query, warm-up
excluded, mean of 3 runs), index build time, index memory (MB) · Generation (~100 test questions): 1–3 rubric
for correctness + groundedness, 30 double-scored by both members · Refusal: answer rate, correct-refusal rate vs. τ.

## Open questions

- Which LLM (Colab small model vs. free API)? → J
- Exact deliverable template for the final report (CS541_Project_template.pdf not yet received).

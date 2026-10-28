---
title: "Bonus run 001 — our KMeansIndex vs FAISS IVF"
type: experiment-run
tags: [bonus, e1, fiqa, kmeans]
created: 2026-10-28
updated: 2026-10-28
status: active
---

# Bonus run 001 — KMeansIndex at 240 cells / nprobe 16

Our own K-means index, run at the same settings as `run_e1_005` so the two are directly comparable.
Source: `experiments/results/run_bonus_001/metrics.json`.

## Hypothesis

Our PyTorch `KMeansIndex` reaches Recall@10 within 3 points of `faiss_ivf` at identical
`n_clusters` / `nprobe` (240 / 16).

## Config diff vs run_e1_005

One factor changed: `method: faiss_ivf` → `method: kmeans_index`,
`method_params: {nlist: 240, nprobe: 16}` → `{n_clusters: 240, nprobe: 16}`.

## Metrics (57,638 docs, 500 queries, dim 384)

| Metric | run 001 (exact) | run 005 (faiss_ivf 16) | bonus 001 (KMeans 16) |
|---|---|---|---|
| Recall@5 | 0.3797 | 0.3542 | 0.3569 |
| Recall@10 | 0.4665 | 0.4383 | **0.4435** |
| MRR@10 | 0.4532 | 0.4277 | 0.4332 |
| nDCG@10 | 0.3881 | 0.3631 | 0.3674 |
| latency mean | 0.6704 ms | 0.0892 ms | 1.5433 ms |
| latency p95 | 0.7674 ms | 0.1065 ms | 1.9781 ms |
| build time | 0.00053 s | 0.1829 s | 0.7650 s |
| index memory | 84.43 MB | 84.43 MB | 85.22 MB |

## Verdict

**Accuracy claim confirmed — and slightly better than predicted.** Our index beats FAISS IVF by
0.52 points of Recall@10 (0.4435 vs 0.4383) and is only 2.31 points below exact search.
Two independent implementations of the same idea landing within half a point is a good sign that
both are correct.

**But the latency result is the real finding.** Our index is 17× slower than FAISS IVF
(1.978 vs 0.107 ms p95) and even 2.6× slower than plain exact search.

## Why, and what to say in the report

- The search loop in `KMeansIndex.search` iterates over queries one at a time in Python and gathers
  candidate ids per query, so per-query Python and tensor-launch overhead dominates. FAISS is C++
  with fused scanning. Same asymptotic complexity (CLO2), very different constants.
- Build time 0.765 s vs 0.183 s: 20 Lloyd iterations in PyTorch, single-threaded.
- Memory 85.22 vs 84.43 MB: +0.79 MB is exactly centroids (240 × 384 × 4) plus the inverted-list
  id array (57,638 × 8) — the index overhead our exact baseline does not have.
- This is a fair, honest conclusion for the bonus: **our implementation reproduces the accuracy of
  an industrial ANN index but not its speed.** That is the point of comparing them.

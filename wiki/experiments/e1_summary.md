---
title: "E1 — Retrieval method: summary"
type: experiment
tags: [e1]
created: 2026-10-11
updated: 2026-10-28
status: active
owner: H
---

# E1 — Retrieval method

Full grid finished on FiQA dev (2026-10-28): `torch_knn`, `faiss_flat`, and `faiss_ivf` at
`nprobe ∈ {1, 4, 16, 64}` with `nlist = 240 ≈ √57,638`. All runs: 57,638 docs, 500 dev queries,
dim 384, k = 10, seed 42. FiQA `test` stays locked.

## Best so far (dev)

| Config | Recall@10 | nDCG@10 | p95 latency (ms) | Memory (MB) | Run |
|---|---|---|---|---|---|
| exact — torch_knn | 0.4665 | 0.3881 | 0.7674 | 84.43 | [001](e1_run_001.md) |
| exact — faiss_flat | 0.4665 | 0.3881 | 1.2350 | 84.43 | [002](e1_run_002.md) |
| ivf nprobe 16 | 0.4383 | 0.3631 | 0.1065 | 84.43 | [005](e1_run_005.md) |
| ivf nprobe 64 | 0.4665 | 0.3882 | 0.3659 | 84.43 | [006](e1_run_006.md) |

Chosen trade-off for downstream work: **ivf nprobe 16** — 2.82 points of Recall@10 for a 7.2× p95
speed-up. If accuracy matters more than speed, nprobe 64 is exact at 2.1×.

## Run history

| Run | Hypothesis | Recall@10 | nDCG@10 | p95 ms | Verdict |
|---|---|---|---|---|---|
| [000](e1_run_000.md) | SciFact pipeline runs and gives valid metrics | 0.7833 | 0.6451 | 0.0316 | confirmed (debug only) |
| [001](e1_run_001.md) | exact torch_knn baseline on FiQA dev | 0.4665 | 0.3881 | 0.7674 | baseline |
| [002](e1_run_002.md) | faiss_flat matches torch_knn, only latency differs | 0.4665 | 0.3881 | 1.2350 | accuracy confirmed, speed rejected |
| [003](e1_run_003.md) | nprobe 1 loses > 15 pt of Recall@10 | 0.2831 | 0.2387 | 0.0153 | confirmed (−18.35 pt) |
| [004](e1_run_004.md) | nprobe 4 is within 5 pt of exact | 0.3734 | 0.3063 | 0.0356 | rejected (−9.32 pt) |
| [005](e1_run_005.md) | nprobe 16 within 2 pt and ≥3× faster | 0.4383 | 0.3631 | 0.1065 | partly: 7.2× yes, 2.82 pt no |
| [006](e1_run_006.md) | nprobe 64 within 1 pt, speed-up shrinks | 0.4665 | 0.3882 | 0.3659 | confirmed (0.00 pt, 2.1×) |
| [bonus 001](bonus_run_001.md) | our KMeansIndex within 3 pt of faiss_ivf | 0.4435 | 0.3674 | 1.9781 | confirmed (+0.52 pt vs IVF) |

## Findings so far

- **Correctness control passes.** `faiss_flat` and `torch_knn` agree to all four metrics, so our
  PyTorch implementation is validated against a reference.
- **The accuracy–latency curve has a clear knee.** Recall@10 gap vs exact: −18.35 / −9.32 / −2.82 /
  0.00 points for nprobe 1 / 4 / 16 / 64, at 50× / 21.5× / 7.2× / 2.1× p95 speed-up. Accuracy is
  bought back at a worsening rate: the first cells are cheap, the last ones cost almost all the
  speed advantage.
- **nprobe 64 is already exact.** Visiting 27% of the corpus reproduces the exact top-10, so the
  loss at low nprobe comes from cell assignment, not from how many vectors are scored.
- **Our KMeansIndex matches FAISS IVF on accuracy (+0.52 pt) but is 17× slower.** Same asymptotics,
  different constants: per-query Python loop vs. fused C++ scan. Good bonus material.
- **Build time and memory only start to matter for the approximate indexes**: 0.0005 s (exact, no
  preprocessing) vs 0.18 s (IVF K-means) vs 0.76 s (our 20 Lloyd iterations); memory 84.43 MB
  payload vs 85.22 MB with centroids and inverted lists.

## Caveat that must be reported

Every run was measured with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`, the macOS workaround for
the torch + faiss-cpu OpenMP clash. All methods were constrained equally, so **accuracy comparisons
and speed ratios between approximate methods are valid**. Two things are affected and must be
re-measured before the report: (1) `faiss_flat` being 1.7× slower than `torch_knn` is an artefact of
FAISS being forced to one thread — a separate check showed torch barely changes between 1 and 5
threads for this matmul (0.608 vs 0.603 ms), because single-query search over 84 MB is
memory-bandwidth bound; (2) absolute latency values are pessimistic for FAISS.

## Next hypothesis

After re-installing `faiss-cpu` from conda-forge, re-run 001/002/005/006: if FAISS gets its threads
back, `faiss_ivf` at nprobe 16 should be at least 10× faster than `torch_knn` on p95 while the
Recall@10 gap stays at 2.8 points. Then hand nprobe 16 to E2 (PCA on top of it).

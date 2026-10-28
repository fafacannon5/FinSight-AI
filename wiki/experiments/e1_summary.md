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

Chosen trade-off for downstream work: **ivf nprobe 16** — 2.82 points of Recall@10 for a 6.7× p95
speed-up (7.2× as measured under the single-thread workaround; see the caveat below). If accuracy
matters more than speed, nprobe 64 is exact at ~2×.

## Run history

| Run | Hypothesis | Recall@10 | nDCG@10 | p95 ms | Verdict |
|---|---|---|---|---|---|
| [000](e1_run_000.md) | SciFact pipeline runs and gives valid metrics | 0.7833 | 0.6451 | 0.0316 | confirmed (debug only) |
| [001](e1_run_001.md) | exact torch_knn baseline on FiQA dev | 0.4665 | 0.3881 | 0.7674 | baseline |
| [002](e1_run_002.md) | faiss_flat matches torch_knn, only latency differs | 0.4665 | 0.3881 | 1.2350 → 0.8285 | accuracy confirmed, latency a tie |
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

## Latency caveat, and how we closed it

Every grid run was measured with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`, the macOS workaround
for the torch + faiss-cpu OpenMP clash (see README). That caps FAISS at one thread, so we re-measured
every back end in a process that imports only the library it needs — FAISS via
`scripts/bench_faiss_isolated.py`, `torch_knn` via `run_experiment.py` with no env vars at all. Both
then run at their natural thread count. Same protocol otherwise (single-query batches, warm-up
excluded, 3 passes, 500 dev queries).

| Method | p95, `OMP_NUM_THREADS=1` | p95, natural threads | change |
|---|---|---|---|
| torch_knn | 0.7674 | 0.7645 | −0.4% |
| faiss_flat | 1.2350 | 0.8285 | −33% |
| ivf nprobe 1 | 0.0153 | 0.0164 | +7% |
| ivf nprobe 4 | 0.0356 | 0.0378 | +6% |
| ivf nprobe 16 | 0.1065 | 0.1137 | +7% |
| ivf nprobe 64 | 0.3659 | 0.3913 | +7% |

What this means:

- **No accuracy number changes** — same index, same ids, same seed.
- **The IVF speed-ups barely move**: nprobe 16 goes from 7.2× to **6.7×** vs `torch_knn`, still well
  above the 3× we hypothesised. Use 6.7× in the report.
- **The one number that was wrong**: `faiss_flat` looked 1.6× *slower* than `torch_knn`. At natural
  threads it is 0.83 vs 0.76 ms — an 8% gap, i.e. the two exact back ends are effectively tied, which
  is what you would expect from two implementations of the same O(N·d) scan.
- **Why**: single-query search over an 84 MB matrix is memory-bandwidth bound. FAISS Flat gains ~33%
  from threads because it touches the whole corpus; FAISS IVF gains nothing (per-query work is tiny,
  thread overhead dominates); torch is flat (0.608 ms at 1 thread vs 0.603 ms at 5 in a direct check).
- **Noise floor**: repeated `run_e1_001` gave p95 0.691 / 0.765 / 0.767 ms, so differences below about
  10% should be reported as ties.

Note: installing `faiss-cpu` from conda-forge does *not* remove the clash — our `torch` keeps its own
`libomp`, so two runtimes would still load.

## Next hypothesis

Hand `ivf nprobe 16` to E2: if PCA 384 → 128 is applied on top of it, Recall@10 should drop by less
than 2 points while p95 improves by ~2×, because the scan cost is linear in `d`. Then sweep corpus
size (10k / 25k / 57,638) to show how the speed-up grows with `N`.

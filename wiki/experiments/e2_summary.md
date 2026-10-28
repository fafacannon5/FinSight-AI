---
title: "E2 — PCA dimensionality: summary"
type: experiment
tags: [e2]
created: 2026-10-11
updated: 2026-10-28
status: active
owner: J
---

# E2 — PCA dimensionality

Grid finished on FiQA dev (2026-10-28): `pca_dim` ∈ {none (384), 128, 64, 32} with exact `torch_knn`,
so any accuracy change is caused by PCA alone (decision: J, to keep E2 separate from IVF's own −2.82 pt).
57,638 docs, 500 dev queries, k = 10, seed 42. FiQA `test` stays locked.

## Results (dev)

| Dim | Variance kept | Recall@10 | Δ Recall@10 | nDCG@10 | p95 (ms) | p95 speed-up | Memory (MB) |
|---|---|---|---|---|---|---|---|
| 384 | 1.000 | 0.4665 | +0.00 pt | 0.3881 | 2.567 | 1.0× | 84.43 |
| 128 | 0.795 | 0.4412 | -2.54 pt | 0.3586 | 0.975 | 2.6× | 28.14 |
| 64 | 0.614 | 0.3671 | -9.94 pt | 0.2937 | 0.663 | 3.9× | 14.07 |
| 32 | 0.456 | 0.2686 | -19.79 pt | 0.2079 | 0.464 | 5.5× | 7.04 |

## Run history

| Run | Hypothesis | Recall@10 | nDCG@10 | p95 ms | Memory MB | Verdict |
|---|---|---|---|---|---|---|
| [000](e2_run_000.md) | baseline — exact torch_knn at full 384-d, re-run in the same session as the PCA runs so latency is comparable | 0.4665 (+0.00 pt) | 0.3881 | 2.567 | 84.43 | Baseline established |
| [001](e2_run_001.md) | PCA to 128-d keeps Recall@10 within 1 point of 384-d while cutting index memory 3x | 0.4412 (-2.54 pt) | 0.3586 | 0.975 | 28.14 | Rejected (narrowly) |
| [002](e2_run_002.md) | PCA to 64-d keeps Recall@10 within 3 points of 384-d while cutting index memory 6x | 0.3671 (-9.94 pt) | 0.2937 | 0.663 | 14.07 | Rejected |
| [003](e2_run_003.md) | PCA to 32-d loses more than 5 points of Recall@10 (only 45.6% of the variance is kept) | 0.2686 (-19.79 pt) | 0.2079 | 0.464 | 7.04 | Confirmed |

## Findings

- **Memory scales exactly with dimension** (84.43 → 28.14 → 14.07 → 7.04 MB), as expected for a flat index.
- **Recall falls less than the variance does, but the gap closes fast.** Relative Recall@10 loss vs variance
  discarded: 128-d 5.4% vs 20.5% · 64-d 21.3% vs 38.6% · 32-d 42.4% vs 54.4%. The first discarded directions are
  mostly noise for retrieval; below 128-d PCA starts cutting directions that carry the ranking signal.
  In absolute terms the Recall@10 loss roughly quadruples, then doubles, as the dimension halves (−2.54 → −9.94 → −19.79 pt).
- **128-d is the only reasonable trade-off:** 3× less memory and 2.6× faster p95 for 2.54 pt of Recall@10.
- **Speed-up is sub-linear** (3× fewer dims → 2.6×; 12× fewer → 5.5×): fixed per-query costs (query transform,
  top-k selection, Python overhead) do not shrink with the dimension.

## Caveats

- Latency is from one machine under background load; the 384-d run varied 2.57–3.48 ms p95 across three
  repeats (the table uses the last repeat, which is what `metrics.json` holds). Accuracy is deterministic.
- J's 384-d latency (2.57 ms p95) is ~3× H's E1 run 001 (0.77 ms) for identical accuracy: different machines.
  Never mix E1 and E2 latencies in one table without saying so.

## Next

- Optional: 96-d run to locate the knee between 128 and 64 (grid in experiment_loop.md lists only 128/64/32,
  so this needs a note before running).
- Optional: PCA 128 + IVF nprobe 16, to see whether the two losses add up.

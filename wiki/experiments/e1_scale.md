# E1-scale: corpus size sweep (runs e1s_001–004)

**Owner:** H · **Date:** 2026-11-12 · **Split:** FiQA dev (tuning split; test untouched).
**Question.** Does the FAISS IVF speed-up over exact search grow with corpus size, as assumed when config B was chosen?

## Config

Main-grid settings unchanged (384-d MiniLM embeddings, k = 10, seed 42, `OMP_NUM_THREADS = 1` latency
protocol). The corpus is subsampled with `corpus_size` — **all** qrels-relevant docs are kept, the rest
are filled at random, so the sub-tasks are nested, not disjoint. `nlist = 240` is **fixed across sizes**
(not √N): the probe fraction (16/240 ≈ 6.7 % of the corpus) and the configuration stay identical, so only
N varies. Configs: `experiments/configs/run_e1s_00{1..4}.yaml`.

## Results

| N | torch_knn p95 (ms) | ivf16 p95 (ms) | speed-up | torch R@10 | ivf16 R@10 | Δ (pt) |
|---|---|---|---|---|---|---|
| 10,000 | 0.0970 | 0.0285 | **3.40×** | 0.6295 | 0.6012 | −2.83 |
| 25,000 | 0.3199 | 0.0493 | **6.49×** | 0.5372 | 0.5124 | −2.48 |
| 57,638 | 0.7645 | 0.1065 | **7.18×** | 0.4665 | 0.4383 | −2.82 |

57k rows are from `run_e1_001` / `run_e1_005` metrics.json, same protocol (the summary table's 0.7674 ms
is a repeat of the same run — within the ±10 % noise floor). Source:
`experiments/results/run_e1s_00{1..4}/metrics.json`.

## Verdict: confirmed, with a nuance

- **Speed-up grows with N — 3.4× → 6.5× → 7.2× — but sub-linearly, toward the probe-ratio bound.**
  With `nlist` fixed, IVF scores `nprobe · N/nlist ≈ 6.7 %` of the corpus, so its cost also grows
  linearly in N, just with a ~15× smaller slope (240/16). The measured 7.2× sits below that bound
  because fixed per-query costs (top-k heap, memory bandwidth) shrink the relative advantage.
- **The accuracy loss of IVF is independent of corpus size** (−2.83 / −2.48 / −2.82 pt): the dev-grid
  choice of `nprobe 16` does not become better or worse as the corpus grows.
- **Recall@10 falls with N for both methods** (0.63 → 0.54 → 0.47). This is *task difficulty*, not
  degradation: the subsample keeps every relevant doc, so a larger N only adds distractors. Do not
  read the N-trend as a quality drop.

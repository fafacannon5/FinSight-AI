# Wiki log

> Append-only, newest first. Format: `## [YYYY-MM-DD] {operation} | {description}`

## [2026-10-28] experiment | E1 grid complete on FiQA dev (runs 001-006 + bonus 001)

- FiQA dev, 57,638 docs, 500 queries, dim 384, k=10, nlist=240 (≈√N); first FiQA download (~28 MB)
  and corpus encoding (~64 s) after H's OK; caches written to `data/` (~88 MB, gitignored)
- Recall@10: torch_knn 0.4665 (baseline) · faiss_flat 0.4665 (identical — correctness control passes) ·
  ivf nprobe 1/4/16/64 = 0.2831 / 0.3734 / 0.4383 / 0.4665; bonus KMeansIndex 240/16 = 0.4435
- p95 latency (ms): 0.7674 / 1.2350 / 0.0153 / 0.0356 / 0.1065 / 0.3659 / bonus 1.9781
  → speed-ups vs baseline 0.62× / 50× / 21.5× / 7.2× / 2.1×; our KMeans 0.39×
- Accuracy gaps vs exact: −18.35 / −9.32 / −2.82 / 0.00 pt; bonus −2.31 pt (and +0.52 pt vs faiss_ivf)
- Chosen trade-off: ivf nprobe 16 (2.82 pt for 7.2×); nprobe 64 is already exact at 2.1×
- Caveat recorded: all runs single-threaded (OMP_NUM_THREADS=1 workaround); faiss_flat being slower
  than torch_knn is an artefact, latency needs a re-run on a conda faiss-cpu install
- Pages updated: experiments/e1_run_001..006.md + bonus_run_001.md (new), e1_summary.md, overview.md

## [2026-10-28] experiment | run_e1_000: SciFact smoke test passes (M5)

- `python scripts/run_experiment.py experiments/configs/run_e1_000_scifact_smoke.yaml` →
  `experiments/results/run_e1_000/metrics.json`
- SciFact test, torch_knn, k=10: Recall@5 0.7379 · Recall@10 0.7833 · MRR@10 0.6047 · nDCG@10 0.6451;
  n_docs 5183, n_queries 300, dim 384; latency mean 0.0283 ms / p95 0.0316 ms; memory 7.59 MB
- All four M5 acceptance checks pass; random-ranking baseline 0.00218 → observed is 359× the baseline
- First downloads on this machine, after H's OK: `datasets` was missing → installed; BeIR/scifact (~4.5 MB);
  all-MiniLM-L6-v2 (~87 MB); embedding caches rebuilt in `data/` (~8.5 MB, gitignored)
- Debug dataset only: SciFact numbers are never reported
- Branch `h/e1-retrieval` pushed (451709a, 7cbce53, 3095f43); PR I2 still pending J's review
- Pages updated: experiments/e1_run_000.md (new), experiments/e1_summary.md, overview.md

## [2026-10-25] milestone | M7 PCAReducer passes tests (branch j/e2-pca)

- `tests/test_pca.py` 4/4 pass (incl. sklearn match up to sign)
- Corpus-only sanity check on FiQA (no queries used): variance kept 128-d 0.795 · 64-d 0.614 · 32-d 0.456;
  rows unit-norm, no NaN; fit + transform ≈ 1 s
- Written by Claude at J's request after a PCA walkthrough + quiz; disclosed in AI_USAGE.md
- Next: merge I3 (PR j/e2-pca → main, reviewer H); E2 runs need H's TorchKNN (I2)
- Pages updated: overview.md

## [2026-10-25] merge | I1: shared/foundation → main (PR #1)

- M1–M3 on main as 111c4e9; `shared/foundation` deleted locally and on GitHub
- Merge was clicked from J's account; ask H to review PR #1 afterwards. Process: reviewer merges.

## [2026-10-25] milestone | M3 embedding cache built → ready for merge I1

- First model download with J's OK: sentence-transformers/all-MiniLM-L6-v2 (model.safetensors 87 MB) → ~/.cache/huggingface/
- encode(["hello"]) → (1, 384), norm 1.0 ✓
- data/fiqa_corpus_minilm.npy (57,638, 384) float32, all row norms 1.0000, finite ✓ — 195 s on Apple Silicon (MPS)
- data/scifact_corpus_minilm.npy (5,183, 384) ✓ — 30 s
- Cache now keyed by an id list (`*.ids.json`); alignment spot check: 5 random docs re-encoded, cos = 1.0 with cached rows
- Disclosure correction: embed.py, timing.py, run_experiment.py were full code in the original scaffold (AI_USAGE.md)
- Next: merge I1 (PR shared/foundation → main, reviewer H), then M4 TorchKNN (H)
- Pages updated: dataset/fiqa.md, overview.md

## [2026-10-25] milestone | M2 load_beir verified on SciFact + FiQA

- First dataset download with J's OK: BeIR/scifact (~4.5 MB) + BeIR/fiqa (~28 MB) → ~/.cache/huggingface/
- SciFact: 5,183 docs, 300 test queries ✓ · FiQA: 57,638 docs, 5,500 / 500 / 648 queries ✓
- Gotchas: qrels ids are int64 on the Hub (cast to str); FiQA dev split is named `validation`
- `load_beir` written by Claude at J's request; disclosed in AI_USAGE.md
- Next: M3 `embed.py` (first model download: all-MiniLM-L6-v2 — explain before running)
- Pages updated: dataset/scifact.md, dataset/fiqa.md, overview.md

## [2026-10-25] milestone | M1 metrics.py complete

- `recall_at_k`, `mrr_at_k`, `ndcg_at_k` implemented on branch `shared/foundation`; `tests/test_metrics.py` 7/7 pass
- Written by Claude at J's request (full code + line-by-line explanation); disclosed in AI_USAGE.md
- Next: M2 `data.py` (first dataset download — explain before running)
- Pages updated: overview.md

## [2026-10-11] decision | Framework adjustments after review

- Agents never commit or push on their own; they suggest commit messages
- Default user = Jing (J); ask J/H only when ownership matters
- Removed expected SciFact score; smoke test = valid, finite metrics + sanity checks
- PyTorch PCA from scratch kept, with sklearn as reference check only
- Merge workflow: PR per integration point (I1–I7), reviewer = other member, final tag v1.0
- No silent downloads: agents explain each external model/dataset before first use (conventions/external_resources.md)

## [2026-10-11] milestone | Repository scaffolded with agent framework

- AGENTS.md + workflows (guide, test, experiment loop, explain, review, query, ingest, lint)
- `src/finsight/` stubs with unit tests (red until implemented)
- Next: M1 `metrics.py`

## [2026-10-11] milestone | Proposal finalized

- Title: Design and Evaluation of a RAG Chatbot: Accuracy–Efficiency Trade-offs in Retrieval
- Team: Jing Cannon (J), Haiyan Zhu (H)
- Dataset: FiQA (main), SciFact (debug)

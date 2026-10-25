# Wiki log

> Append-only, newest first. Format: `## [YYYY-MM-DD] {operation} | {description}`

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

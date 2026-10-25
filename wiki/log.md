# Wiki log

> Append-only, newest first. Format: `## [YYYY-MM-DD] {operation} | {description}`

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

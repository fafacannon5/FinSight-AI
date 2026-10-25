---
title: "FiQA-2018 (BEIR)"
type: dataset
tags: [fiqa, beir]
created: 2026-10-11
updated: 2026-10-25
status: verified
---

# FiQA-2018 (BEIR version)

Financial opinion question answering; answers are forum posts.

| Property | Expected value | Verified? |
|---|---|---|
| Corpus documents | 57,638 | ☑ 57,638 (2026-10-25) |
| Queries total | 6,648 | ☑ 6,648 |
| Split | train 5,500 · dev 500 · test 648 | ☑ 5,500 / 500 / 648 queries |
| Labels | query → relevant doc ids (qrels) | ☑ 14,166 / 1,238 / 1,706 pairs; all scores = 1; every qrels doc exists in corpus |

Loading (verified 2026-10-25, M2): Hugging Face `BeIR/fiqa` configs `corpus` / `queries` (split names equal the
config names) + `BeIR/fiqa-qrels` splits `train`, **`validation`** (= our `dev`), `test`. `load_beir(..., "dev")` maps
to `validation`. qrels ids are int64 on the Hub → cast with `str()`. Download ≈ 28 MB, cached in
`~/.cache/huggingface/`; load ≈ 3 s from cache, so no extra `data/` cache. Test split: only counted, never evaluated.

Notes:
- Retrieval unit = whole document. Long docs are truncated by the encoder (MiniLM max 256 word pieces) —
  report as a limitation.
- Embedding cache: 57,638 × 384 × 4 B ≈ 88 MB → `data/` (gitignored).
- **ID-type pitfall.** All ids must be `str`. If ids mismatch (int `1` vs str `"1"`), nothing matches and every
  metric silently becomes 0 — no error. Where it can come back:
  1. Mapping search rows back to ids: FAISS / TorchKNN return row numbers; `doc_ids[row]` must use the same list,
     in the same order, as the embedding cache.
  2. Reading results back from JSON / CSV: ids may be parsed as numbers again.
  Guards: `data.check_ids` (called at the end of `load_beir`; unit-tested in `tests/test_data.py`) and
  `scripts/run_experiment.py`, which refuses to write `metrics.json` when Recall@10 == 0.

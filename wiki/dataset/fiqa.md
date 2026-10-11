---
title: "FiQA-2018 (BEIR)"
type: dataset
tags: [fiqa, beir]
created: 2026-10-11
updated: 2026-10-11
status: draft
---

# FiQA-2018 (BEIR version)

Financial opinion question answering; answers are forum posts.

| Property | Expected value | Verified? |
|---|---|---|
| Corpus documents | 57,638 | ☐ (M2) |
| Queries total | 6,648 | ☐ |
| Split | train 5,500 · dev 500 · test 648 | ☐ |
| Labels | query → relevant doc ids (qrels) | ☐ |

Loading (to verify in M2): Hugging Face `BeIR/fiqa` (configs `corpus`, `queries`) + `BeIR/fiqa-qrels`
(splits `train`, `validation`/`dev`, `test`). Record the exact names that worked here.

Notes:
- Retrieval unit = whole document. Long docs are truncated by the encoder (MiniLM max 256 word pieces) —
  report as a limitation.
- Embedding cache: 57,638 × 384 × 4 B ≈ 88 MB → `data/` (gitignored).

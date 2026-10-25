---
title: "SciFact (BEIR)"
type: dataset
tags: [scifact, beir, smoke-test]
created: 2026-10-11
updated: 2026-10-25
status: verified
---

# SciFact (BEIR version) — smoke test only

| Property | Expected value | Verified? |
|---|---|---|
| Corpus documents | 5,183 | ☑ 5,183 (2026-10-25) |
| Test queries | 300 | ☑ 300 (339 qrels pairs) |

Used to check the pipeline end to end (M5). Not used for reported results, so using its test split is allowed.
Done = the pipeline runs without error, metrics are finite and in [0, 1], Recall@10 ≥ Recall@5, and scores are
far above a random-ranking baseline. There is **no target score**: never change code to move a metric toward an
expected number. If a result looks wrong, find the cause with tests (`pytest`), not by tuning.

## Loading (verified 2026-10-25, M2)

- Hugging Face `BeIR/scifact` configs `corpus` (split `corpus`) and `queries` (split `queries`, 1,109 rows total);
  `BeIR/scifact-qrels` splits `train` (919 pairs) and `test` (339 pairs). No dev split.
- qrels `query-id` / `corpus-id` are **int64** on the Hub; corpus and query `_id` are strings → `load_beir` casts with `str()`.
- Download ≈ 4.5 MB, cached in `~/.cache/huggingface/`. Load time ≈ 3 s from cache.

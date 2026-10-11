---
title: "SciFact (BEIR)"
type: dataset
tags: [scifact, beir, smoke-test]
created: 2026-10-11
updated: 2026-10-11
status: draft
---

# SciFact (BEIR version) — smoke test only

| Property | Expected value | Verified? |
|---|---|---|
| Corpus documents | 5,183 | ☐ |
| Test queries | 300 | ☐ |

Used to check the pipeline end to end (M5). Not used for reported results, so using its test split is allowed.
Done = the pipeline runs without error, metrics are finite and in [0, 1], Recall@10 ≥ Recall@5, and scores are
far above a random-ranking baseline. There is **no target score**: never change code to move a metric toward an
expected number. If a result looks wrong, find the cause with tests (`pytest`), not by tuning.
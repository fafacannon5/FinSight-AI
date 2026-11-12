---
title: "Unanswerable set (FiQA dev, M8)"
type: dataset
tags: [e3, m8, refusal]
created: 2026-11-01
updated: 2026-11-01
status: active
---

# Unanswerable set — FiQA dev (M8)

## How it was built

1. `make_unanswerable(qrels, corpus_ids, n=50, seed=42)` picked 50 dev queries whose relevant docs no other
   query needs and removed all of them (116 docs; 57,522 docs left). Script: `scripts/e3_build_unanswerable.py`.
2. Exact TorchKNN at 384-d over the reduced corpus. No removed doc was retrieved; answerable Recall@10 0.4684.
3. **Manual check (J and H, 2026-11-01):** for each candidate, read the top-5 retrieved docs.
   **DROP** if any of them materially answers the question, **OK** only if none is sufficient.
   Sheet: `experiments/results/e3_unanswerable/review_completed.md` (gitignored).

## Result: 10 OK · 40 DROP

Removing the labelled docs rarely makes a FiQA question unanswerable: FiQA is a forum with many answers
per topic, and the qrels mark only some of them. 40 of 50 candidates were still answered by another
(unlabelled) post in the top-5.

**Verified unanswerable qids (use only these):**
`10054`, `1548`, `1616`, `1630`, `1658`, `4905`, `6164`, `7066`, `7084`, `7595`

| Group | n | Median top-1 cosine |
|---|---|---|
| Answerable dev queries | 450 | 0.6518 |
| Verified unanswerable (OK) | 10 | 0.549 |
| All 50 candidates (before the check) | 50 | 0.6347 |

Source: `experiments/results/e3_unanswerable/tau_curve.json`.

## Caveats

- Only 10 verified examples: each one moves the correct-refusal rate by 0.1.
- Verdicts are a joint J+H decision, not two independent checks. ChatGPT drafted verdicts and reasons; J and H
  discussed each candidate and decided (disclosed in `AI_USAGE.md`).
- Reproduce the candidates with `python scripts/e3_build_unanswerable.py --n 50` (seed 42).

## Related

[E3 summary](../experiments/e3_summary.md) · [E3 run 002 (τ)](../experiments/e3_run_002.md) · [FiQA](fiqa.md)

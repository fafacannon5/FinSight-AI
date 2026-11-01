---
title: "E3 — Reliability (top-k, threshold): summary"
type: experiment
tags: [e3]
created: 2026-10-11
updated: 2026-11-01
status: active
owner: J
---

# E3 — Reliability (top-k, threshold)

Answers are scored with the [scoring rubric](e3_rubric.md). Unanswerable questions: [verified set](../dataset/unanswerable.md).

## Best so far (dev)

| Setting | Value | Evidence | Run |
|---|---|---|---|
| top-k for generation | **k = 5** (correctness 2.53, groundedness 2.87, refusal rate 0.25) | 20 questions, J+H consensus | [run 001](e3_run_001.md) |
| refusal threshold | **τ = 0.60** on top-1 cosine (answer rate 0.76, correct refusal 0.80) | 450 answerable / 10 verified unanswerable | [run 002](e3_run_002.md) |

## Run history
| Run | Hypothesis | Key metric | Verdict |
|---|---|---|---|
| [001](e3_run_001.md) | larger k → more correct, less grounded | correctness 2.36 → 2.65, groundedness 3.00 → 2.65 (k 1 → 10) | confirmed (small n) |
| [002](e3_run_002.md) | top-1 cosine separates unanswerable questions | AUC 0.82 (verified) vs. 0.56 (unverified) | confirmed, τ = 0.60 |

## Findings so far

- More context helps correctness and reduces the model's own refusals, but costs groundedness.
- FiQA's incomplete labels dominate E3: 40 of 50 "unanswerable" candidates were still answerable.
  Without the manual check, threshold refusal would have looked useless.

## Open

- Fix k and τ in `wiki/decisions/final_configs.md` before the final test evaluation.

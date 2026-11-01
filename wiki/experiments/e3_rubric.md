---
title: "E3 — Answer scoring rubric (correctness / groundedness, 1–3)"
type: experiment
tags: [e3, scoring, rubric]
created: 2026-11-01
updated: 2026-11-01
status: active
owner: J
---

# E3 — Answer scoring rubric

Used for every manual score in E3 (`scoring.csv` of `run_e3_gen_*` on dev, and later the final test run).
J and H use **this same rubric**; otherwise the 30 double-scored answers say nothing about agreement.

## Two independent scores

| Column | Question it answers | Compare against |
|---|---|---|
| `correctness_1to3` | Is the answer **right**? | The labelled answer docs (marked ✅ in `scoring_sheet.md`) |
| `groundedness_1to3` | Is every claim **supported by the k docs the model was given**? | Only the docs in the prompt at that k |

Kept separate because they fail differently: *right but ungrounded* = the model answered from its own
knowledge (a RAG failure — the user cannot verify it); *grounded but wrong* = retrieval gave the wrong or
off-topic docs. Watching both shows which one improves as k grows.

## Scale

| Score | correctness | groundedness |
|---|---|---|
| **3** | Covers the main point of the labelled answer; no errors | Every claim is found in the given docs; cited ids are right |
| **2** | Partly right: misses a main point or has a minor error | Mostly supported; one or two claims have no source, or a citation points to the wrong doc |
| **1** | Wrong or does not address the question | The main content is not in the given docs (made up / outside knowledge) |

## Special cases

- **Model refused** (answer is the refusal text): leave both scores empty; write `refused` in `notes`.
  If the given docs *could* answer the question, write `refused-wrongly`. Refusals are counted as a
  refusal rate, not averaged into the scores.
- **`relevant_in_top_k` is empty** (no labelled doc in the first k): correctness is often low, but FiQA's
  labels are incomplete — judge the content; an unlabelled doc may still answer correctly.
- **Citation format** (`[id]`, `【id】`, `[d<id>]`): all count as the same citation; format is not scored.
- `scorer` = `J` or `H`. One row per scorer; for double-scored answers, copy the row and fill it in again.

## Analysis (after scoring)

Per k: mean correctness, mean groundedness (non-refused answers only), refusal rate.
Agreement on the 30 double-scored answers: exact-match rate and Cohen's κ per column.

## Related

[E3 summary](e3_summary.md) · [LLM choice](../decisions/llm_choice.md) · [Overview](../overview.md)

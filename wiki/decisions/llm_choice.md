---
title: "Decision: generation LLM"
type: decision
tags: [llm, m9, e3, demo]
created: 2026-10-28
updated: 2026-10-28
status: active
---

# Decision: generation LLM for M9 / E3 / demo

## Context

`rag.generate()` needs one fixed LLM (Hard rule 3) for E3 answer quality and the Streamlit demo.
Expected volume is small: ~30 dev questions × 3 values of k to choose k, then ~100 test questions once,
plus the demo — about 190 calls. The τ sweep needs no LLM (it only uses retrieval scores).
Machine: Apple M1 Pro, 17 GB RAM.

## Options

| Option | For | Against |
|---|---|---|
| A. Local small model (1.5B–3B, e.g. Qwen) on the Mac | free, no key, fully reproducible, offline demo | several-GB download; weak at citing ids and staying inside the documents |
| B. Same small model on Colab | GPU speed | session timeouts, re-download each session, hard to combine with the Streamlit demo |
| C. Free API — Gemini Flash | strong model, no download | free quota reported to change without notice (forum: 250 → 20 requests/day); Google staff call it "best effort" |
| **D. Free API — Groq** | strong open-weight models, no download, stable published limits | needs an account + key; daily token cap |

## Decision

**Groq, model `openai/gpt-oss-120b`** (OpenAI, open weights). Chosen by J, 2026-10-28.

First choice was `llama-3.3-70b-versatile`, because Groq's models page listed it as a production model.
The API answered `404 model_not_found` for this key: the docs page was stale. The key's own
`GET /openai/v1/models` (2026-10-28) offered for chat: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`qwen/qwen3.8-27b`, `allam-2-7b`. J picked the strongest, `openai/gpt-oss-120b`; fallback `openai/gpt-oss-20b`.

Settings for every call: `temperature = 0`, `reasoning_effort = "low"` (gpt-oss reasons before answering;
reasoning tokens count against the limits), fixed `max_tokens`, prompt from `rag.build_prompt`.

Smoke test (2026-10-28, toy 2-document ETF prompt): HTTP 200 in 0.43 s, answer correct and cited `[d42]`;
192 prompt + 46 completion tokens (11 of them reasoning).

## Consequences

- **Key handling:** `GROQ_API_KEY` lives only in `.env` (gitignored), read from the environment by
  `rag.generate`. Never in code, configs, logs or chat.
- **Rate limits (this key, from the smoke test's response headers, 2026-10-28):**
  `x-ratelimit-limit-requests: 1000` and `x-ratelimit-limit-tokens: 8000` (tokens per minute).
  The docs also list ~200K tokens/day for this model on the Developer plan
  (<https://console.groq.com/docs/rate-limits>); the daily token cap for this account is not in the headers —
  **TODO (J): confirm it on the account's limits page.** At ~2K tokens per call (k = 5) the per-minute cap
  allows ~4 calls/min; `generate` waits and retries on HTTP 429. 190 calls ≈ 380K tokens.
- **Reproducibility:** hosted models can change or be retired. Record the model id and date with every
  run, and save every raw response next to its run's `metrics.json` so scoring never needs a re-call.
- **Fallback:** if Groq becomes unusable, option A with a local open-weight model; that needs a new
  decision page first.

## Who / date

J, 2026-10-28. Reviewer: H (via the I5 pull request).

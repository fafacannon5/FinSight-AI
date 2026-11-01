---
title: "Decision: final configs for the locked FiQA test evaluation"
type: decision
tags: [final, e1, e2, e3, m12]
created: 2026-11-01
updated: 2026-11-01
status: active
---

# Decision: final configs (run once on FiQA test)

## Context

Hard rule 1: the FiQA `test` split is used once, with configs fixed beforehand. Everything below was tuned
on `dev` only. The research question is an accuracy–efficiency trade-off, so we report an accurate
reference config and an efficient config side by side.

## Options

| Option | For | Against |
|---|---|---|
| A only (384-d exact) | best accuracy, simplest | no efficiency side to the trade-off |
| B only (128-d + IVF) | fastest, smallest | no reference to measure the loss against |
| **A + B, generation with A** | shows the trade-off; LLM calls and manual scoring stay at one config | two retrieval runs (cheap, no LLM) |

## Decision (J and H, 2026-11-01)

| | Config A — accurate | Config B — efficient |
|---|---|---|
| Embedding | MiniLM-L6-v2, 384-d | same + PCA 128-d (fit on corpus only, re-normalized) |
| Index | exact `torch_knn` | `faiss_ivf` nlist 240, nprobe 16, seed 42 |
| Dev Recall@10 / nDCG@10 | 0.4665 / 0.3881 | 0.4178 / 0.3388 |
| Dev p95 latency / memory | 2.700 ms / 84.43 MB | 0.089 ms / 28.14 MB (same session, J's machine) |
| Refusal τ (top-1 cosine) | **0.60** | **0.66** |
| Generation | **yes**: top-k = **5**, `openai/gpt-oss-120b`, prompt = `rag.build_prompt` | no |

Evidence: [E1 summary](../experiments/e1_summary.md) · [E2 run 004/005](../experiments/e2_run_004.md) ·
[E3 run 001](../experiments/e3_run_001.md) (k) · [E3 run 002](../experiments/e3_run_002.md) (τ A) ·
[E3 run 003](../experiments/e3_run_003.md) (τ B).

## Test protocol

1. Retrieval, both configs: `python scripts/run_experiment.py <cfg> --final` on all 648 test queries
   (configs `experiments/configs/final_a.yaml`, `final_b.yaml` — to write, copying run_e2_005 / run_e2_004
   with `split: test`).
2. Generation, config A only: a fixed sample of test questions (seed 42; size to agree, proposal: ~100),
   k = 5, scored with the [rubric](../experiments/e3_rubric.md); refusals counted separately.
3. Refusal: apply τ = 0.60 (A) / 0.66 (B) unchanged. A test unanswerable set needs the same
   `make_unanswerable` + manual check as on dev — **open: decide whether to build it.**
4. No re-tuning after any test number is seen; problems go to the report as limitations.

## Consequences

- **Proposed, to confirm:** the prompt example `[d42]` stays as is: changing the prompt now would invalidate the dev k choice.
  Citation-format quirks (`【id】`, `[d<id>]`) are normalized at scoring time.
- Hosted LLM: record model id + date per answer (already done by `e3_generate.py`).
- The Streamlit demo uses config A with k = 5 and τ = 0.60.

## Who / date

J and H, 2026-11-01. Written by Claude from the dev results; reviewer: H (via the I5 pull request).

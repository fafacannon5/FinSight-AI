---
title: "Final test evaluation — configs A + B (retrieval) and A (generation)"
type: experiment-run
tags: [final, test, retrieval, generation, refusal]
created: 2026-11-11
updated: 2026-11-11
status: active
owner: shared
---

# Final test evaluation (FiQA test, run once)

Run once on 2026-11-11 at J's explicit request ("跑测试集"), with the configs fixed in
[final_configs](../decisions/final_configs.md). Nothing was tuned afterwards.

## Retrieval (`final_a`, `final_b`; 648 test queries, full 57,638-doc corpus, same session)

| Config | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | p95 (ms) | Memory (MB) |
|---|---|---|---|---|---|---|
| A — 384-d exact TorchKNN | 0.3671 | **0.4413** | 0.3687 | 0.4451 | 11.630 | 84.43 |
| B — 128-d PCA + IVF (nlist 240, nprobe 16) | 0.3258 | **0.3878** | 0.3275 | 0.4060 | 0.080 | 28.14 |
| B − A (test) | −4.13 pt | **−5.35 pt** | −4.12 pt | −3.91 pt | | 3.0× smaller |
| B − A (dev, runs 005 / 004) | | −4.88 pt | −4.93 pt | −5.76 pt | 30× faster | 3.0× smaller |

- Test is a little below dev for both configs (A 0.4665 → 0.4413, B 0.4178 → 0.3878), as expected after
  choosing settings on dev. The accuracy gap A − B carries over (−5.35 vs −4.88 pt).
- **Limitation — latency of A on test is not reliable.** Machine load average was ≈ 9.8 during the run;
  multithreaded TorchKNN slowed down (p95 11.6 ms vs 2.7 ms on dev), while B's tiny FAISS search did not
  (0.080 vs 0.089 ms). The apparent 145× speed-up is inflated; the dev same-session 30× is the speed result.
  Not re-run, by protocol.

## Generation (`final_gen_a`; config A, k = 5, τ = 0.60, 100 test questions, seed 42)

| | Count |
|---|---|
| Questions sent to the LLM | 100 (all, see the [2026-11-02 log decision](../log.md)) |
| Refused by τ (top-1 < 0.60) → system answer rate | 16 → **0.84** (dev: 0.760 on 450 answerable, [e3_run_002](e3_run_002.md)) |
| A labelled relevant doc in the top 5 | 62 / 100 (58 / 84 answered) |
| LLM itself replied with `REFUSAL_TEXT` | 23 (9 of them also τ-refused; 3 had a relevant doc in the top 5) |
| System gives a real answer (not τ-refused, not LLM-refused) | 70 |

- Test answer rate 0.84 > dev 0.76: on the full corpus nothing was removed (dev used the corpus minus 116 docs
  for the unanswerable set), and only 100 questions were sampled. No test unanswerable set (protocol), so
  correct refusal is reported from dev only.

Tokens 104,914 (budget 150K), `openai/gpt-oss-120b`, reasoning low, temperature 0, no errors.

**Open:** manual 1–3 correctness / groundedness scoring of `final_gen_a/scoring.csv` by J + H
([rubric](e3_rubric.md)), then `notebooks/05_final_test_results.ipynb`.

# AI usage log (CS 541 disclosure)

CS 541 allows AI tools for brainstorming, debugging and understanding concepts and requires disclosure.
Every agent-assisted change gets a row. The final report's "Use of AI Tools" section is written from this table.

| Date | Area / file | Tool | What the AI did (scaffold / hint / full code / debug / review / writing) | Reviewed by |
|---|---|---|---|---|
| 2026-10-11 | Project scope, proposal draft | Claude, ChatGPT | Brainstorming and drafting | J, H |
| 2026-10-11 | Repo structure, agent docs, stubs, unit tests | Claude | Scaffold (stubs raise NotImplementedError; tests define expected behavior) | J |
| 2026-10-25 | `src/finsight/metrics.py` `recall_at_k`, `mrr_at_k`, `ndcg_at_k` | Claude | Full code (Level-1 scaffold, then blanks filled by Claude at member's request) + line-by-line explanation of Recall/MRR/nDCG | J |
| 2026-10-25 | `src/finsight/data.py` `load_beir`, `check_ids`; `tests/test_data.py` id-check tests; Recall@10 == 0 guard in `scripts/run_experiment.py`; `wiki/dataset/*.md` notes | Claude | Full code (written after inspecting the real HF datasets with J's OK to download) + line-by-line explanation | J |
| 2026-10-25 | Correction to the 2026-10-11 scaffold row | Claude | That scaffold was not stubs only: `embed.py`, `timing.py` and `scripts/run_experiment.py` were generated as full code | J |
| 2026-10-25 | `src/finsight/embed.py` id-list cache check (`ids_path_for`, `ids=` in `load_or_encode`); `tests/test_embed.py`; `run_experiment.py` passes ids | Claude | Full code at J's request + explanation | J |
| 2026-10-25 | `src/finsight/pca.py` `PCAReducer.fit` / `transform` | Claude | Full code at J's request, after a concept walkthrough + 3-question quiz on PCA; line-by-line explanation | J |

# Workflow: Review

Parent: [AGENTS.md](../../AGENTS.md)

**Trigger:** "review my code" / "帮我看看代码" / before merging a branch into `main`.

1. `git diff main...HEAD` (or the files named).
2. Run `pytest -q`.
3. Check, in this order:
   - **Correctness:** shapes, normalization (Hard rule 2), off-by-one in top-k, ids vs. indices mix-ups.
   - **Leakage:** any use of the test split or test qrels outside *Final evaluation*.
   - **Fairness of comparisons:** same embeddings, same queries, same k across methods.
   - **Timing hygiene:** warm-up excluded, single-query batches, same machine.
   - **Explainability:** would the owner be able to explain this in the video?
4. Report findings ranked by severity, each with file:line and a one-line fix suggestion.
   Do not rewrite the member's code unless asked.

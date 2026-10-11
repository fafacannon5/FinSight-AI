# Deliverable code standards

Parent: [AGENTS.md](../../AGENTS.md)

Code in `src/`, `tests/`, `notebooks/`, `app/`, `scripts/` must:

- Be explainable line by line by the member who owns it (see owners in AGENTS.md).
- Have docstrings with argument shapes, e.g. `doc_emb: (N, d) float32, L2-normalized`.
- Keep library logic in `src/finsight/`; notebooks only load results, plot, and explain.
- Use `RANDOM_SEED = 42`; run top-to-bottom after `pip install -r requirements.txt`.
- Read paths from config, never hard-code a personal path (`C:\Users\...`, `/Users/...`).
- Notebook markdown cells: motivation → what we ran → what we observed → decision.
  Write them in the member's own words; agents may draft but the member edits.
- Not reference internal run ids or `experiments/` paths in report-facing text
  (cleanliness, not concealment — AI use is still disclosed per [ai_disclosure.md](ai_disclosure.md)).

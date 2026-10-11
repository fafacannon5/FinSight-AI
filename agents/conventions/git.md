# Git conventions

Parent: [AGENTS.md](../../AGENTS.md)

- **Agents never commit or push on their own.** They may run `git status` / `git diff` and suggest messages;
  a member commits. Never push to `main`; merge only through a pull request.
- `main` is always green (`pytest -q` passes). Work on branches: `h/e1-retrieval`, `j/e2-pca`,
  `j/e3-refusal`, `shared/data-embed`. Merge via pull request after [review.md](../workflow/review.md); the other member reviews and merges.
  Merge at every integration point, not only at the end — step-by-step in [merge.md](../workflow/merge.md).
- Commit prefixes: `feat(<scope>)`, `fix(<scope>)`, `test(<scope>)`, `guide(<scope>)`,
  `experiment(<exp>)`, `docs(wiki)`, `wip:`.
- When a member commits agent-assisted work, the suggested message includes an `Assisted-by:` trailer ([ai_disclosure.md](ai_disclosure.md)).
- Never commit `data/`, `*.npy`, `*.pt`, `.env`, `experiments/results/` (they are in `.gitignore`).

# Workflow: Merge (branch → main via pull request)

Parent: [AGENTS.md](../../AGENTS.md) · Also read: [git.md](../conventions/git.md)

**Trigger:** "merge" / "合并" / "提 PR" / "pull request" / "同步 main" / "sync with main", and at the end of each
integration point below.

The agent **guides and checks**; the member runs every commit / push / merge (Hard rule 9).
Give commands one block at a time and wait for the member to paste the output.

---

## Integration points (merge early, not only at the end)

| # | When | Branch → main | Why now |
|---|---|---|---|
| I1 | After M1–M3 (done together) | `shared/foundation` | metrics, data, embeddings are used by both members |
| I2 | After M4 (TorchKNN passes tests) | `h/e1-retrieval` | J's E2 needs TorchKNN to run |
| I3 | After M7 (PCA passes tests) | `j/e2-pca` | E2 runs on main with H's retrieval |
| I4 | After M6 + E1 runs | `h/e1-retrieval` | E1 results + FAISS back ends |
| I5 | After M8–M9 | `j/e3-refusal` | unanswerable set + RAG layer |
| I6 | After E2/E3 runs + demo | `j/...` | results, Streamlit demo |
| I7 | Week 13: final | — | both PRs merged → final test evaluation on `main` → tag `v1.0` |

Before each integration point: both members sync their branch with `main` (section B).

---

## A. Start a new branch (from an up-to-date main)

```bash
git checkout main
git pull origin main
git checkout -b j/e2-pca          # J: j/...   H: h/...   together: shared/...
```

## B. Sync your branch with the latest main (do this at least weekly and before every PR)

```bash
git checkout j/e2-pca
git fetch origin
git merge origin/main             # brings the teammate's merged work into your branch
pytest -q                         # must pass before you continue
```

Conflict? → section D.

## C. Open a pull request

1. Agent runs the checklist (section E) and reports it.
2. Member commits and pushes (agent suggests the message, incl. `Assisted-by:` trailer when relevant):
   ```bash
   git add -A
   git status                       # check nothing from data/, *.npy, .env is staged
   git commit -m "feat(pca): PCAReducer with re-normalization" -m "Assisted-by: Cursor"
   git push -u origin j/e2-pca
   ```
3. On GitHub: **Compare & pull request** → base `main` ← compare `j/e2-pca`.
   Title = what changed; description = the PR template below. **Reviewer = the other member.**
4. The reviewer reads the diff (an agent can help via [review.md](review.md)), pulls the branch,
   runs `pytest -q`, and either approves or comments.
5. The reviewer clicks **Merge pull request** (prefer *Squash and merge* for a clean history),
   then **Delete branch**.
6. Both members: `git checkout main && git pull origin main`.

PR description template:

```markdown
## What
- ...
## How tested
- pytest -q: N passed
- (if experiment) run ids + one-line result
## Notes for reviewer
- files to look at first, open questions
## AI assistance
- tool + what it did (also added to AI_USAGE.md)
```

## D. Resolving conflicts

1. `git status` lists conflicted files. The agent explains each conflict in plain words:
   what *your* side changed, what *main* changed, what the merged result should be.
2. Files where both sides should be KEPT (append-only): `wiki/log.md`, `AI_USAGE.md`,
   `wiki/index.md` rows → keep both entries, newest first.
3. `wiki/overview.md`: keep both updates; status line must describe the combined state.
4. Code (`src/finsight/*.py`): never pick a side blindly. If both changed the same function, stop and ask
   the owner of that module (AGENTS.md owner table) — message the teammate if needed.
5. `requirements.txt`: keep the union of packages.
6. After editing: `git add <file>` for each, `pytest -q`, then `git commit` (member runs it).
7. Stuck or unsure: `git merge --abort` returns to the state before the merge. Nothing is lost.

## E. Pre-PR checklist (agent runs and reports)

- [ ] `pytest -q` passes on the branch after syncing with main
- [ ] no staged files from `data/`, `*.npy`, `*.pt`, `.env`, `experiments/results/`
- [ ] no use of the FiQA test split outside final evaluation
- [ ] new public functions have tests and docstrings with shapes
- [ ] `wiki/log.md` and relevant `wiki/experiments/*_summary.md` updated
- [ ] `AI_USAGE.md` row added for AI-assisted work
- [ ] the owner can explain every changed line

## F. Final release (I7, Week 13)

1. Both branches merged; `main` passes `pytest -q`.
2. Record chosen configs in `wiki/decisions/final_configs.md` (both members agree).
3. On `main`, run the final test-set evaluation once (experiment_loop.md → *Final evaluation*).
4. Commit results/figures via a last PR, then tag:
   ```bash
   git checkout main && git pull
   git tag -a v1.0 -m "CS 541 final submission"
   git push origin v1.0
   ```
5. Submit the tag's code (GitHub link or zip of that tag).

## One-time GitHub setup (repo owner, once)

- Add the teammate: Settings → Collaborators → Add people.
- Protect `main` if your plan allows it (Settings → Branches / Rules: require a pull request before merging).
  If not available on your plan, follow the rule by agreement: nobody pushes to `main` directly.

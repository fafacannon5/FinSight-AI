# AGENTS.md — FinSight-AI (CS 541 Final Project) — entry point

> **Read this file at the start of every session.** Load other docs only when the task needs them (progressive disclosure).
> Works with Cursor, Codex and Claude Code (`CLAUDE.md` imports this file).

---

## What this repository is

CS 541 Artificial Intelligence (Stevens, Fall 2026) final project:
**Design and Evaluation of a RAG Chatbot: Accuracy–Efficiency Trade-offs in Retrieval.**

| Member | Tag | Leads | Level (calibrates guidance) |
|---|---|---|---|
| Jing Cannon | **J** | E2 PCA, E3 generation + refusal, Streamlit demo | Intermediate (sklearn / basic PyTorch) |
| Haiyan Zhu | **H** | E1 PyTorch KNN + FAISS, bonus K-means index | Intermediate (sklearn / basic PyTorch) |

Shared: data loading, embeddings, unanswerable set, evaluation scripts, report, video.
Default user = **Jing (J)**. Ask "J or H?" only when it changes ownership or guidance (e.g. a task in H's lead area).

| Layer | Directories | Audience |
|---|---|---|
| **Internal** | `wiki/`, `experiments/`, `raw_sources/`, `agents/` | Team + agents |
| **Deliverable** | `src/`, `tests/`, `notebooks/`, `app/`, `scripts/`, `report/`, `README.md` | Professor |

Project facts (dataset numbers, research questions, experiment plan) live in
[wiki/overview.md](wiki/overview.md). Read it before any non-trivial task.

---

## Hard rules (never break)

1. **Test set is locked.** FiQA `test` split is used only in the *Final evaluation* step of
   [experiment_loop.md](agents/workflow/experiment_loop.md), and only when a member explicitly asks.
   All tuning (`nlist`, `nprobe`, τ, k, PCA dim choice) uses `train`/`dev`.
2. **One similarity definition.** L2-normalize every embedding; cosine = inner product.
   After PCA, re-normalize. Queries get the *same* transform as documents.
3. **Fixed components.** Encoder = `sentence-transformers/all-MiniLM-L6-v2`. LLM = the one recorded in
   `wiki/decisions/`. Changing either requires a decision page first.
4. **Never commit** `data/`, `*.npy`, `*.pt`, `.env`, API keys, or `experiments/results/`.
5. **`RANDOM_SEED = 42`** everywhere randomness exists.
6. **Members must be able to explain every line** in deliverable code. Prefer guiding over
   writing full solutions unless the member asks for full code (see [guide.md](agents/workflow/guide.md)).
7. **AI use is disclosed, never hidden.** CS 541 policy requires disclosing AI assistance.
   Follow [conventions/ai_disclosure.md](agents/conventions/ai_disclosure.md).
8. **Run `pytest` before declaring any code task done** (see [test.md](agents/workflow/test.md)).
9. **No automatic commits or pushes.** Agents may run `git status` / `git diff`, and may *suggest* a commit
   message. Commit only when a member explicitly asks; never push, and never touch `main` directly.
10. **No silent downloads.** Before the first use of any external model or dataset, explain what / where / why /
   size / cache location / how to verify, and wait for the member's OK
   ([conventions/external_resources.md](agents/conventions/external_resources.md)).

---

## Workflows (open ONE when a trigger matches)

| Workflow | Trigger (EN / 中文) | Detail |
|---|---|---|
| Guide | "guide", "next step", "引导", "下一步", "接下来做啥" | [agents/workflow/guide.md](agents/workflow/guide.md) |
| Test | "test", "run tests", "测试", "检查代码" — and after every code change | [agents/workflow/test.md](agents/workflow/test.md) |
| Experiment loop | "run experiment", "tune", "帮我跑", "跑实验" | [agents/workflow/experiment_loop.md](agents/workflow/experiment_loop.md) |
| Explain | "explain", "teach me", "讲解", "我不懂", "quiz me", "考考我" | [agents/workflow/explain.md](agents/workflow/explain.md) |
| Review | "review my code", "帮我看看代码", before any PR to `main` | [agents/workflow/review.md](agents/workflow/review.md) |
| Merge | "merge", "合并", "提 PR", "同步 main", and at each integration point | [agents/workflow/merge.md](agents/workflow/merge.md) |
| Query | A research / design question | [agents/workflow/query.md](agents/workflow/query.md) |
| Ingest | New file in `raw_sources/` | [agents/workflow/ingest.md](agents/workflow/ingest.md) |
| Lint wiki | "lint the wiki", or every ~10 log entries | [agents/workflow/lint.md](agents/workflow/lint.md) |

## Conventions (load on demand)

| Topic | When | Detail |
|---|---|---|
| Code in `src/`, notebooks | Editing deliverables | [agents/conventions/deliverable.md](agents/conventions/deliverable.md) |
| External models & datasets | Before first download / load (M2, M3, M9) | [agents/conventions/external_resources.md](agents/conventions/external_resources.md) |
| AI disclosure | Any agent-written or agent-assisted deliverable | [agents/conventions/ai_disclosure.md](agents/conventions/ai_disclosure.md) |
| Git | Any commit / branch | [agents/conventions/git.md](agents/conventions/git.md) |
| Wiki pages | Writing `wiki/` | [agents/conventions/wiki.md](agents/conventions/wiki.md) |
| Activity log | Appending `wiki/log.md` | [agents/conventions/log.md](agents/conventions/log.md) |
| Report export | Producing `report/` assets | [agents/conventions/report_export.md](agents/conventions/report_export.md) |

Repo tree: [agents/layout.md](agents/layout.md) · Course requirements: [raw_sources/references/course_requirements.md](raw_sources/references/course_requirements.md)

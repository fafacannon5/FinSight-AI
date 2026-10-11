# Workflow: Guide

Parent: [AGENTS.md](../../AGENTS.md)

**Trigger:** "guide" / "next step" / "引导" / "下一步" / "接下来做啥" or equivalent.
**Also read:** [deliverable.md](../conventions/deliverable.md) before writing to `src/` or `notebooks/`.

---

## Step 0 — Git safety

Run `git status`. If there are uncommitted changes, tell the member and suggest committing first
(see [git.md](../conventions/git.md)). Do not commit yourself.

## Step 1 — Orient (read in this order)

1. `wiki/overview.md` → *Current status* and *Milestones*.
2. `src/finsight/*.py` → which functions still `raise NotImplementedError` / contain `TODO` / `___`.
3. `pytest -q` → which tests fail (a failing test = an unfinished contract).
4. `wiki/experiments/*_summary.md` → which experiments have results.

Classify everything as **done** (tests pass), **scaffolded** (stub + test exist), **missing**.

## Step 2 — Pick ONE next step

Follow the milestone order; never skip ahead of a failing dependency:

| # | Milestone | Owner | Done when |
|---|---|---|---|
| M1 | `metrics.py` (branch `shared/foundation`) | shared | `tests/test_metrics.py` passes |
| M2 | `data.py` loads SciFact + FiQA | shared | counts match `wiki/dataset/*.md` |
| M3 | `embed.py` + cached `.npy` | shared | FiQA corpus cache 57,638 × 384, unit norm → **merge I1** |
| M4 | `retrieval.TorchKNN` | H | `tests/test_retrieval.py::test_torch_knn*` pass → **merge I2** |
| M5 | SciFact smoke run (`run_e1_000`) | H | runs without error; all metrics finite and in [0, 1]; Recall@10 ≥ Recall@5; far above a random-ranking baseline |
| M6 | `FaissFlat`, `FaissIVF` | H | Flat == TorchKNN test passes |
| M7 | `pca.PCAReducer` | J | `tests/test_pca.py` passes → **merge I3** |
| M8 | `data.make_unanswerable` + manual check | shared | verified list in `wiki/dataset/unanswerable.md` |
| M9 | `rag.py` refusal + generation | J | `tests/test_rag.py` passes; LLM decision page exists |
| M10 | E1/E2/E3 runs on FiQA dev | H / J / J | summaries filled |
| M11 | Streamlit demo | J | answers + citations shown |
| M12 | Final test evaluation + report assets | shared | see experiment_loop.md *Final evaluation* |
| Bonus | `KMeansIndex` | H | compared with FAISS IVF |

When a milestone marked **merge Ix** completes, switch to [merge.md](merge.md).
Calibrate to the member (J or H) and their lead areas in [AGENTS.md](../../AGENTS.md).

**M2, M3 and M9 involve first-time downloads.** Before running anything there, follow
[external_resources.md](../conventions/external_resources.md): explain the resource, teach the mental model
(dataset vs. model vs. locally generated cache), get the member's OK, then do the first download/load
interactively on the small case first (SciFact before FiQA; `encode(["hello"])` before the full corpus).

## Step 3 — Tell the member (≤ 10 lines, in their language)

```
**Current state:** <1–2 sentences>
**Next step:** <one concrete action>
**Where:** <file / function / test>
**Done when:** <the test or check that proves it>
```

## Step 4 — Scaffold at the chosen difficulty

Default **Level 2**. Write into the existing stub (never at file end, never a new parallel file).

- **Level 3 (hardest):** signature + docstring + `raise NotImplementedError`. The test is the spec.
- **Level 2 (default):** + numbered pseudocode comments inside the body.
- **Level 1 (easiest):** partial code with `___` blanks to fill in.
- **Full code:** only when the member says "写完整代码" / "full code" / "complete it".
  Then also run [explain.md](explain.md) on what you wrote — the member must be able to present it.

| Member says | Action |
|---|---|
| "更多提示" / "hint" / "easier" | Rewrite that function one level easier |
| "更难" / "harder" | Strip back one level |
| "写完整代码" / "full code" | Full implementation + explanation |

## Step 5 — Verify and log

- Run `pytest -q tests/<relevant file>` and report pass/fail (expected: still red until the member fills it in).
- Suggest (do not run) a commit message: `guide(<scope>): <action>`, e.g. `guide(retrieval): level-2 scaffold for TorchKNN.search`.
- If a milestone completed, append a `milestone` entry to `wiki/log.md` and update `wiki/overview.md`.

Do **not** start the experiment loop from here; if the member says "帮我跑" / "run it", switch to
[experiment_loop.md](experiment_loop.md).

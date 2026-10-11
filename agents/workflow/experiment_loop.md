# Workflow: Experiment loop

Parent: [AGENTS.md](../../AGENTS.md)

**Trigger:** "run experiment" / "tune" / "帮我跑" / "跑实验" for E1, E2, E3 or the bonus.
**Precondition:** `pytest -q` passes for every module the run uses. If not → [test.md](test.md) first.

## Git safety

`git status` → if dirty, tell the member and suggest committing first. Do not commit yourself.
After each FILE step, suggest (do not run) a commit message:
`experiment(<e1|e2|e3|bonus>): run_NNN — <one-line finding>`. The member decides when to commit.

## Loop

```
ORIENT → HYPOTHESIZE → CONFIGURE → EXECUTE → OBSERVE → FILE → NEXT
```

1. **Orient:** read `wiki/experiments/<exp>_summary.md` (create from the README template if missing).
2. **Hypothesize:** one falsifiable sentence, e.g. *"IVF with nprobe=16 keeps Recall@10 within 2 points
   of exact search while cutting p95 latency by ≥ 3×."*
3. **Configure:** copy a template in `experiments/configs/` → `run_<exp>_NNN.yaml`. Change one factor.
4. **Execute:** `python scripts/run_experiment.py experiments/configs/run_<exp>_NNN.yaml`
   → writes `experiments/results/run_<exp>_NNN/metrics.json`.
5. **Observe:** read metrics.json. Sanity-check: Recall@10 ≥ Recall@5; latency > 0; exact methods agree.
6. **File:** write `wiki/experiments/<exp>_run_NNN.md` (hypothesis, config diff, metrics table, verdict)
   and add a row to `<exp>_summary.md`. Append `experiment` entry to `wiki/log.md`.
7. **Next:** new hypothesis → step 2.

**Stop when:** 3 cycles without meaningful change on dev, the planned grid is covered, or the member says stop.

## Planned grids (from the proposal — don't expand without a decision page)

| Exp | Factor | Values | Fixed |
|---|---|---|---|
| E1 | method | torch_knn, faiss_flat, faiss_ivf (`nlist` ≈ √N ≈ 240, `nprobe` ∈ {1,4,16,64}) | dim 384, k=10 |
| E1-scale | corpus size | 10k, 25k, 57,638 | torch_knn + faiss_ivf |
| E2 | pca_dim | none(384), 128, 64, 32 | torch_knn, k=10 |
| E3 | top_k, τ | k ∈ {1,3,5,10}; τ swept on dev | best E1/E2 config, fixed LLM + prompt |

## Final evaluation (test set) — only on explicit request

Trigger: a member says "final evaluation" / "跑测试集" **and** the chosen configs are recorded in
`wiki/decisions/final_configs.md`. Then run each chosen config once with `split: test` and
`python scripts/run_experiment.py <cfg> --final`. Never re-tune afterwards; if something is wrong,
log it as a limitation.

## Transition to deliverables

Distill each experiment into `notebooks/0X_<exp>.ipynb`: load the result JSONs, make the figures and
tables for `report/`, and explain the reasoning in markdown cells written in the member's own words.

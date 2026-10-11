# Workflow: Test

Parent: [AGENTS.md](../../AGENTS.md)

**Trigger:** "test" / "run tests" / "测试" / "检查代码" — and **automatically after every code change**.

## Step 1 — Run

```bash
pytest -q                       # everything
pytest -q tests/test_metrics.py # one file
pytest -q -k torch_knn          # by name
```

Tests use small synthetic arrays only — no downloads, no GPU, < 30 s total.
FAISS tests skip automatically if `faiss` is not installed.

## Step 2 — Report (short)

```
✅ passed: N   ❌ failed: M   ⏭ skipped: K
Failing: <test name> — <one-line reason in plain words>
```

## Step 3 — On failure

1. Read the assertion message and the function under test.
2. Explain the bug in plain words first (what the code does vs. what the test expects).
3. Guidance mode (default): point to the line and give a hint. Fix it yourself only if the member asks
   ("帮我修" / "fix it"), or the bug is in agent-written code.
4. **Never edit a test to make it pass** unless the member agrees the test itself is wrong; then record why
   in `wiki/log.md`.

## Step 4 — When to add tests

Add a test whenever you add a public function in `src/finsight/`. Every test must:
- use hand-checkable numbers or a brute-force reference (e.g. numpy argsort vs. our KNN);
- run in < 5 s; use `RANDOM_SEED = 42`.

## Correctness checks that matter for this project

| Check | Why |
|---|---|
| TorchKNN top-k == numpy brute force | our exact search is correct |
| FAISS Flat top-k == TorchKNN | fair E1 comparison |
| FAISS IVF with `nprobe = nlist` == exact | IVF configured correctly |
| PCA output rows have unit norm | cosine still = inner product (Hard rule 2) |
| PCA fit uses corpus only; transform reuses stored mean | no query leakage |
| Metrics match hand-computed values | numbers in the report are right |
| Answer rate non-increasing as τ rises | refusal curve is sane |

## No target numbers

Never edit code to move a metric toward an "expected" value. A surprising number is investigated with
tests and sanity checks (finite, in range, Recall@10 ≥ Recall@5, exact methods agree), then reported as is.

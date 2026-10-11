# Wiki conventions

Parent: [AGENTS.md](../../AGENTS.md)

Every page starts with frontmatter:

```yaml
---
title: "Page title"
type: overview|dataset|experiment|experiment-run|literature|decision
tags: [e1, faiss]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft|active|archived
---
```

Names: `dataset/fiqa.md`, `experiments/e1_summary.md`, `experiments/e1_run_003.md`,
`literature/johnson2019_faiss.md`, `decisions/llm_choice.md`.
Link with relative paths: `[E1 summary](../experiments/e1_summary.md)`.
Numbers in the wiki must come from a results file or a cited source — never from memory.

# Experiments

Raw outputs: `experiments/results/<run_id>/metrics.json` (gitignored). This folder holds the narrative.

## Summary page template (`<exp>_summary.md`)

```markdown
---
title: "E1 — Retrieval method: summary"
type: experiment
tags: [e1]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: active
owner: H
---

## Best so far (dev)
| Config | Recall@10 | nDCG@10 | p95 latency (ms) | Memory (MB) | Run |

## Run history
| Run | Hypothesis | Recall@10 | nDCG@10 | p95 ms | Verdict |

## Findings so far
## Next hypothesis
```

## Run page template (`<exp>_run_NNN.md`)

Hypothesis · config diff vs. previous run · metrics table · verdict (confirmed / rejected / inconclusive) · next step.

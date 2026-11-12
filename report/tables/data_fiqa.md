| Split | Queries | qrels pairs | Relevant docs / query (mean · median · max) | Used for |
|---|---|---|---|---|
| train | 5,500 | 14,166 | 2.58 · 2 · 23 | not used (no model is trained; PCA / IVF are fitted on the corpus) |
| dev | 500 | 1,238 | 2.48 · 2 · 20 | all tuning (E1–E3) |
| test | 648 | 1,706 | 2.63 · 2 · 15 | final evaluation, run once |
| SciFact test | 300 | 339 | — | smoke test only (5,183 docs) |

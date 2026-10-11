# Notebooks (deliverable)

Notebooks load `experiments/results/*/metrics.json`, make figures/tables for `report/`, and explain the
reasoning in markdown written by the owner. Library code stays in `src/finsight/`.

| Notebook | Owner | Content |
|---|---|---|
| `01_data_and_embeddings.ipynb` | shared | dataset stats, embedding cache check |
| `02_e1_retrieval_methods.ipynb` | H | recall vs. latency (Pareto plot), memory table, scaling curve |
| `03_e2_pca.ipynb` | J | recall/latency/memory vs. dimension, explained variance |
| `04_e3_reliability.ipynb` | J | refusal curve, answer-quality rubric results |
| `05_final_test_results.ipynb` | shared | test-set results for the chosen configs |

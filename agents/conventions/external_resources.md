# External models and datasets

Parent: [AGENTS.md](../../AGENTS.md)

**Do not silently download a new model or dataset.** Before the FIRST use of any external resource,
stop and explain it to the member (in their language), then wait for "OK / 好 / go" before running:

1. **What** is downloaded — model, dataset, or a locally generated cache (not a download at all).
2. **Where** it comes from (Hugging Face repo id, API provider).
3. **Why** the project needs it, and which milestone / experiment uses it.
4. **Size** — approximate download and disk space when known; say "unknown — will check" otherwise.
5. **Once or every run** — downloaded once and cached, or recomputed.
6. **Where** it is cached locally.
7. **How to verify** the download / load worked (the exact check to run and the expected output).

## Resources in this project

| Resource | Kind | Source | Used in | Cached at | Verify |
|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (~90 MB) | Hugging Face | M3, all experiments | `~/.cache/huggingface/hub/` (once) | `encode(["hello"]).shape == (1, 384)` and row norm ≈ 1 |
| SciFact (BEIR) corpus / queries / qrels | Dataset | Hugging Face `BeIR/scifact`, `BeIR/scifact-qrels` | M2, M5 smoke test | `~/.cache/huggingface/datasets/` (once) | 5,183 docs; 300 test queries |
| FiQA-2018 (BEIR) corpus / queries / qrels | Dataset | Hugging Face `BeIR/fiqa`, `BeIR/fiqa-qrels` | M2, E1–E3 | `~/.cache/huggingface/datasets/` (once) | 57,638 docs; 5,500 / 500 / 648 queries |
| `data/*_minilm.npy` | **Generated locally**, not downloaded | our `embed.py` (MiniLM on our machine) | M3 onward | `data/` (gitignored) | FiQA corpus shape (57,638, 384), unit-norm rows |
| Generation LLM | Model (size depends on choice — a ~1.5B model is several GB) or API | TBD → `wiki/decisions/llm_choice.md` | M9, E3, demo | HF cache, or none for an API | one test prompt returns text; API key only in `.env` |

Record exact repo ids, split names and sizes observed on first download in `wiki/dataset/<name>.md`
(or the LLM decision page). Adding any resource not in this table needs a decision page first
(Hard rule 3).

## Mental model to teach at M2 / M3

```
Hugging Face ── Datasets: SciFact, FiQA          (downloaded once, cached)
             └─ Model:    all-MiniLM-L6-v2        (downloaded once, cached)
Our machine  ── data/*.npy embeddings            (COMPUTED here: text → MiniLM → 384-d vectors → .npy)
```

The `.npy` files are never downloaded and never pushed; anyone can regenerate them from the code.

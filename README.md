# FinSight-AI — Design and Evaluation of a RAG Chatbot: Accuracy–Efficiency Trade-offs in Retrieval

CS 541 Artificial Intelligence · Stevens Institute of Technology · Fall 2026
**Team:** Jing Cannon · Haiyan Zhu

We build a retrieval-augmented chatbot for financial question answering (FiQA) and measure how retrieval
choices trade accuracy against latency and memory:

- **RQ1** exact (our PyTorch brute-force KNN, FAISS Flat) vs. approximate (FAISS IVF) search
- **RQ2** PCA reduction of 384-d embeddings to 128 / 64 / 32 dimensions
- **RQ3** top-k and a similarity threshold for refusing unanswerable questions

Report: [report/final_report.tex](report/final_report.tex) · Slides: [report/FinSight-AI_slides.pptx](report/FinSight-AI_slides.pptx)

## Results at a glance

All settings were tuned on the FiQA dev split (500 questions); the test split (648 questions) was evaluated
once with two fixed configurations.

| Config (FiQA test, full 57,638-post corpus) | Recall@10 | nDCG@10 | p95 latency (dev, same session) | Index memory |
|---|---|---|---|---|
| **A — accurate:** 384-d, exact search (our PyTorch KNN) | 0.4413 | 0.3687 | 2.70 ms | 84.4 MB |
| **B — efficient:** PCA 128-d + FAISS IVF (240 cells, nprobe 16) | 0.3878 | 0.3275 | 0.089 ms (30× faster) | 28.1 MB |

- **RQ1:** IVF with nprobe 16 loses 2.8 pt Recall@10 for a 6.7× lower p95; nprobe 64 is exact again. Our own
  K-means index matches FAISS IVF on accuracy but is 17× slower (Python loop vs. C++).
- **RQ2:** 128 dimensions keep 79.5% of the variance and lose 2.5 pt Recall@10 for 3× less memory; 64-d loses 9.9 pt.
- **RQ3:** a threshold τ = 0.60 on the top-1 cosine answers 76% of answerable dev questions and refuses 8 of 10
  verified unanswerable ones; k = 5 balances correctness and groundedness.
- **End to end (100 test questions, config A, `openai/gpt-oss-120b`):** 70 answered (correctness 2.26 / 3,
  groundedness 2.36 / 3), 16 refused by τ, 14 refused by the LLM. Without a relevant post in the top 5,
  correctness drops to 1.87 while groundedness stays the same — retrieval is the bottleneck.

Test latency of config A was inflated by background load during the single test run, so speed numbers come
from dev. Limitations are listed in Section 5 of the report.

## Setup

```bash
git clone https://github.com/fafacannon5/FinSight-AI.git && cd FinSight-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # unit tests (synthetic data, no downloads)
```

On macOS, run the tests as `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest -q`; without these two variables
the FAISS tests crash (see [the macOS note](#macos-faiss-and-pytorch-each-ship-their-own-openmp-runtime) below).

The first experiment run downloads two public resources from Hugging Face into its local cache
(`~/.cache/huggingface`): the FiQA / SciFact datasets (`BeIR/fiqa`, `BeIR/fiqa-qrels`, `BeIR/scifact`) and the
encoder `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB). Corpus embeddings are cached in `data/`
(not committed).

Answer generation needs a free Groq API key: put `GROQ_API_KEY=...` in a repo-root `.env` file
(gitignored, never committed). Retrieval experiments need no key.

### macOS: FAISS and PyTorch each ship their own OpenMP runtime

On macOS the PyPI wheels of `torch` and `faiss-cpu` each bundle their own copy of `libomp`. Loading
both into one process and calling `index.search()` aborts the interpreter with `OMP: Error #15`,
followed by a segmentation fault. The code itself is fine.

```bash
# workaround used for the grid runs: one OpenMP thread avoids the duplicate-runtime abort
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest -q tests/test_retrieval.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_experiment.py <config>.yaml
```

`OMP_NUM_THREADS=1` is the part that matters: `KMP_DUPLICATE_LIB_OK=TRUE` on its own lets the
program start with both runtimes loaded, but it segfaults on the full 57,638-document FiQA corpus.

Note that this caps FAISS at one thread, which is unfair to it. Installing `faiss-cpu` from
conda-forge does **not** fix the clash: our `torch` keeps its own `libomp`
(`site-packages/torch/lib/libomp.dylib`), so two runtimes would still be loaded. The clean fix is to
measure each back end in a process that imports only the library it needs — no clash, and every
library keeps its normal thread count:

```bash
python scripts/run_experiment.py experiments/configs/run_e1_001.yaml   # torch only, no env vars
python scripts/bench_faiss_isolated.py --method flat                   # faiss only, no env vars
python scripts/bench_faiss_isolated.py --method ivf --nlist 240 --nprobe 16
```

`TorchKNN` and `KMeansIndex` never import FAISS, so they are unaffected by any of this.

## Reproducing the experiments

Every retrieval run is one YAML file in `experiments/configs/` and writes `metrics.json` to
`experiments/results/<run_id>/` (gitignored). Run from the repo root.

```bash
# smoke test (SciFact, small)
python scripts/run_experiment.py experiments/configs/run_e1_000_scifact_smoke.yaml

# E1 — search algorithm (FiQA dev): exact, FAISS Flat, IVF nprobe 1/4/16/64, our K-means index
for c in run_e1_001 run_e1_002 run_e1_003 run_e1_004 run_e1_005 run_e1_006 run_bonus_001; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_experiment.py experiments/configs/$c.yaml
done

# E1 corpus-size sweep (10k / 25k sub-corpora, exact vs. IVF nprobe 16)
for c in run_e1s_001 run_e1s_002 run_e1s_003 run_e1s_004; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_experiment.py experiments/configs/$c.yaml
done

# E2 — PCA 384/128/64/32 (exact search), then configs B and A in one session (run_e2_004 / 005)
for c in run_e2_000 run_e2_001 run_e2_002 run_e2_003 run_e2_004 run_e2_005; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_experiment.py experiments/configs/$c.yaml
done
```

E3 (reliability, FiQA dev) has a manual step in the middle:

```bash
python scripts/e3_build_unanswerable.py --n 50          # 1. remove labelled posts of 50 questions; sheet for the manual check
python scripts/e3_generate.py --run-id run_e3_gen_001 --n 20 --budget 150000   # 2. LLM answers at k = 1/3/5/10 (needs GROQ_API_KEY)
#                                                          3. manual: OK/DROP verdicts and 1–3 scores (rubric in the report)
python scripts/e3_analyze.py                            # 4. quality vs. k and the τ curve for config A
python scripts/e3_tau_config.py --name config_b --pca-dim 128 --method faiss_ivf --nlist 240 --nprobe 16   # τ for config B
```

The **final test evaluation** is locked: every script refuses the test split unless `--final` is passed.
We ran it once; repeating it is only for checking reproducibility, never for tuning.

```bash
python scripts/run_experiment.py experiments/configs/final_a.yaml --final
python scripts/run_experiment.py experiments/configs/final_b.yaml --final
python scripts/final_generate.py --final                # 100 test questions, config A (needs GROQ_API_KEY)
```

LLM answers come from a hosted model and may differ slightly between runs; the manual scores are ours and are
not regenerated by any script.

## Notebooks (figures and tables for the report)

Run from `notebooks/` after the experiments above. They read the result files and write
`report/figures/*.pdf|png` and `report/tables/*.md|tex`.

| Notebook | Content |
|---|---|
| `01_data_and_embeddings.ipynb` | dataset statistics, text lengths vs. the encoder limit, embedding cache check |
| `02_e1_retrieval_methods.ipynb` | E1: accuracy vs. latency of all search methods, memory, IVF speed-up vs. corpus size |
| `03_e2_pca.ipynb` | E2: explained variance, quality / latency / memory vs. dimension |
| `04_e3_reliability.ipynb` | E3: top-1 score distributions, τ curve, answer quality vs. k |
| `05_final_test_results.ipynb` | final test: dev vs. test retrieval, outcome and quality of the 100 answers |

## Demo

```bash
streamlit run app/streamlit_app.py
```

Uses config A (384-d exact search, k = 5, τ = 0.60). It needs `data/fiqa_corpus_minilm.npy` (created by any
FiQA run above) and `GROQ_API_KEY` in `.env`. Without the key, retrieval and threshold refusals still work;
answering shows an error.

## Layout

`src/finsight/` library (metrics, data loading, embeddings, PCA, retrieval back ends, RAG layer, report helpers) ·
`tests/` unit tests · `scripts/` experiment runners · `experiments/configs/` run configs · `notebooks/` analysis
and figures · `app/` Streamlit demo · `report/` report, slides, figures and tables.

## Reproducibility

Fixed encoder `all-MiniLM-L6-v2`; every embedding is L2-normalized, so cosine similarity equals the inner
product (re-normalized after PCA; queries get the same transform as documents). `RANDOM_SEED = 42` everywhere.
All tuning on dev; the FiQA test split was evaluated once with the final configurations.

## Use of AI tools

We used Claude (Claude Code) and ChatGPT; every use is listed in [AI_USAGE.md](AI_USAGE.md) and summarised in
Section 6 of the report.

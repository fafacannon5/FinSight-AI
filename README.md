# FinSight-AI — Design and Evaluation of a RAG Chatbot: Accuracy–Efficiency Trade-offs in Retrieval

CS 541 Artificial Intelligence · Stevens Institute of Technology · Fall 2026
**Team:** Jing Cannon · Haiyan Zhu

We build a retrieval-augmented chatbot for financial question answering (FiQA) and measure how retrieval
choices trade accuracy against latency and memory:

- **RQ1** exact (our PyTorch brute-force KNN, FAISS Flat) vs. approximate (FAISS IVF) search
- **RQ2** PCA reduction of 384-d embeddings to 128 / 64 / 32 dimensions
- **RQ3** top-k and a similarity threshold for refusing unanswerable questions

## Setup

```bash
git clone <repo-url> && cd FinSight-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # unit tests (synthetic data, no downloads)
```

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

## Run

```bash
python scripts/run_experiment.py experiments/configs/run_e1_000_scifact_smoke.yaml   # pipeline smoke test
streamlit run app/streamlit_app.py                                                   # demo
```

## Layout

`src/finsight/` library · `tests/` unit tests · `scripts/` experiment runner · `notebooks/` analysis & figures ·
`app/` demo · `report/` figures and tables. Data and embedding caches are created locally in `data/`.

## Reproducibility

Fixed encoder `all-MiniLM-L6-v2`, cosine similarity on L2-normalized vectors, `RANDOM_SEED = 42`.
All tuning on train/dev; the FiQA test split is evaluated once with the final configurations.

## Use of AI tools

See [AI_USAGE.md](AI_USAGE.md).

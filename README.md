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

On macOS, the PyPI builds of `torch` and `faiss-cpu` each bundle a copy of `libomp`. Importing both
in one process and calling `index.search()` aborts the interpreter (`OMP: Error #15`, then a
segmentation fault). This breaks `tests/test_retrieval.py` and any experiment that uses the FAISS
back ends — the code itself is fine.

```bash
# quick workaround: one OpenMP thread avoids the duplicate-runtime abort (test_retrieval.py 6/6 pass)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest -q tests/test_retrieval.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_experiment.py <config>.yaml

# permanent fix: install FAISS from conda so both share conda's llvm-openmp
conda install -c conda-forge faiss-cpu
```

`TorchKNN` and `KMeansIndex` are pure PyTorch and unaffected.

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

"""Measure FAISS latency in a process that never imports PyTorch.

Why this script exists
----------------------
On macOS the PyPI wheels of `torch` and `faiss-cpu` each ship their own copy of `libomp`
(we verified three distinct files on our machine: torch's, conda's and the one bundled inside
the faiss wheel). Loading two OpenMP runtimes in one process aborts with `OMP: Error #15`.
The workaround we used for the grid runs, `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`, also
caps FAISS at one thread, which makes its latency look worse than it is.

`scripts/run_experiment.py` always imports `finsight.retrieval`, which imports torch at module
level, so any FAISS run through it lives in a process with two runtimes. This script measures the
FAISS back ends alone: it imports `numpy` + `faiss` only, so FAISS keeps its normal thread count
and the numbers are directly comparable with a torch-only run.

It rebuilds the two indexes with the same calls as `finsight.retrieval.FaissFlat` / `FaissIVF`
(same quantizer, same `cp.seed = 42`, same metric) and uses the same timing hygiene as
`finsight.timing.measure_latency`: single-query batches, warm-up excluded, repeated passes.

Usage
-----
    python scripts/bench_faiss_isolated.py --method flat
    python scripts/bench_faiss_isolated.py --method ivf --nlist 240 --nprobe 16
    python scripts/bench_faiss_isolated.py --dataset fiqa --split test --method ivf --nprobe 16 --final

The embedding cache is read from `data/{dataset}_corpus_minilm.npy` and
`data/{dataset}_{split}_queries_minilm.npy`; build it first by running
`scripts/run_experiment.py` on the corresponding config.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def build(method: str, doc_emb: np.ndarray, nlist: int, nprobe: int, seed: int):
    """Build the same index as finsight.retrieval.FaissFlat / FaissIVF. doc_emb: (N, d) float32."""
    import faiss  # imported here so this module can be imported in a torch-free process

    doc_emb = np.ascontiguousarray(doc_emb, dtype=np.float32)
    d = doc_emb.shape[1]
    if method == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(doc_emb)
        return index
    quantizer = faiss.IndexFlatIP(d)  # query vs. cell centroids
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed              # K-means seed → reproducible cells (our seed 42)
    index.train(doc_emb)              # corpus embeddings only, never queries
    index.add(doc_emb)
    index.nprobe = nprobe             # nprobe == nlist degenerates to exact search
    return index


def measure(index, q_emb: np.ndarray, k: int, warmup: int, repeats: int, max_queries: int | None):
    """Time index.search one query at a time. Returns mean and p95 in milliseconds."""
    q = q_emb if max_queries is None else q_emb[:max_queries]
    for i in range(min(warmup, len(q))):
        index.search(q[i:i + 1], k)
    times = []
    for _ in range(repeats):
        for i in range(len(q)):
            t0 = time.perf_counter()
            index.search(q[i:i + 1], k)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--method", choices=["flat", "ivf"], default="flat")
    ap.add_argument("--nlist", type=int, default=240)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final", action="store_true", help="required to touch the FiQA test split")
    args = ap.parse_args()

    if args.dataset == "fiqa" and args.split == "test" and not args.final:
        print("Refusing: FiQA test split is locked. Use --final only for the final evaluation.",
              file=sys.stderr)
        return 2

    if "torch" in sys.modules:  # the whole point of this script
        print("Refusing: torch is already imported, so FAISS would share the process with a "
              "second OpenMP runtime.", file=sys.stderr)
        return 3

    doc_path = REPO / "data" / f"{args.dataset}_corpus_minilm.npy"
    qry_path = REPO / "data" / f"{args.dataset}_{args.split}_queries_minilm.npy"
    doc_emb = np.ascontiguousarray(np.load(doc_path), dtype=np.float32)
    q_emb = np.ascontiguousarray(np.load(qry_path), dtype=np.float32)

    index = build(args.method, doc_emb, args.nlist, args.nprobe, args.seed)
    mean_ms, p95_ms = measure(index, q_emb, args.k, args.warmup, args.repeats, args.queries)

    out = {"dataset": args.dataset, "split": args.split,
           "method": f"faiss_{args.method}" + ("" if args.method == "flat"
                                               else f" nlist={args.nlist} nprobe={args.nprobe}"),
           "n_docs": int(doc_emb.shape[0]), "n_queries": int(q_emb.shape[0]), "dim": int(doc_emb.shape[1]),
           "k": args.k, "latency_mean_ms": mean_ms, "latency_p95_ms": p95_ms,
           "torch_imported": "torch" in sys.modules}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

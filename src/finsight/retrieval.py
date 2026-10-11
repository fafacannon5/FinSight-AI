"""Nearest-neighbor search back ends with one shared interface.

All indexes assume rows are L2-normalized, so inner product == cosine similarity.
    fit(doc_emb)            doc_emb: (N, d) float32
    search(q_emb, k)        q_emb: (n_q, d) float32 → (scores (n_q, k), indices (n_q, k)), best first
    memory_mb()             approximate memory used by the index, in MB
Course link: exact KNN = CS 541 Week 10; IVF = K-means partitioning (Week 4).
"""
from __future__ import annotations

import numpy as np


class TorchKNN:
    """Exact brute-force KNN implemented by us in PyTorch (the method under study)."""

    def __init__(self, device: str = "cpu", batch_size: int = 1024):
        self.device = device
        self.batch_size = batch_size
        self.doc = None  # torch.Tensor (N, d)

    def fit(self, doc_emb: np.ndarray) -> "TorchKNN":
        # 1. convert to a float32 torch tensor on self.device and store in self.doc
        raise NotImplementedError

    def search(self, q_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # 1. convert q_emb to a torch tensor
        # 2. for each batch of queries: scores = q_batch @ self.doc.T          # (b, N)
        # 3. torch.topk(scores, k, dim=1) → values, indices (already sorted best first)
        # 4. concatenate batches, return numpy arrays
        # Complexity: O(n_q · N · d) — say this in the report (CLO2).
        raise NotImplementedError

    def memory_mb(self) -> float:
        # bytes of self.doc = N · d · 4
        raise NotImplementedError


class FaissFlat:
    """FAISS exact inner-product index (reference implementation for correctness + speed)."""

    def __init__(self):
        self.index = None

    def fit(self, doc_emb: np.ndarray) -> "FaissFlat":
        # import faiss; self.index = faiss.IndexFlatIP(d); self.index.add(doc_emb)
        raise NotImplementedError

    def search(self, q_emb: np.ndarray, k: int):
        # return self.index.search(q_emb, k)  → (scores, indices)
        raise NotImplementedError

    def memory_mb(self) -> float:
        raise NotImplementedError


class FaissIVF:
    """FAISS IVF index: K-means splits the corpus into `nlist` cells; a query searches `nprobe` cells.

    Train (K-means) on the corpus embeddings only. Tune nlist / nprobe on dev, never on test.
    """

    def __init__(self, nlist: int = 240, nprobe: int = 8, seed: int = 42):
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed
        self.index = None

    def fit(self, doc_emb: np.ndarray) -> "FaissIVF":
        # 1. quantizer = faiss.IndexFlatIP(d)
        # 2. self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        # 3. set the K-means seed if your faiss version allows (self.index.cp.seed = self.seed)
        # 4. self.index.train(doc_emb); self.index.add(doc_emb); self.index.nprobe = self.nprobe
        raise NotImplementedError

    def search(self, q_emb: np.ndarray, k: int):
        raise NotImplementedError

    def memory_mb(self) -> float:
        raise NotImplementedError


class KMeansIndex:
    """BONUS (optional): our own IVF-style index using K-means written in PyTorch. Compare with FaissIVF."""

    def __init__(self, n_clusters: int = 240, nprobe: int = 8, n_iter: int = 20, seed: int = 42):
        self.n_clusters, self.nprobe, self.n_iter, self.seed = n_clusters, nprobe, n_iter, seed

    def fit(self, doc_emb: np.ndarray) -> "KMeansIndex":
        raise NotImplementedError

    def search(self, q_emb: np.ndarray, k: int):
        raise NotImplementedError

    def memory_mb(self) -> float:
        raise NotImplementedError


def build_index(method: str, **params):
    """Factory used by scripts/run_experiment.py: 'torch_knn' | 'faiss_flat' | 'faiss_ivf' | 'kmeans_index'."""
    table = {"torch_knn": TorchKNN, "faiss_flat": FaissFlat, "faiss_ivf": FaissIVF, "kmeans_index": KMeansIndex}
    if method not in table:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(table)}")
    return table[method](**params)

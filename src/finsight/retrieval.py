"""Nearest-neighbor search back ends with one shared interface.

All indexes assume rows are L2-normalized, so inner product == cosine similarity.
    fit(doc_emb)            doc_emb: (N, d) float32
    search(q_emb, k)        q_emb: (n_q, d) float32 → (scores (n_q, k), indices (n_q, k)), best first
    memory_mb()             approximate memory used by the index, in MB
Course link: exact KNN = CS 541 Week 10; IVF = K-means partitioning (Week 4).
"""
from __future__ import annotations

import numpy as np
import torch


class TorchKNN:
    """Exact brute-force KNN implemented by us in PyTorch (the method under study)."""

    def __init__(self, device: str = "cpu", batch_size: int = 1024):
        self.device = device
        self.batch_size = batch_size
        self.doc = None  # torch.Tensor (N, d)

    def fit(self, doc_emb: np.ndarray) -> "TorchKNN":
        # 1. convert to a float32 torch tensor on self.device and store in self.doc
        self.doc = torch.as_tensor(np.ascontiguousarray(doc_emb), dtype=torch.float32, device=self.device)
        return self

    def search(self, q_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # 1. convert q_emb to a torch tensor
        # 2. for each batch of queries: scores = q_batch @ self.doc.T          # (b, N)
        # 3. torch.topk(scores, k, dim=1) → values, indices (already sorted best first)
        # 4. concatenate batches, return numpy arrays
        # Complexity: O(n_q · N · d) — say this in the report (CLO2).
        q = torch.as_tensor(np.ascontiguousarray(q_emb), dtype=torch.float32, device=self.device)
        chunks_scores, chunks_idx = [], []
        with torch.no_grad():  # inference only: no autograd graph, less memory
            for start in range(0, q.shape[0], self.batch_size):
                scores = q[start:start + self.batch_size] @ self.doc.T  # (b, N); unit rows → cosine
                values, indices = torch.topk(scores, k, dim=1)          # already sorted, best first
                chunks_scores.append(values)
                chunks_idx.append(indices)
        # .cpu() before .numpy(): numpy() is not defined for MPS / CUDA tensors
        return torch.cat(chunks_scores).cpu().numpy(), torch.cat(chunks_idx).cpu().numpy()

    def memory_mb(self) -> float:
        # bytes of self.doc = N · d · 4
        return self.doc.numel() * self.doc.element_size() / 2**20


class FaissFlat:
    """FAISS exact inner-product index (reference implementation for correctness + speed)."""

    def __init__(self):
        self.index = None

    def fit(self, doc_emb: np.ndarray) -> "FaissFlat":
        # import faiss; self.index = faiss.IndexFlatIP(d); self.index.add(doc_emb)
        import faiss  # imported inside the method so the tests skip cleanly when faiss is missing
        doc_emb = np.ascontiguousarray(doc_emb, dtype=np.float32)
        self.index = faiss.IndexFlatIP(doc_emb.shape[1])
        self.index.add(doc_emb)
        return self

    def search(self, q_emb: np.ndarray, k: int):
        # return self.index.search(q_emb, k)  → (scores, indices)
        return self.index.search(np.ascontiguousarray(q_emb, dtype=np.float32), k)

    def memory_mb(self) -> float:
        return self.index.ntotal * self.index.d * 4 / 2**20


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
        import faiss
        doc_emb = np.ascontiguousarray(doc_emb, dtype=np.float32)
        d = doc_emb.shape[1]
        quantizer = faiss.IndexFlatIP(d)  # compares the query against the cell centroids
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.cp.seed = self.seed    # K-means seed → reproducible cell assignment (our seed 42)
        self.index.train(doc_emb)         # corpus embeddings only, never queries
        self.index.add(doc_emb)
        self.index.nprobe = self.nprobe   # cells visited per query; nprobe == nlist is exact
        return self

    def search(self, q_emb: np.ndarray, k: int):
        return self.index.search(np.ascontiguousarray(q_emb, dtype=np.float32), k)

    def memory_mb(self) -> float:
        return self.index.ntotal * self.index.d * 4 / 2**20


class KMeansIndex:

    def __init__(self, n_clusters: int = 240, nprobe: int = 8, n_iter: int = 20, seed: int = 42):
        self.n_clusters, self.nprobe, self.n_iter, self.seed = n_clusters, nprobe, n_iter, seed
        self.doc = None        # (N, d)
        self.centroids = None  # (C, d)
        self.order = None      # (N,) doc ids sorted by cluster
        self.offsets = None    # (C+1,) cluster c ids = order[offsets[c]:offsets[c+1]]

    def fit(self, doc_emb: np.ndarray) -> "KMeansIndex":
        # 1. numpy → float32 tensor
        x = torch.as_tensor(np.ascontiguousarray(doc_emb), dtype=torch.float32).clone()
        N, d = x.shape
        C = self.n_clusters
        if N < C:
            raise ValueError(f"N={N} < n_clusters={C}")

        # 2. init centroids from random docs (seeded generator → reproducible)
        g = torch.Generator().manual_seed(self.seed)
        centroids = x[torch.randperm(N, generator=g)[:C]].clone()

        # 3. Lloyd iterations: assign by inner-product argmax (== nearest for L2-normalized rows)
        for _ in range(self.n_iter):
            assign = (x @ centroids.T).argmax(dim=1)
            counts = torch.bincount(assign, minlength=C).to(x.dtype)
            new = torch.zeros_like(centroids)
            new.index_add_(0, assign, x)
            new = new / counts.clamp_min(1.0).unsqueeze(1)
            empty = counts == 0
            if empty.any():  # reseed empty clusters
                ridx = torch.randperm(N, generator=g)[:int(empty.sum())]
                new[empty] = x[ridx]
            centroids = new

        # 4. inverted lists: sort doc ids by cluster, store offsets
        assign = (x @ centroids.T).argmax(dim=1)
        self.order = torch.argsort(assign)
        counts = torch.bincount(assign, minlength=C)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        self.doc, self.centroids = x, centroids
        return self

    def search(self, q_emb: np.ndarray, k: int):
        # 1. convert q_emb to a tensor
        q = torch.as_tensor(np.ascontiguousarray(q_emb), dtype=torch.float32)
        nprobe = min(self.nprobe, self.n_clusters)

        # 2. pad with -inf / -1 when a query has < k candidates (same as faiss)
        out_s = torch.full((q.shape[0], k), -float("inf"), dtype=torch.float32)
        out_i = torch.full((q.shape[0], k), -1, dtype=torch.long)

        # 3. per query: pick top-nprobe cells, brute-force only docs in those cells
        with torch.no_grad():
            probe = (q @ self.centroids.T).topk(nprobe, dim=1).indices
            for i in range(q.shape[0]):
                cand = torch.cat([self.order[int(self.offsets[c]):int(self.offsets[c + 1])]
                                  for c in probe[i].tolist()])
                if cand.numel() == 0:
                    continue
                kk = min(k, cand.numel())
                v, j = (self.doc[cand] @ q[i]).topk(kk)  # unit rows → cosine, best first
                out_s[i, :kk], out_i[i, :kk] = v, cand[j]

        # 4. back to numpy
        return out_s.numpy(), out_i.numpy()

    def memory_mb(self) -> float:
        # bytes of doc + centroids + inverted-list ids
        if self.doc is None:
            return 0.0
        b = (self.doc.numel() * self.doc.element_size()
             + self.centroids.numel() * self.centroids.element_size()
             + self.order.numel() * self.order.element_size())
        return b / 2**20


def build_index(method: str, **params):
    """Factory used by scripts/run_experiment.py: 'torch_knn' | 'faiss_flat' | 'faiss_ivf' | 'kmeans_index'."""
    table = {"torch_knn": TorchKNN, "faiss_flat": FaissFlat, "faiss_ivf": FaissIVF, "kmeans_index": KMeansIndex}
    if method not in table:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(table)}")
    return table[method](**params)

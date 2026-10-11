"""PCA from scratch in PyTorch (CS 541 Week 5), used for E2.

Rules (AGENTS.md Hard rule 2):
- fit on CORPUS embeddings only;
- transform queries with the SAME mean and components;
- re-normalize rows to unit length after projecting, so inner product is still cosine.
"""
from __future__ import annotations

import numpy as np


class PCAReducer:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None                 # (d,)
        self.components_ = None           # (n_components, d), rows = principal directions
        self.explained_variance_ratio_ = None  # (n_components,), descending

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """X: (N, d) corpus embeddings."""
        # 1. mean_ = X.mean(axis=0); Xc = X - mean_
        # 2. SVD: U, S, Vt = torch.linalg.svd(torch.from_numpy(Xc), full_matrices=False)
        # 3. components_ = Vt[:n_components]
        # 4. variance per component = S**2 / (N - 1); ratio = var / var.sum() (over ALL components), keep first n
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, n_components) float32 with unit-norm rows."""
        # 1. Z = (X - mean_) @ components_.T
        # 2. Z = Z / ||Z|| row-wise (guard against division by zero)
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

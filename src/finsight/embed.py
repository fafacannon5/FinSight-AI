"""Encode texts with the fixed encoder and cache results as .npy (data/ is gitignored)."""
from __future__ import annotations

import os

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode(texts: list[str], batch_size: int = 64, model_name: str = MODEL_NAME) -> np.ndarray:
    """Return (len(texts), 384) float32, L2-normalized rows."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float32)


def load_or_encode(cache_path: str, texts: list[str], **kwargs) -> np.ndarray:
    """Load cached embeddings if present and the row count matches, else encode and save."""
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            return emb
    emb = encode(texts, **kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, emb)
    return emb

"""Encode texts with the fixed encoder and cache results as .npy (data/ is gitignored)."""
from __future__ import annotations

import json
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


def ids_path_for(cache_path: str) -> str:
    """data/x.npy → data/x.ids.json: the id of every row, in row order."""
    return os.path.splitext(cache_path)[0] + ".ids.json"


def load_or_encode(cache_path: str, texts: list[str], ids: list[str], **kwargs) -> np.ndarray:
    """Return (len(texts), 384) embeddings; row i belongs to ids[i].

    Reuses the cache only if the saved id list equals `ids` exactly (same ids, same order).
    A row count match is not enough: the same number of rows in a different order would
    silently pair every vector with the wrong document.
    """
    if len(ids) != len(texts):
        raise ValueError(f"len(ids)={len(ids)} != len(texts)={len(texts)}")
    ids_path = ids_path_for(cache_path)
    if os.path.exists(cache_path) and os.path.exists(ids_path):
        with open(ids_path) as f:
            if json.load(f) == list(ids):
                return np.load(cache_path)
    emb = encode(texts, **kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, emb)
    with open(ids_path, "w") as f:
        json.dump(list(ids), f)
    return emb

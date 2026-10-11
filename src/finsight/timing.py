"""Latency measurement with consistent hygiene: single-query batches, warm-up excluded, repeated runs."""
from __future__ import annotations

import time

import numpy as np


def measure_latency(index, q_emb: np.ndarray, k: int, warmup: int = 10, repeats: int = 3,
                    max_queries: int | None = None) -> dict[str, float]:
    """Time index.search one query at a time. Returns mean and p95 in milliseconds (over all repeats)."""
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
    return {"latency_mean_ms": float(arr.mean()), "latency_p95_ms": float(np.percentile(arr, 95))}


def timed(fn, *args, **kwargs):
    """Run fn and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0

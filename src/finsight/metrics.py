"""Retrieval metrics (BEIR-style, binary relevance).

Conventions used everywhere:
    retrieved: dict[qid, list[doc_id]]  -- ranked, best first
    qrels:     dict[qid, dict[doc_id, int]]  -- relevance > 0 means relevant
Each metric is computed per query and averaged over the queries in `qrels`
that have at least one relevant document.
"""
from __future__ import annotations

import math


def _relevant(qrels_q: dict[str, int]) -> set[str]:
    return {d for d, r in qrels_q.items() if r > 0}


def recall_at_k(retrieved: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> float:
    """Mean over queries of |relevant ∩ top-k| / |relevant|.

    Example: relevant = {d1, d2}, ranking = [d1, d3, d2] → Recall@1 = 0.5, Recall@3 = 1.0.
    """
    # 1. loop over qids in qrels; skip queries with no relevant docs
    # 2. top = set(retrieved.get(qid, [])[:k])
    # 3. per-query recall = len(top & relevant) / len(relevant)
    # 4. return the mean
    raise NotImplementedError


def mrr_at_k(retrieved: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """Mean reciprocal rank of the FIRST relevant doc within top-k (0 if none).

    Example: ranking = [d3, d1, ...], d1 relevant → 1/2.
    """
    # 1. for each query, scan retrieved[qid][:k] with rank starting at 1
    # 2. at the first relevant doc, add 1/rank and stop; if none, add 0
    # 3. return the mean
    raise NotImplementedError


def ndcg_at_k(retrieved: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """nDCG@k with binary gains: DCG = Σ_{rank i≥1, doc relevant} 1 / log2(i + 1).

    IDCG = the DCG of an ideal ranking that puts min(|relevant|, k) relevant docs first.
    """
    # 1. DCG over retrieved[qid][:k]
    # 2. IDCG = sum(1 / log2(i + 1) for i in 1..min(len(relevant), k))
    # 3. per-query nDCG = DCG / IDCG ; return the mean
    raise NotImplementedError


def evaluate_all(retrieved, qrels, ks=(5, 10)) -> dict[str, float]:
    """Convenience wrapper used by scripts/run_experiment.py."""
    out = {f"recall@{k}": recall_at_k(retrieved, qrels, k) for k in ks}
    out["mrr@10"] = mrr_at_k(retrieved, qrels, 10)
    out["ndcg@10"] = ndcg_at_k(retrieved, qrels, 10)
    return out


__all__ = ["recall_at_k", "mrr_at_k", "ndcg_at_k", "evaluate_all"]

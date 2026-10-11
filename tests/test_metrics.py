"""Hand-checkable cases for retrieval metrics."""
import math

import pytest

from finsight.metrics import mrr_at_k, ndcg_at_k, recall_at_k

QRELS = {"q1": {"d1": 1, "d2": 1}, "q2": {"d9": 1}}
RETRIEVED = {"q1": ["d1", "d3", "d2"], "q2": ["d5", "d9", "d7"]}


def test_recall_at_k_single_query():
    q = {"q1": QRELS["q1"]}
    assert recall_at_k(RETRIEVED, q, 1) == pytest.approx(0.5)
    assert recall_at_k(RETRIEVED, q, 3) == pytest.approx(1.0)


def test_recall_averages_over_queries():
    # q1@2 = 0.5 (d1 only), q2@2 = 1.0 (d9 at rank 2) → mean 0.75
    assert recall_at_k(RETRIEVED, QRELS, 2) == pytest.approx(0.75)


def test_recall_missing_query_counts_as_zero():
    assert recall_at_k({}, {"q2": {"d9": 1}}, 10) == pytest.approx(0.0)


def test_mrr_first_relevant_rank():
    # q1: first relevant at rank 1 → 1 ; q2: rank 2 → 0.5 ; mean 0.75
    assert mrr_at_k(RETRIEVED, QRELS, 10) == pytest.approx(0.75)
    # cut-off: q2's relevant doc is outside top-1 → contributes 0
    assert mrr_at_k(RETRIEVED, QRELS, 1) == pytest.approx(0.5)


def test_ndcg_binary_gain():
    q = {"q1": QRELS["q1"]}
    dcg = 1 / math.log2(2) + 1 / math.log2(4)   # d1 at rank 1, d2 at rank 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # two relevant docs at ranks 1, 2
    assert ndcg_at_k(RETRIEVED, q, 10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k({"q1": ["d1", "d2", "d3"]}, {"q1": QRELS["q1"]}, 10) == pytest.approx(1.0)


def test_zero_relevance_is_not_relevant():
    qrels = {"q1": {"d1": 0, "d2": 1}}
    assert recall_at_k({"q1": ["d1"]}, qrels, 1) == pytest.approx(0.0)

"""Unanswerable-set construction (no downloads)."""
from finsight.data import make_unanswerable

QRELS = {f"q{i}": {f"d{i}": 1, f"d{i + 100}": 1} for i in range(20)}
CORPUS = [f"d{i}" for i in range(200)]


def test_removes_all_relevant_docs_of_selected_queries():
    qids, removed = make_unanswerable(QRELS, CORPUS, n=5, seed=42)
    assert len(qids) == 5 and len(set(qids)) == 5
    for q in qids:
        assert set(QRELS[q]) <= removed


def test_deterministic_with_seed():
    assert make_unanswerable(QRELS, CORPUS, 5, seed=42) == make_unanswerable(QRELS, CORPUS, 5, seed=42)

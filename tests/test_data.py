"""Unanswerable-set construction and id checks (no downloads)."""
import pytest

from finsight.data import check_ids, make_unanswerable

QRELS = {f"q{i}": {f"d{i}": 1, f"d{i + 100}": 1} for i in range(20)}
CORPUS = [f"d{i}" for i in range(200)]


def test_removes_all_relevant_docs_of_selected_queries():
    qids, removed = make_unanswerable(QRELS, CORPUS, n=5, seed=42)
    assert len(qids) == 5 and len(set(qids)) == 5
    for q in qids:
        assert set(QRELS[q]) <= removed


def test_deterministic_with_seed():
    assert make_unanswerable(QRELS, CORPUS, 5, seed=42) == make_unanswerable(QRELS, CORPUS, 5, seed=42)


def test_check_ids_accepts_consistent_str_ids():
    check_ids({"d1": "doc text"}, {"q1": "question"}, {"q1": {"d1": 1}})  # must not raise


def test_check_ids_rejects_int_ids():
    # the HF qrels files store ids as int64; forgetting str() must fail loudly, not score 0
    with pytest.raises(ValueError, match="must be str"):
        check_ids({"1": "doc text"}, {"1": "question"}, {1: {1: 1}})


def test_check_ids_rejects_qrels_doc_missing_from_corpus():
    with pytest.raises(ValueError, match="not found in corpus"):
        check_ids({"d1": "doc text"}, {"q1": "question"}, {"q1": {"d2": 1}})

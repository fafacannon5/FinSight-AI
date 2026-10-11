"""Exact search must match a brute-force numpy reference; FAISS back ends must agree with it."""
import numpy as np
import pytest

from finsight.retrieval import TorchKNN

RNG = np.random.default_rng(42)


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


DOCS = _unit(RNG.normal(size=(500, 32)))
QUERIES = _unit(RNG.normal(size=(20, 32)))
K = 10


def _reference_topk(q, d, k):
    scores = q @ d.T
    idx = np.argsort(-scores, axis=1)[:, :k]
    return np.take_along_axis(scores, idx, axis=1), idx


def test_torch_knn_matches_numpy():
    scores, idx = TorchKNN().fit(DOCS).search(QUERIES, K)
    ref_scores, ref_idx = _reference_topk(QUERIES, DOCS, K)
    assert idx.shape == (20, K)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(scores, ref_scores, atol=1e-5)


def test_torch_knn_scores_sorted_descending():
    scores, _ = TorchKNN().fit(DOCS).search(QUERIES, K)
    assert np.all(np.diff(scores, axis=1) <= 1e-6)


def test_torch_knn_small_batches_same_result():
    _, a = TorchKNN(batch_size=3).fit(DOCS).search(QUERIES, K)
    _, b = TorchKNN(batch_size=1024).fit(DOCS).search(QUERIES, K)
    np.testing.assert_array_equal(a, b)


def test_torch_knn_memory():
    assert TorchKNN().fit(DOCS).memory_mb() == pytest.approx(500 * 32 * 4 / 2**20, rel=0.01)


def test_faiss_flat_matches_torch_knn():
    pytest.importorskip("faiss")
    from finsight.retrieval import FaissFlat
    _, a = FaissFlat().fit(DOCS).search(QUERIES, K)
    _, b = TorchKNN().fit(DOCS).search(QUERIES, K)
    np.testing.assert_array_equal(a, b)


def test_faiss_ivf_full_probe_is_exact():
    pytest.importorskip("faiss")
    from finsight.retrieval import FaissIVF
    _, a = FaissIVF(nlist=8, nprobe=8).fit(DOCS).search(QUERIES, K)
    _, ref = _reference_topk(QUERIES, DOCS, K)
    np.testing.assert_array_equal(a, ref)

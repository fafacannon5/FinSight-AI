"""PCA must project, re-normalize, and reuse the corpus mean for queries."""
import numpy as np
import pytest

from finsight.pca import PCAReducer

RNG = np.random.default_rng(42)
X = RNG.normal(size=(300, 16)).astype(np.float32) * np.linspace(3, 0.1, 16).astype(np.float32)


def test_shape_and_unit_norm():
    Z = PCAReducer(4).fit_transform(X)
    assert Z.shape == (300, 4)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)


def test_explained_variance_ratio_descending_and_bounded():
    p = PCAReducer(5).fit(X)
    r = np.asarray(p.explained_variance_ratio_)
    assert r.shape == (5,)
    assert np.all(np.diff(r) <= 1e-7)
    assert 0 < r.sum() <= 1 + 1e-6


def test_queries_use_corpus_mean():
    p = PCAReducer(4).fit(X)
    q = RNG.normal(size=(1, 16)).astype(np.float32)
    expected = (q - X.mean(axis=0)) @ np.asarray(p.components_).T
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(p.transform(q), expected, atol=1e-4)


def test_matches_sklearn_up_to_sign():
    sk = pytest.importorskip("sklearn.decomposition")
    ours = np.asarray(PCAReducer(3).fit(X).components_)
    ref = sk.PCA(n_components=3).fit(X).components_
    for a, b in zip(ours, ref):
        assert abs(abs(float(np.dot(a, b))) - 1.0) < 1e-3

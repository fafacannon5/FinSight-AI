"""Embedding cache keyed by id list (no model download: encode is replaced by a fake)."""
import numpy as np
import pytest

from finsight import embed


@pytest.fixture
def fake_encode(monkeypatch):
    calls = []

    def fake(texts, **kwargs):
        calls.append(list(texts))
        return np.arange(len(texts), dtype=np.float32).reshape(-1, 1)  # row i = [i]

    monkeypatch.setattr(embed, "encode", fake)
    return calls


def test_second_call_with_same_ids_uses_cache(tmp_path, fake_encode):
    path = str(tmp_path / "c.npy")
    embed.load_or_encode(path, ["a", "b"], ids=["d1", "d2"])
    embed.load_or_encode(path, ["a", "b"], ids=["d1", "d2"])
    assert len(fake_encode) == 1


def test_same_count_different_order_re_encodes(tmp_path, fake_encode):
    # the silent bug this guards against: same row count, rows belong to different docs
    path = str(tmp_path / "c.npy")
    embed.load_or_encode(path, ["a", "b"], ids=["d1", "d2"])
    embed.load_or_encode(path, ["b", "a"], ids=["d2", "d1"])
    assert len(fake_encode) == 2


def test_cache_without_id_file_re_encodes(tmp_path, fake_encode):
    path = str(tmp_path / "c.npy")
    np.save(path, np.zeros((2, 1), dtype=np.float32))  # old-style cache: no .ids.json
    embed.load_or_encode(path, ["a", "b"], ids=["d1", "d2"])
    assert len(fake_encode) == 1


def test_ids_and_texts_length_mismatch_raises(tmp_path, fake_encode):
    with pytest.raises(ValueError):
        embed.load_or_encode(str(tmp_path / "c.npy"), ["a", "b"], ids=["d1"])

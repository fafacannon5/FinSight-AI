"""Refusal rule and refusal curve."""
import numpy as np
import pytest

from finsight.rag import build_prompt, refusal_curve, should_refuse


def test_should_refuse_threshold():
    assert should_refuse(0.30, 0.50) is True
    assert should_refuse(0.50, 0.50) is False
    assert should_refuse(0.80, 0.50) is False


def test_refusal_curve_values():
    ans = np.array([0.9, 0.7, 0.4])
    unans = np.array([0.6, 0.3])
    curve = refusal_curve(ans, unans, np.array([0.5]))
    assert curve[0]["tau"] == pytest.approx(0.5)
    assert curve[0]["answer_rate"] == pytest.approx(2 / 3)
    assert curve[0]["correct_refusal_rate"] == pytest.approx(1 / 2)


def test_refusal_curve_monotonic():
    rng = np.random.default_rng(42)
    curve = refusal_curve(rng.uniform(0.3, 0.9, 50), rng.uniform(0.1, 0.6, 50), np.linspace(0, 1, 21))
    answer = [c["answer_rate"] for c in curve]
    refuse = [c["correct_refusal_rate"] for c in curve]
    assert all(a >= b for a, b in zip(answer, answer[1:]))
    assert all(a <= b for a, b in zip(refuse, refuse[1:]))


def test_prompt_contains_ids_and_question():
    p = build_prompt("What is an ETF?", [("d42", "An ETF is a fund..."), ("d7", "Index funds...")])
    assert "What is an ETF?" in p and "d42" in p and "d7" in p


class _FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_generate_sends_fixed_model_at_temperature_zero(monkeypatch):
    requests = pytest.importorskip("requests")
    from finsight import rag

    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(json)
        return _FakeResponse(200, {"choices": [{"message": {"content": "  An ETF is a fund [d42]. "}}]})

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(requests, "post", fake_post)
    assert rag.generate("q") == "An ETF is a fund [d42]."
    assert sent["model"] == rag.LLM_MODEL and sent["temperature"] == 0


def test_generate_retries_after_rate_limit(monkeypatch):
    requests = pytest.importorskip("requests")
    from finsight import rag

    replies = iter([_FakeResponse(429, headers={"retry-after": "0"}),
                    _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})])
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(replies))
    assert rag.generate("q") == "ok"

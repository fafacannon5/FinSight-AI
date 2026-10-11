"""Decision + generation layer (E3).

refuse if the best retrieval similarity < tau; otherwise build a prompt from the top-k documents
and ask the fixed LLM (see wiki/decisions/llm_choice.md) to answer and cite document ids.
"""
from __future__ import annotations

import numpy as np

REFUSAL_TEXT = "I couldn't find enough information in the documents to answer this question."


def should_refuse(top_score: float, tau: float) -> bool:
    """True when the best similarity is strictly below the threshold."""
    raise NotImplementedError


def refusal_curve(answerable_top_scores: np.ndarray, unanswerable_top_scores: np.ndarray,
                  taus: np.ndarray) -> list[dict]:
    """For each tau: answer_rate on answerable queries and correct_refusal_rate on unanswerable ones.

    answer_rate          = fraction of answerable queries with top score >= tau
    correct_refusal_rate = fraction of unanswerable queries with top score <  tau
    Returns [{"tau": t, "answer_rate": a, "correct_refusal_rate": r}, ...] in the order of `taus`.
    """
    raise NotImplementedError


def build_prompt(question: str, docs: list[tuple[str, str]]) -> str:
    """docs = [(doc_id, text), ...] best first. The prompt must ask the model to answer ONLY from the
    documents and to cite ids like [doc_id]."""
    raise NotImplementedError


def generate(prompt: str, **llm_kwargs) -> str:
    """Call the fixed LLM. Implementation depends on wiki/decisions/llm_choice.md (Colab model or API).
    Read any API key from the environment (.env), never from code."""
    raise NotImplementedError

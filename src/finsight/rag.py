"""Decision + generation layer (E3).

refuse if the best retrieval similarity < tau; otherwise build a prompt from the top-k documents
and ask the fixed LLM (see wiki/decisions/llm_choice.md) to answer and cite document ids.
"""
from __future__ import annotations

import numpy as np

REFUSAL_TEXT = "I couldn't find enough information in the documents to answer this question."


def should_refuse(top_score: float, tau: float) -> bool:
    """True when the best similarity is strictly below the threshold."""
    return bool(top_score < tau)


def refusal_curve(answerable_top_scores: np.ndarray, unanswerable_top_scores: np.ndarray,
                  taus: np.ndarray) -> list[dict]:
    """For each tau: answer_rate on answerable queries and correct_refusal_rate on unanswerable ones.

    answer_rate          = fraction of answerable queries with top score >= tau
    correct_refusal_rate = fraction of unanswerable queries with top score <  tau
    Returns [{"tau": t, "answer_rate": a, "correct_refusal_rate": r}, ...] in the order of `taus`.
    """
    ans = np.asarray(answerable_top_scores, dtype=float)
    unans = np.asarray(unanswerable_top_scores, dtype=float)
    return [{"tau": float(t),
             "answer_rate": float((ans >= t).mean()),       # same rule as should_refuse: answer iff score >= tau
             "correct_refusal_rate": float((unans < t).mean())}
            for t in taus]


def build_prompt(question: str, docs: list[tuple[str, str]]) -> str:
    """docs = [(doc_id, text), ...] best first. The prompt must ask the model to answer ONLY from the
    documents and to cite ids like [doc_id]."""
    context = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in docs)
    return (
        "You are a financial question-answering assistant.\n"
        "Answer the question using ONLY the documents below. After every claim, cite the id of the "
        "document it comes from in square brackets, e.g. [d42].\n"
        f'If the documents do not contain the answer, reply exactly: "{REFUSAL_TEXT}"\n\n'
        f"Documents:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


LLM_MODEL = "openai/gpt-oss-120b"  # fixed by wiki/decisions/llm_choice.md (Groq)
REASONING_EFFORT = "low"           # gpt-oss "thinks" before answering; low keeps token use small
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate(prompt: str, model: str = LLM_MODEL, max_tokens: int = 512, max_retries: int = 3,
             timeout_s: float = 60.0) -> str:
    """Call the fixed LLM. Implementation depends on wiki/decisions/llm_choice.md (Colab model or API).
    Read any API key from the environment (.env), never from code.

    temperature = 0 so repeated calls give (nearly) the same answer. On HTTP 429 (rate limit) waits
    for the server's Retry-After and tries again, at most `max_retries` times.
    """
    import os
    import time
    from pathlib import Path

    import requests
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # repo-root .env; never committed
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set: put it in the repo-root .env file")

    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "reasoning_effort": REASONING_EFFORT,
               "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(max_retries + 1):
        resp = requests.post(GROQ_URL, json=payload, timeout=timeout_s,
                             headers={"Authorization": f"Bearer {key}"})
        if resp.status_code == 429 and attempt < max_retries:
            time.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))
            continue
        resp.raise_for_status()  # error message contains the URL and status, never the key
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("unreachable")

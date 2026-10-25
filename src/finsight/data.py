"""Dataset loading (BEIR versions of FiQA and SciFact) and the unanswerable subset for E3.

Return types used across the project:
    corpus:  dict[doc_id, str]            (title + " " + text)
    queries: dict[qid, str]
    qrels:   dict[qid, dict[doc_id, int]]
"""
from __future__ import annotations

import random

DATA_DIR = "data"


def load_beir(name: str, split: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load corpus, the queries of `split`, and their qrels.

    name:  "fiqa" | "scifact";  split: "train" | "dev" | "test"  (SciFact has no dev split)
    Hint (verify in M2 and record what worked in wiki/dataset/<name>.md):
        datasets.load_dataset("BeIR/<name>", "corpus")   → columns _id, title, text
        datasets.load_dataset("BeIR/<name>", "queries")  → columns _id, text
        datasets.load_dataset("BeIR/<name>-qrels")       → columns query-id, corpus-id, score; split names may
                                                           be "validation" instead of "dev".
    Only return queries that appear in the qrels of `split`. Cache to DATA_DIR if loading is slow.

    Verified 2026-10-25: qrels ids are int64 on the Hub (corpus/query ids are strings) → cast with str().
    FiQA's dev split is called "validation". Hugging Face caches the download in ~/.cache/huggingface/.
    """
    from datasets import load_dataset  # imported here so the unit tests never need the library

    hf_split = "validation" if split == "dev" else split
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")[hf_split]

    corpus = {row["_id"]: (row["title"] + " " + row["text"]).strip() for row in corpus_ds}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid, doc_id = str(row["query-id"]), str(row["corpus-id"])
        qrels.setdefault(qid, {})[doc_id] = int(row["score"])

    queries = {row["_id"]: row["text"] for row in queries_ds if row["_id"] in qrels}
    check_ids(corpus, queries, qrels)
    return corpus, queries, qrels


def check_ids(corpus: dict[str, str], queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> None:
    """Raise ValueError unless every id is a str and every qrels id points to an existing query / doc.

    Why: if ids mismatch (e.g. int 1 vs str "1"), nothing matches and every metric silently becomes 0.
    """
    def fail(problem: str, bad: list) -> None:
        raise ValueError(f"{problem}: {len(bad)} id(s), e.g. {bad[:5]!r}")

    for label, ids in [("corpus", corpus), ("queries", queries), ("qrels queries", qrels),
                       ("qrels docs", [d for q in qrels.values() for d in q])]:
        bad = [i for i in ids if not isinstance(i, str)]
        if bad:
            fail(f"{label} ids must be str", bad)

    missing_docs = sorted({d for q in qrels.values() for d in q if d not in corpus})
    if missing_docs:
        fail("qrels doc ids not found in corpus", missing_docs)
    missing_queries = sorted(q for q in qrels if q not in queries)
    if missing_queries:
        fail("qrels query ids not found in queries", missing_queries)


def make_unanswerable(qrels: dict[str, dict[str, int]], corpus_ids: list[str], n: int,
                      seed: int = 42) -> tuple[list[str], set[str]]:
    """Pick `n` queries to make unanswerable by removing ALL of their relevant docs from the corpus.

    Returns (unanswerable_qids, removed_doc_ids). The caller re-indexes corpus_ids minus removed_doc_ids.
    The selected queries must still be checked by hand (top-5 retrieved docs must not answer them);
    record the verified list in wiki/dataset/unanswerable.md.
    """
    # 1. rng = random.Random(seed); sample n qids from qrels (sorted first for determinism)
    # 2. removed = union of their relevant doc ids
    # 3. return the sampled qids and `removed`
    raise NotImplementedError


__all__ = ["load_beir", "check_ids", "make_unanswerable"]

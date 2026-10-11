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
    """
    raise NotImplementedError


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


__all__ = ["load_beir", "make_unanswerable"]

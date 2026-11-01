"""E3 step 1 (M8): build the unanswerable set on FiQA dev and a sheet for the manual check.

Usage:
    python scripts/e3_build_unanswerable.py --n 50

1. make_unanswerable picks n dev queries and removes ALL their relevant docs from the corpus.
2. TorchKNN (exact, 384-d) indexes the reduced corpus; every dev query is searched once.
3. Writes (experiments/results/ is gitignored):
     e3_unanswerable/top_scores.json  top-1 similarity per dev query + its group, for the τ sweep
     e3_unanswerable/review.md        each unanswerable question with its top-5 docs, for the manual check
The verified list then goes to wiki/dataset/unanswerable.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from finsight import RANDOM_SEED  # noqa: E402
from finsight.data import load_beir, make_unanswerable  # noqa: E402
from finsight.embed import load_or_encode  # noqa: E402
from finsight.metrics import recall_at_k  # noqa: E402
from finsight.retrieval import TorchKNN  # noqa: E402

SNIPPET_CHARS = 400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    corpus, queries, qrels = load_beir("fiqa", "dev")
    all_ids = sorted(corpus)
    unans_qids, removed = make_unanswerable(qrels, all_ids, n=args.n, seed=RANDOM_SEED)

    # same caches (and row order) as run_experiment.py
    all_emb = load_or_encode(str(REPO / "data" / "fiqa_corpus_minilm.npy"), [corpus[d] for d in all_ids], ids=all_ids)
    qids = sorted(queries)
    q_emb = load_or_encode(str(REPO / "data" / "fiqa_dev_queries_minilm.npy"), [queries[q] for q in qids], ids=qids)

    keep = [i for i, d in enumerate(all_ids) if d not in removed]
    doc_ids = [all_ids[i] for i in keep]
    index = TorchKNN().fit(all_emb[keep])
    scores, idx = index.search(q_emb, 10)
    retrieved = {q: [doc_ids[j] for j in row] for q, row in zip(qids, idx)}

    unans = set(unans_qids)
    answerable = [q for q in qids if q not in unans]
    # sanity: removal must not hurt the answerable queries (their docs were never removed)
    r10 = recall_at_k(retrieved, {q: qrels[q] for q in answerable}, 10)
    leaked = [q for q in unans_qids if set(retrieved[q]) & set(qrels[q])]
    if leaked:
        raise RuntimeError(f"relevant docs still retrievable for {leaked[:5]}: removal failed")

    out = REPO / "experiments" / "results" / "e3_unanswerable"
    out.mkdir(parents=True, exist_ok=True)
    top = {q: {"top1": float(s[0]), "group": "unanswerable" if q in unans else "answerable"}
           for q, s in zip(qids, scores)}
    (out / "top_scores.json").write_text(json.dumps(
        {"n_unanswerable": len(unans_qids), "n_answerable": len(answerable), "removed_docs": sorted(removed),
         "n_docs_indexed": len(doc_ids), "recall@10_answerable": r10, "seed": RANDOM_SEED, "queries": top}, indent=1))

    lines = [f"# Manual check: {len(unans_qids)} unanswerable FiQA dev questions",
             "", "For each question: do any of the top-5 documents answer it? Mark OK (none answers) or DROP.", ""]
    for n, q in enumerate(unans_qids, 1):
        lines += [f"## {n}. qid {q} — top-1 similarity {top[q]['top1']:.3f}", "", f"**Q:** {queries[q]}", ""]
        for rank, d in enumerate(retrieved[q][:5], 1):
            text = corpus[d].replace("\n", " ")
            lines.append(f"{rank}. `[{d}]` {text[:SNIPPET_CHARS]}{'…' if len(text) > SNIPPET_CHARS else ''}")
        lines += ["", "Verdict: OK / DROP", ""]
    (out / "review.md").write_text("\n".join(lines))

    a = np.array([top[q]["top1"] for q in answerable])
    u = np.array([top[q]["top1"] for q in unans_qids])
    print(json.dumps({"n_unanswerable": len(unans_qids), "n_answerable": len(answerable),
                      "removed_docs": len(removed), "n_docs_indexed": len(doc_ids),
                      "recall@10_answerable": round(r10, 4),
                      "top1_answerable_mean": round(float(a.mean()), 4),
                      "top1_unanswerable_mean": round(float(u.mean()), 4),
                      "review_sheet": str((out / "review.md").relative_to(REPO))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

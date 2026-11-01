"""E3 step 2: answers for answerable FiQA dev questions at k ∈ {1, 3, 5, 10}, for manual scoring.

Usage:
    python scripts/e3_generate.py --run-id run_e3_gen_001 --n 20 --budget 150000

Same setting as the τ sweep: corpus minus the 116 docs removed by make_unanswerable(n=50), exact
TorchKNN at 384-d. Each question is retrieved ONCE (top-10); k = 1/3/5 reuse the first k of those docs.
Documents are cut to DOC_CHARS characters (~ what MiniLM's 256-word-piece window sees) to bound tokens.
Stops before a call once `--budget` total tokens have been used, so the 200K tokens/day cap is never hit.

Writes (experiments/results/ is gitignored):
    <run-id>/answers.jsonl   one line per (question, k): prompt docs, raw answer, token usage
    <run-id>/scoring.csv     sheet for the 1–3 correctness / groundedness rubric
    <run-id>/summary.json    settings + token totals
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from finsight import RANDOM_SEED  # noqa: E402
from finsight.data import load_beir, make_unanswerable  # noqa: E402
from finsight.embed import load_or_encode  # noqa: E402
from finsight.rag import LLM_MODEL, REASONING_EFFORT, build_prompt, generate_with_usage  # noqa: E402
from finsight.retrieval import TorchKNN  # noqa: E402

KS = (1, 3, 5, 10)
DOC_CHARS = 1000
N_UNANSWERABLE = 50  # must match scripts/e3_build_unanswerable.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run_e3_gen_001")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=150_000, help="stop before exceeding this many tokens")
    args = ap.parse_args()

    corpus, queries, qrels = load_beir("fiqa", "dev")
    all_ids = sorted(corpus)
    unans_qids, removed = make_unanswerable(qrels, all_ids, n=N_UNANSWERABLE, seed=RANDOM_SEED)
    answerable = sorted(q for q in queries if q not in set(unans_qids))
    picked = random.Random(RANDOM_SEED).sample(answerable, args.n)

    all_emb = load_or_encode(str(REPO / "data" / "fiqa_corpus_minilm.npy"), [corpus[d] for d in all_ids], ids=all_ids)
    qids = sorted(queries)
    q_emb = load_or_encode(str(REPO / "data" / "fiqa_dev_queries_minilm.npy"), [queries[q] for q in qids], ids=qids)
    keep = [i for i, d in enumerate(all_ids) if d not in removed]
    doc_ids = [all_ids[i] for i in keep]
    index = TorchKNN().fit(all_emb[keep])
    row = {q: i for i, q in enumerate(qids)}
    scores, idx = index.search(q_emb[[row[q] for q in picked]], max(KS))

    out = REPO / "experiments" / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    used, done, stopped = 0, 0, False
    with open(out / "answers.jsonl", "w") as f:
        for qi, q in enumerate(picked):
            top = [(doc_ids[j], float(s)) for j, s in zip(idx[qi], scores[qi])]
            for k in KS:
                if used >= args.budget:
                    stopped = True
                    break
                docs = [(d, corpus[d][:DOC_CHARS]) for d, _ in top[:k]]
                answer, usage = generate_with_usage(build_prompt(queries[q], docs), max_tokens=400, max_retries=8)
                used += usage.get("total_tokens", 0)
                done += 1
                f.write(json.dumps({"qid": q, "k": k, "question": queries[q],
                                    "docs": [{"id": d, "score": s} for d, s in top[:k]],
                                    "relevant_in_top_k": sorted(set(d for d, _ in top[:k]) & set(qrels[q])),
                                    "answer": answer, "usage": usage, "model": LLM_MODEL,
                                    "date": time.strftime("%Y-%m-%d")}) + "\n")
                f.flush()
                print(f"[{done:3d}/{args.n * len(KS)}] qid {q} k={k:2d} tokens {usage.get('total_tokens', 0):5d} "
                      f"total {used:7d}", flush=True)
            if stopped:
                print(f"STOP: token budget {args.budget} reached", flush=True)
                break

    with open(out / "answers.jsonl") as f, open(out / "scoring.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["qid", "k", "relevant_in_top_k", "question", "answer",
                    "correctness_1to3", "groundedness_1to3", "scorer", "notes"])
        for line in f:
            r = json.loads(line)
            w.writerow([r["qid"], r["k"], " ".join(r["relevant_in_top_k"]), r["question"], r["answer"], "", "", "", ""])

    summary = {"run_id": args.run_id, "n_questions": args.n, "ks": list(KS), "calls": done, "total_tokens": used,
               "budget": args.budget, "stopped_on_budget": stopped, "model": LLM_MODEL,
               "reasoning_effort": REASONING_EFFORT, "temperature": 0, "doc_chars": DOC_CHARS,
               "corpus_docs": len(doc_ids), "seed": RANDOM_SEED, "qids": picked}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "qids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

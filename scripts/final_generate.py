"""M12 final evaluation, generation part: config A answers for 100 FiQA TEST questions, for manual scoring.

LOCKED TEST SPLIT (hard rule 1). Run once, only when a member explicitly asks for the final evaluation:
    python scripts/final_generate.py --final

Protocol from wiki/decisions/final_configs.md: config A = MiniLM 384-d, exact TorchKNN over the FULL corpus
(no docs removed), k = 5, refusal threshold τ = 0.60 on the top-1 cosine, fixed LLM and rag.build_prompt.
100 questions sampled from the 648 test queries with seed 42. Documents cut to DOC_CHARS, as in e3_generate.py.

Every question goes to the LLM, also those with top-1 < τ, so that
  - the answer quality of all 100 is comparable with dev (E3 run 001 had no τ), and
  - the system result (τ refusal first, then LLM) is read off the `refused_by_tau` column.
Stops before a call once `--budget` total tokens have been used (Groq cap: 200K tokens/day).

Writes (experiments/results/ is gitignored):
    final_gen_a/answers.jsonl   one line per question: top-1 score, τ decision, docs, raw answer, token usage
    final_gen_a/scoring.csv     sheet for the 1–3 correctness / groundedness rubric (wiki/experiments/e3_rubric.md)
    final_gen_a/summary.json    settings, token totals, answer rate at τ
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

RUN_ID = "final_gen_a"
N_QUESTIONS = 100  # final_configs.md, test protocol step 2
FINAL_K = 5
FINAL_TAU = 0.60
DOC_CHARS = 1000   # same cut as scripts/e3_generate.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="required to touch the FiQA test split")
    ap.add_argument("--budget", type=int, default=150_000, help="stop before exceeding this many tokens")
    args = ap.parse_args()
    if not args.final:  # check before loading anything from the test split
        print("Refusing: FiQA test split is locked. Use --final only for the final evaluation.", file=sys.stderr)
        return 2

    from finsight.data import load_beir
    from finsight.embed import load_or_encode
    from finsight.rag import LLM_MODEL, REASONING_EFFORT, build_prompt, generate_with_usage, should_refuse
    from finsight.retrieval import TorchKNN

    out = REPO / "experiments" / "results" / RUN_ID
    if (out / "answers.jsonl").exists():  # the test set is used once: never overwrite a finished run
        print(f"Refusing: {out}/answers.jsonl already exists.", file=sys.stderr)
        return 4

    corpus, queries, qrels = load_beir("fiqa", "test")
    picked = random.Random(RANDOM_SEED).sample(sorted(queries), N_QUESTIONS)

    doc_ids = sorted(corpus)
    emb = load_or_encode(str(REPO / "data" / "fiqa_corpus_minilm.npy"), [corpus[d] for d in doc_ids], ids=doc_ids)
    qids = sorted(queries)
    q_emb = load_or_encode(str(REPO / "data" / "fiqa_test_queries_minilm.npy"), [queries[q] for q in qids], ids=qids)
    row = {q: i for i, q in enumerate(qids)}
    scores, idx = TorchKNN().fit(emb).search(q_emb[[row[q] for q in picked]], FINAL_K)

    out.mkdir(parents=True, exist_ok=True)
    used, done, n_refused, stopped = 0, 0, 0, False
    with open(out / "answers.jsonl", "w") as f:
        for qi, q in enumerate(picked):
            if used >= args.budget:
                stopped = True
                print(f"STOP: token budget {args.budget} reached", flush=True)
                break
            top = [(doc_ids[j], float(s)) for j, s in zip(idx[qi], scores[qi])]
            refused = should_refuse(top[0][1], FINAL_TAU)
            n_refused += refused
            docs = [(d, corpus[d][:DOC_CHARS]) for d, _ in top]
            answer, usage = generate_with_usage(build_prompt(queries[q], docs), max_tokens=400, max_retries=8)
            used += usage.get("total_tokens", 0)
            done += 1
            f.write(json.dumps({"qid": q, "k": FINAL_K, "question": queries[q],
                                "top1_score": top[0][1], "refused_by_tau": refused,
                                "docs": [{"id": d, "score": s} for d, s in top],
                                "relevant_in_top_k": sorted(set(d for d, _ in top) & set(qrels[q])),
                                "answer": answer, "usage": usage, "model": LLM_MODEL,
                                "date": time.strftime("%Y-%m-%d")}) + "\n")
            f.flush()
            print(f"[{done:3d}/{N_QUESTIONS}] top-1 {top[0][1]:.3f} {'τ-refused' if refused else 'answered '} "
                  f"tokens {usage.get('total_tokens', 0):5d} total {used:7d}", flush=True)

    with open(out / "answers.jsonl") as f, open(out / "scoring.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["qid", "k", "top1_score", "refused_by_tau", "relevant_in_top_k", "question", "answer",
                    "correctness_1to3", "groundedness_1to3", "scorer", "notes"])
        for line in f:
            r = json.loads(line)
            w.writerow([r["qid"], r["k"], f"{r['top1_score']:.3f}", r["refused_by_tau"],
                        " ".join(r["relevant_in_top_k"]), r["question"], r["answer"], "", "", "", ""])

    summary = {"run_id": RUN_ID, "split": "test", "n_questions": N_QUESTIONS, "k": FINAL_K, "tau": FINAL_TAU,
               "calls": done, "refused_by_tau": n_refused,
               "answer_rate_at_tau": (done - n_refused) / done if done else None,
               "total_tokens": used, "budget": args.budget, "stopped_on_budget": stopped,
               "model": LLM_MODEL, "reasoning_effort": REASONING_EFFORT, "temperature": 0,
               "doc_chars": DOC_CHARS, "corpus_docs": len(doc_ids), "seed": RANDOM_SEED, "qids": picked}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "qids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

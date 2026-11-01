"""E3 step 4: re-tune the refusal threshold τ for one final retrieval config (dev only).

Usage:
    python scripts/e3_tau_config.py --name config_a                                   # 384-d, exact
    python scripts/e3_tau_config.py --name config_b --pca-dim 128 --method faiss_ivf --nlist 240 --nprobe 16

Top-1 cosine depends on the embedding space and the index, so τ = 0.60 (384-d, exact) does not carry
over to 128-d PCA + IVF. Same setting as scripts/e3_build_unanswerable.py: corpus minus the 116 removed
docs; PCA is fitted on that indexed corpus only and queries get the same transform (Hard rule 2).
Unanswerable side = the 10 manually verified qids (verified_qids in e3_unanswerable/tau_curve.json).
No LLM calls.

Writes (experiments/results/ is gitignored):
    e3_tau_<name>/tau_curve.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from e3_analyze import TAUS, auc  # noqa: E402
from finsight import RANDOM_SEED  # noqa: E402
from finsight.data import load_beir, make_unanswerable  # noqa: E402
from finsight.embed import load_or_encode  # noqa: E402
from finsight.metrics import recall_at_k  # noqa: E402
from finsight.pca import PCAReducer  # noqa: E402
from finsight.rag import refusal_curve  # noqa: E402
from finsight.retrieval import build_index  # noqa: E402

N_UNANSWERABLE = 50  # must match scripts/e3_build_unanswerable.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--method", default="torch_knn")
    ap.add_argument("--nlist", type=int, default=240)
    ap.add_argument("--nprobe", type=int, default=16)
    args = ap.parse_args()

    corpus, queries, qrels = load_beir("fiqa", "dev")
    all_ids = sorted(corpus)
    unans_qids, removed = make_unanswerable(qrels, all_ids, n=N_UNANSWERABLE, seed=RANDOM_SEED)
    verified = json.load(open(REPO / "experiments" / "results" / "e3_unanswerable" / "tau_curve.json"))["verified_qids"]

    all_emb = load_or_encode(str(REPO / "data" / "fiqa_corpus_minilm.npy"), [corpus[d] for d in all_ids], ids=all_ids)
    qids = sorted(queries)
    q_emb = load_or_encode(str(REPO / "data" / "fiqa_dev_queries_minilm.npy"), [queries[q] for q in qids], ids=qids)
    keep = [i for i, d in enumerate(all_ids) if d not in removed]
    doc_ids = [all_ids[i] for i in keep]
    doc_emb = all_emb[keep]
    if args.pca_dim:
        pca = PCAReducer(args.pca_dim).fit(doc_emb)  # indexed corpus only, never queries
        doc_emb, q_emb = pca.transform(doc_emb), pca.transform(q_emb)

    params = {"nlist": args.nlist, "nprobe": args.nprobe} if args.method == "faiss_ivf" else {}
    index = build_index(args.method, **params).fit(doc_emb)
    scores, idx = index.search(q_emb, 10)
    retrieved = {q: [doc_ids[j] for j in row if j >= 0] for q, row in zip(qids, idx)}  # IVF pads with -1
    top1 = {q: float(s[0]) for q, s in zip(qids, scores)}

    unans = set(unans_qids)
    answerable = [q for q in qids if q not in unans]
    a = np.array([top1[q] for q in answerable])
    u = np.array([top1[q] for q in verified])
    curve = refusal_curve(a, u, TAUS)
    for c in curve:
        c["balanced"] = round((c["answer_rate"] + c["correct_refusal_rate"]) / 2, 4)
    best = max(curve, key=lambda c: (c["balanced"], -c["tau"]))  # ties → smallest τ (answers more)

    out = REPO / "experiments" / "results" / f"e3_tau_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "pca_dim": args.pca_dim or 384, "method": args.method, "method_params": params,
              "n_docs_indexed": len(doc_ids), "n_answerable": len(a), "n_unanswerable_verified": len(u),
              "recall@10_answerable": round(recall_at_k(retrieved, {q: qrels[q] for q in answerable}, 10), 4),
              "top1_median": {"answerable": round(float(np.median(a)), 4),
                              "unanswerable_verified": round(float(np.median(u)), 4)},
              "auc_verified": round(auc(a, u), 3), "best_balanced": best, "seed": RANDOM_SEED, "curve": curve}
    (out / "tau_curve.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

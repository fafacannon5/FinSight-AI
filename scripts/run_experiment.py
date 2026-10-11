"""Run one retrieval experiment from a YAML config and write metrics.json.

Usage:
    python scripts/run_experiment.py experiments/configs/run_e1_001.yaml
    python scripts/run_experiment.py experiments/configs/final_e1.yaml --final   # test split (locked)

Config keys (see experiments/configs/TEMPLATE_retrieval.yaml):
    run_id, experiment, hypothesis, dataset, split, method, method_params, pca_dim, k,
    corpus_size, latency_queries, random_seed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from finsight import RANDOM_SEED  # noqa: E402
from finsight.data import load_beir  # noqa: E402
from finsight.embed import load_or_encode  # noqa: E402
from finsight.metrics import evaluate_all  # noqa: E402
from finsight.retrieval import build_index  # noqa: E402
from finsight.timing import measure_latency, timed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--final", action="store_true", help="required to touch the FiQA test split")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["dataset"] == "fiqa" and cfg["split"] == "test" and not args.final:
        print("Refusing: FiQA test split is locked. Use --final only for the final evaluation.", file=sys.stderr)
        return 2
    seed = cfg.get("random_seed", RANDOM_SEED)
    rng = np.random.default_rng(seed)

    corpus, queries, qrels = load_beir(cfg["dataset"], cfg["split"])
    doc_ids = sorted(corpus)
    if cfg.get("corpus_size"):  # E1-scale: keep all relevant docs, fill the rest at random
        relevant = {d for q in qrels.values() for d in q if d in corpus}
        others = [d for d in doc_ids if d not in relevant]
        n_extra = max(0, cfg["corpus_size"] - len(relevant))
        doc_ids = sorted(relevant | set(rng.choice(others, size=min(n_extra, len(others)), replace=False)))

    cache = REPO / "data" / f"{cfg['dataset']}_corpus_minilm.npy"
    all_ids = sorted(corpus)
    all_emb = load_or_encode(str(cache), [corpus[d] for d in all_ids])
    pos = {d: i for i, d in enumerate(all_ids)}
    doc_emb = all_emb[[pos[d] for d in doc_ids]]

    qids = sorted(queries)
    q_emb = load_or_encode(str(REPO / "data" / f"{cfg['dataset']}_{cfg['split']}_queries_minilm.npy"),
                           [queries[q] for q in qids])

    if cfg.get("pca_dim"):
        from finsight.pca import PCAReducer
        pca = PCAReducer(cfg["pca_dim"]).fit(doc_emb)   # fit on corpus only
        doc_emb, q_emb = pca.transform(doc_emb), pca.transform(q_emb)

    index = build_index(cfg["method"], **(cfg.get("method_params") or {}))
    _, build_s = timed(index.fit, np.ascontiguousarray(doc_emb, dtype=np.float32))
    q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
    k = cfg.get("k", 10)
    _, idx = index.search(q_emb, k)
    retrieved = {q: [doc_ids[j] for j in row if j >= 0] for q, row in zip(qids, idx)}

    metrics = {"run_id": cfg["run_id"], "experiment": cfg["experiment"], "split": cfg["split"],
               "n_docs": len(doc_ids), "n_queries": len(qids), "dim": int(doc_emb.shape[1])}
    metrics.update(evaluate_all(retrieved, qrels))
    metrics.update(measure_latency(index, q_emb, k, max_queries=cfg.get("latency_queries", 200)))
    metrics["build_time_s"] = build_s
    metrics["index_memory_mb"] = index.memory_mb()
    metrics["config"] = cfg

    out = REPO / "experiments" / "results" / cfg["run_id"]
    os.makedirs(out, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: v for k_, v in metrics.items() if k_ != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

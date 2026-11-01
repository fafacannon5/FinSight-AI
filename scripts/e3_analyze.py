"""E3 step 3: analyse the manual scores (answer quality vs. k) and the τ refusal curve.

Usage:
    python scripts/e3_analyze.py

Inputs (experiments/results/ is gitignored):
    run_e3_gen_001/scoring_completed.csv   scores filled in by hand (rubric: wiki/experiments/e3_rubric.md)
    run_e3_gen_001/answers.jsonl           the raw answers (source of relevant_in_top_k)
    e3_unanswerable/review_completed.md    manual OK / DROP verdicts for the 50 unanswerable candidates
    e3_unanswerable/top_scores.json        top-1 similarity per dev query

Quality means use ONLY answered rows: rows whose notes are `refused` / `refused-wrongly` are counted
separately as refusals. The τ curve uses only the candidates verified as OK.

Writes:
    run_e3_gen_001/analysis.json
    e3_unanswerable/tau_curve.json and tau_curve.png
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from finsight.rag import refusal_curve  # noqa: E402

RESULTS = REPO / "experiments" / "results"
GEN = RESULTS / "run_e3_gen_001"
UNANS = RESULTS / "e3_unanswerable"
REFUSAL_NOTES = ("refused", "refused-wrongly")
TAUS = np.round(np.arange(0.30, 0.901, 0.01), 2)


def parse_verdicts(text: str) -> dict[str, str]:
    """qid → "OK" | "DROP" from the review sheet (each '## n. qid X' block ends with 'Verdict: ...')."""
    verdicts, qid = {}, None
    for line in text.splitlines():
        m = re.match(r"## \d+\. qid (\S+)", line)
        if m:
            qid = m.group(1)
        m = re.match(r"Verdict:\s*(OK|DROP)\s*$", line.strip())
        if m and qid is not None:
            verdicts[qid] = m.group(1)
            qid = None
    return verdicts


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random answerable > score of a random unanswerable); ties count 1/2."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def mean_or_none(xs: list[float]) -> float | None:
    return round(float(np.mean(xs)), 3) if xs else None


def quality_by_k(rows: list[dict]) -> dict:
    out = {}
    for k in sorted({int(r["k"]) for r in rows}):
        at_k = [r for r in rows if int(r["k"]) == k]
        answered = [r for r in at_k if r["notes"].strip() not in REFUSAL_NOTES]
        notes = [r["notes"].strip() for r in at_k]
        hit = [r for r in answered if r["relevant_in_top_k"]]
        miss = [r for r in answered if not r["relevant_in_top_k"]]
        out[k] = {
            "n": len(at_k), "answered": len(answered),
            "refused": notes.count("refused"), "refused_wrongly": notes.count("refused-wrongly"),
            "refusal_rate": round((len(at_k) - len(answered)) / len(at_k), 3),
            "correctness": mean_or_none([float(r["correctness_1to3"]) for r in answered]),
            "groundedness": mean_or_none([float(r["groundedness_1to3"]) for r in answered]),
            # split by whether a labelled doc was in the first k (answered rows only)
            "correctness_label_in_top_k": mean_or_none([float(r["correctness_1to3"]) for r in hit]),
            "correctness_no_label_in_top_k": mean_or_none([float(r["correctness_1to3"]) for r in miss]),
            "answered_label_in_top_k": len(hit),
        }
    return out


def main() -> int:
    rows = list(csv.DictReader(open(GEN / "scoring_completed.csv", newline="")))
    raw = {(str(r["qid"]), str(r["k"])): r for r in map(json.loads, open(GEN / "answers.jsonl"))}
    for r in rows:  # the completed sheet lost this column; take it from the raw answers
        r["relevant_in_top_k"] = raw[(r["qid"], r["k"])]["relevant_in_top_k"]
    missing = [(r["qid"], r["k"]) for r in rows if r["notes"].strip() not in REFUSAL_NOTES
               and not (r["correctness_1to3"] and r["groundedness_1to3"])]
    if missing:
        raise ValueError(f"answered rows without both scores: {missing}")

    scorers = sorted({r["scorer"] for r in rows})
    analysis = {"run_id": "run_e3_gen_001", "n_rows": len(rows), "scorers": scorers,
                "quality_by_k": quality_by_k(rows),
                "agreement": None if len(scorers) < 2 else "TODO: double-scored rows present"}
    (GEN / "analysis.json").write_text(json.dumps(analysis, indent=1))

    verdicts = parse_verdicts((UNANS / "review_completed.md").read_text())
    top = json.load(open(UNANS / "top_scores.json"))
    unans = [q for q, v in top["queries"].items() if v["group"] == "unanswerable"]
    if set(verdicts) != set(unans):
        raise ValueError("review sheet qids do not match top_scores.json")
    ok = sorted(q for q in unans if verdicts[q] == "OK")
    a = np.array([v["top1"] for v in top["queries"].values() if v["group"] == "answerable"])
    u_ok = np.array([top["queries"][q]["top1"] for q in ok])
    u_all = np.array([top["queries"][q]["top1"] for q in unans])

    curve = refusal_curve(a, u_ok, TAUS)
    for c in curve:
        c["balanced"] = round((c["answer_rate"] + c["correct_refusal_rate"]) / 2, 4)
    best = max(curve, key=lambda c: (c["balanced"], -c["tau"]))  # ties → smallest τ (answers more)
    tau = {"n_answerable": len(a), "n_unanswerable_verified": len(ok), "n_dropped": len(unans) - len(ok),
           "verified_qids": ok,
           "top1_median": {"answerable": round(float(np.median(a)), 4),
                           "unanswerable_verified": round(float(np.median(u_ok)), 4),
                           "unanswerable_all50": round(float(np.median(u_all)), 4)},
           "auc_verified": round(auc(a, u_ok), 3), "auc_all50": round(auc(a, u_all), 3),
           "best_balanced": best, "curve": curve}
    (UNANS / "tau_curve.json").write_text(json.dumps(tau, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(TAUS, [c["answer_rate"] for c in curve], label=f"answer rate (answerable, n={len(a)})")
    ax.plot(TAUS, [c["correct_refusal_rate"] for c in curve],
            label=f"correct refusal rate (verified unanswerable, n={len(ok)})")
    ax.axvline(best["tau"], color="grey", ls="--", lw=1)
    ax.set_xlabel("τ (refuse if top-1 cosine < τ)")
    ax.set_ylabel("rate")
    ax.set_title("FiQA dev — refusal threshold")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(UNANS / "tau_curve.png", dpi=150)

    print(json.dumps({"quality_by_k": analysis["quality_by_k"], "scorers": scorers,
                      **{k: v for k, v in tau.items() if k not in ("curve", "verified_qids")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

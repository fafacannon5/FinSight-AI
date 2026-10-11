# Repository layout

Parent: [AGENTS.md](../AGENTS.md)

```
FinSight-AI/
├── AGENTS.md                 ← agent entry (Cursor / Codex read it directly)
├── CLAUDE.md                 ← imports AGENTS.md for Claude Code
├── README.md                 ← [DELIVERABLE] project overview + how to run
├── requirements.txt
├── agents/                   ← workflows, conventions (internal)
├── raw_sources/              ← immutable inputs: proposal, syllabus notes, papers
├── wiki/                     ← agent-maintained knowledge base (internal)
│   ├── overview.md           ← living synthesis — read first
│   ├── index.md · log.md
│   ├── dataset/ · experiments/ · literature/ · decisions/
├── experiments/              ← agent workspace (internal)
│   ├── configs/              ← one YAML per run (committed)
│   └── results/              ← metrics.json per run (gitignored)
├── src/finsight/             ← [DELIVERABLE] importable library code
│   ├── data.py               ← load BEIR datasets, unanswerable set
│   ├── embed.py              ← encode + cache embeddings
│   ├── retrieval.py          ← TorchKNN, FaissFlat, FaissIVF (+ bonus KMeansIndex)
│   ├── pca.py                ← PCA from scratch (torch)
│   ├── metrics.py            ← Recall@k, MRR@k, nDCG@k
│   ├── timing.py             ← latency + memory measurement
│   └── rag.py                ← refusal rule, prompt, generation, refusal curve
├── tests/                    ← [DELIVERABLE] pytest unit tests (synthetic data, no downloads)
├── scripts/run_experiment.py ← runs one YAML config → metrics.json
├── notebooks/                ← [DELIVERABLE] E1/E2/E3 analysis notebooks (figures for report)
├── app/streamlit_app.py      ← [DELIVERABLE] demo chatbot
├── data/                     ← downloaded data + embedding caches (gitignored)
└── report/                   ← [DELIVERABLE] figures/, tables/, section_drafts/
```

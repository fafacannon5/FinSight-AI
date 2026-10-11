# Tools

Parent: [AGENTS.md](../../AGENTS.md)

| Tool | Purpose | Invocation |
|---|---|---|
| `scripts/run_experiment.py` | Run one YAML config, write `experiments/results/<run_id>/metrics.json` | `python scripts/run_experiment.py experiments/configs/run_e1_001.yaml [--final]` |
| `pytest` | Unit tests on synthetic data | `pytest -q` |
| `streamlit` | Demo app | `streamlit run app/streamlit_app.py` |

Add a row here whenever a new script is added.

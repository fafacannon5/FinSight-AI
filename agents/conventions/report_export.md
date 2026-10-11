# Report export

Parent: [AGENTS.md](../../AGENTS.md)

- Figures → `report/figures/<exp>_<what>.pdf` (PNG ≥ 300 dpi as fallback) + `<file>.caption.txt`.
- Tables → `report/tables/<what>.tex` (bare `tabular`, ready for `\input{}`) or `.md` if the report is not LaTeX.
- Every figure/table is produced by a notebook in `notebooks/` so it can be regenerated.

Report section mapping (adjust when the official CS 541 template arrives):

| Source | Report section |
|---|---|
| `wiki/overview.md` problem + RQs | Introduction |
| `wiki/literature/` | Related work |
| `wiki/dataset/` | Dataset |
| `src/finsight/` design | Method / System design |
| `wiki/experiments/*_summary.md` + notebooks | Experiments & Results (E1, E2, E3) |
| `AI_USAGE.md` | Use of AI tools |

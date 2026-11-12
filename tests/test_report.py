"""Report helpers: Markdown tables for report/tables/."""
from finsight.report import latex_table, markdown_table, save_table


def test_markdown_table_layout():
    text = markdown_table([{"Dim": 384, "Recall@10": "0.4665"}, {"Dim": 128, "Recall@10": "0.4412"}])
    assert text.splitlines() == ["| Dim | Recall@10 |", "|---|---|", "| 384 | 0.4665 |", "| 128 | 0.4412 |"]


def test_save_table_writes_file(tmp_path):
    text = save_table([{"k": 5}], "demo", out_dir=tmp_path)
    assert (tmp_path / "demo.md").read_text() == text


def test_latex_table_escapes_and_converts_symbols():
    text = latex_table([{"Config": "A — 384-d", "τ": "0.60", "Speed-up": "2.6×", "run": "run_e2_005"}])
    lines = text.splitlines()
    assert lines[0] == r"\begin{tabular}{llll}"
    assert r"$\tau$" in lines[2] and r"2.6$\times$" in text and r"run\_e2\_005" in text
    assert lines[-1] == r"\end{tabular}"


def test_save_table_writes_tex_too(tmp_path):
    save_table([{"k": 5}], "demo", out_dir=tmp_path)
    assert (tmp_path / "demo.tex").read_text().startswith(r"\begin{tabular}")

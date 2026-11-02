"""Report helpers: Markdown tables for report/tables/."""
from finsight.report import markdown_table, save_table


def test_markdown_table_layout():
    text = markdown_table([{"Dim": 384, "Recall@10": "0.4665"}, {"Dim": 128, "Recall@10": "0.4412"}])
    assert text.splitlines() == ["| Dim | Recall@10 |", "|---|---|", "| 384 | 0.4665 |", "| 128 | 0.4412 |"]


def test_save_table_writes_file(tmp_path):
    text = save_table([{"k": 5}], "demo", out_dir=tmp_path)
    assert (tmp_path / "demo.md").read_text() == text

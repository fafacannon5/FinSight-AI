"""Report export helpers: one look for every figure, and Markdown + LaTeX tables (notebooks/01–05).

Colors: the first three slots of a colorblind-checked categorical palette (blue, orange, aqua). Aqua is
below 3:1 contrast on white, so every series also gets a direct label or a distinct marker / line style.
Figures are saved as vector PDF + 300-dpi PNG with a caption file (agents/conventions/report_export.md).
"""
from __future__ import annotations

from pathlib import Path

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"   # text, secondary text, gridlines
FIGURES, TABLES = Path("../report/figures"), Path("../report/tables")  # relative to notebooks/


def apply_style() -> None:
    """Set matplotlib defaults: 2px lines, light horizontal grid, no top/right spines."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType so text stays selectable in the PDF
    })


def save_figure(fig, name: str, caption: str, out_dir: str | Path = FIGURES) -> None:
    """Write <name>.pdf, <name>.png and <name>.caption.txt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    (out / f"{name}.caption.txt").write_text(caption.strip() + "\n")


def markdown_table(rows: list[dict]) -> str:
    """[{col: value, ...}, ...] → GitHub Markdown table; columns in the order of the first row."""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# pdflatex with \usepackage[utf8]{inputenc} cannot typeset these characters directly
_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "τ": r"$\tau$", "Δ": r"$\Delta$", "×": r"$\times$", "·": r"$\cdot$",
                "−": r"$-$", "≈": r"$\approx$", "≥": r"$\geq$", "…": r"\ldots{}", "—": "---", "–": "--"}


def tex_escape(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in text)


def latex_table(rows: list[dict]) -> str:
    """[{col: value, ...}, ...] → bare tabular for \\input{} (no float, so the report sets caption / size)."""
    cols = list(rows[0])
    row = lambda cells: " & ".join(tex_escape(str(c)) for c in cells) + r" \\"
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline", row(cols), r"\hline"]
    lines += [row(r[c] for c in cols) for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def save_table(rows: list[dict], name: str, out_dir: str | Path = TABLES) -> str:
    """Write <name>.md and <name>.tex; return the Markdown text (so the notebook can display it)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = markdown_table(rows)
    (out / f"{name}.md").write_text(text)
    (out / f"{name}.tex").write_text(latex_table(rows))
    return text

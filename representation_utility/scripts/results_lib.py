"""Helpers to save results tables and figures for the write-up (results/)."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
NUMBERS = RESULTS / "NUMBERS.md"


def save_table(df: pd.DataFrame, name: str, float_fmt: str = "%.3f", index: bool = False):
    """Write results/tables/<name>.csv and a LaTeX twin."""
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)
    df.to_latex(TABLES / f"{name}.tex", index=index, float_format=float_fmt, escape=True)
    print(f"saved results/tables/{name}.csv (+ .tex)  [{len(df)} rows]")


def save_fig(fig, name: str):
    """Write results/figures/<name>.png and .pdf."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"saved results/figures/{name}.png (+ .pdf)")


def add_number(key: str, value: str):
    """Append a 'key: value' line to results/NUMBERS.md (prose-ready values)."""
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    lines = NUMBERS.read_text().splitlines() if NUMBERS.exists() else ["# Key numbers", ""]
    lines = [ln for ln in lines if not ln.startswith(f"- **{key}**:")]
    lines.append(f"- **{key}**: {value}")
    NUMBERS.write_text("\n".join(lines) + "\n")
    print(f"NUMBERS.md <- {key}: {value}")

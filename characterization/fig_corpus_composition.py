"""Corpus composition figure for the dataset-level section: (a) subjects per
dataset, (b) subjects per disease domain, both without UKBioBank. Counts are
read from the appendix dataset table in the Dissertation repo."""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

DISS = Path("/Volumes/KHUE1TB/projects/Dissertation")
TABLE = DISS / "sections/appendix/main.tex"
OUT = DISS / "sections/datasets/figures/corpus_composition.pdf"

INK, MUTED, GRID = "#222222", "#666666", "#e5e5e5"
BAR = "#2a78d6"


def load_table() -> pd.DataFrame:
    body = TABLE.read_text().split("\\endlastfoot")[1].split("\\end{longtable}")[0]
    rows = []
    for line in body.splitlines():
        parts = [p.strip() for p in line.rstrip("\\ ").split("&")]
        if len(parts) != 4:
            continue
        name = re.sub(r"\\cite\{[^}]*\}", "", parts[0]).strip().strip("{}")
        rows.append((name, parts[2].split(",")[0].strip(), int(parts[3].replace(",", ""))))
    return pd.DataFrame(rows, columns=["dataset", "domain", "subjects"])


def style(ax):
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_scale(ax, df: pd.DataFrame):
    d = df.sort_values("subjects", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    ax.scatter(x, d.subjects, s=16, color=BAR, zorder=3, linewidths=0)
    ax.set_yscale("log")
    ax.set_ylim(8, 5000)
    ax.set_xlim(-1.5, len(d) + 0.5)

    med = d.subjects.median()
    for val, lab in ((1000, "1,000"), (med, f"median = {med:.0f}")):
        ax.axhline(val, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(len(d) + 0.3, val * 1.1, lab, ha="right", va="bottom", fontsize=7.5, color=MUTED)

    notes = {"ADNI": ((6, 0), "left"), "HBN-SSI": ((-6, 0), "right")}
    for i, r in d[d.dataset.isin(notes)].iterrows():
        off, ha = notes[r.dataset]
        ax.annotate(f"{r.dataset} ({r.subjects:,})", (i, r.subjects), xytext=off,
                    textcoords="offset points", ha=ha, va="center", fontsize=7.5, color=INK)
    ax.set_xticks([])
    ax.set_xlabel("Datasets, sorted by size", fontsize=8.5)
    ax.set_ylabel("Subjects (log scale)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    style(ax)


def panel_domains(ax, df: pd.DataFrame):
    g = (df.groupby("domain")
         .agg(subjects=("subjects", "sum"), n=("dataset", "size"))
         .sort_values("subjects"))
    y = np.arange(len(g))
    ax.barh(y, g.subjects, height=0.62, color=BAR, zorder=2)
    for yi, (s, n) in enumerate(zip(g.subjects, g.n)):
        ax.text(s + 120, yi, f"{s:,}  ({n} dataset{'s' if n > 1 else ''})",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, g.index, fontsize=8)
    ax.set_xlim(0, g.subjects.max() * 1.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Subjects", fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    style(ax)


def main():
    df = load_table()
    df = df[df.dataset != "UKBioBank"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    panel_scale(a, df)
    panel_domains(b, df)
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        ax.set_title(lab, loc="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

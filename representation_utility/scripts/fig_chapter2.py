"""Chapter 2 figures: per-encoder cross-dataset transfer (with controls) and
dataset identification vs transfer. Reads existing tables only; writes to
results/figures/ and copies the PDFs to the Dissertation figures folder."""
from __future__ import annotations
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_lib import FIGURES, TABLES, save_fig

DISS_FIG = Path("/Volumes/KHUE1TB/projects/Dissertation/sections/datasets/figures")

OLD, BRAIN, BROAD = "#8a8a85", "#eb6834", "#2a78d6"
ENCODERS = [  # (key, label, colour, is_control)
    ("handcrafted", "Image statistics", OLD, False),
    ("untrained", "Untrained CNN", OLD, False),
    ("medical", "MedicalNet", OLD, False),
    ("selfsup", "SwinUNETR (SSL)", OLD, False),
    ("brainiac", "BrainIAC", BRAIN, False),
    ("brainfm", "BrainFM", BRAIN, False),
    ("sammed3d", "SAM-Med3D", BROAD, False),
    ("3dino", "3DINO", BROAD, False),
    ("3dino_rand", "3DINO, random weights", BROAD, True),
    ("3dino_z", "3DINO, z-score input", BROAD, True),
    ("medical_pct", "MedicalNet, percentile input", OLD, True),
]


def multi_modality_bacc() -> pd.DataFrame:
    d = pd.read_csv(TABLES / "lodo_transfer_fm.csv")
    d = d[d.n_modalities > 1]
    return d.pivot(index="dataset", columns="representation", values="balanced_acc")


def fig_transfer(pm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    rows = ENCODERS[::-1]
    rng = np.random.default_rng(0)
    for y, (key, label, col, ctrl) in enumerate(rows):
        vals = pm[key].to_numpy()
        ax.barh(y, vals.mean(), height=0.62, color=col if not ctrl else "white",
                edgecolor=col, hatch="////" if ctrl else None, linewidth=1.0, zorder=2)
        ax.scatter(vals, y + rng.uniform(-0.18, 0.18, len(vals)), s=9, color="#333333",
                   zorder=3, linewidths=0)
        ax.text(1.02, y, f"{vals.mean():.2f}", va="center", fontsize=8, color="#222222")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
    ax.axhline(2.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.text(0.005, 2.62, "controls", ha="left", va="bottom", fontsize=7.5, color="#555555")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Balanced accuracy on the unseen dataset (10 multi-modality datasets)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def fig_identify_vs_transfer(pm: pd.DataFrame):
    dec = pd.read_csv(TABLES / "decodability_fm.csv")
    dec = dec[dec.factor == "dataset"].set_index("representation").balanced_acc
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    offsets = {"untrained": (4, -9), "medical": (4, -9), "handcrafted": (-4, 4),
               "selfsup": (-4, 4), "brainfm": (-4, 5), "brainiac": (-4, 6), "3dino_rand": (0, -13), "sammed3d": (-4, 5), "3dino": (-4, 4)}
    for key, label, col, ctrl in ENCODERS:
        if ctrl and key != "3dino_rand":
            continue
        x, y = dec[key], pm[key].mean()
        ax.scatter(x, y, s=38, color="white" if ctrl else col, edgecolor=col,
                   linewidths=1.4, zorder=3)
        dx, dy = offsets.get(key, (5, -3))
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"), fontsize=7.5, color="#222222")
    ax.set_xlim(0.75, 0.97)
    ax.set_ylim(0.38, 0.78)
    ax.set_xlabel("Source-dataset identification (balanced accuracy)", fontsize=8.5)
    ax.set_ylabel("Cross-dataset transfer (balanced accuracy)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(color="#e5e5e5", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def main():
    pm = multi_modality_bacc()
    for name, fig in (("ch2_fm_transfer", fig_transfer(pm)),
                      ("ch2_identify_vs_transfer", fig_identify_vs_transfer(pm))):
        save_fig(fig, name)
        plt.close(fig)
        DISS_FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIGURES / f"{name}.pdf", DISS_FIG / f"{name}.pdf")
        print(f"copied {name}.pdf -> {DISS_FIG}")


if __name__ == "__main__":
    main()

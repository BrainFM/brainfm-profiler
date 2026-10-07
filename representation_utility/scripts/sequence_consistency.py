"""Cross-dataset consistency of sequence directions, and whether it explains LODO transfer.

Inside every leave-one-dataset-out split (feature space fitted on the 16 training datasets):
  consistency        cosine between the held-out dataset's sequence offsets and the training datasets' mean offsets
  pool_consistency   the same score among the training datasets only (no target data)
  centre_free        probe after subtracting each dataset's feature mean (label-free, usable at test time)
  centre_oracle      probe after subtracting each dataset's mean of sequence means (uses held-out labels)

Writes results/tables/sequence_consistency{,_summary}.csv and results/figures/fig_sequence_consistency.{png,pdf}.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_representations import load_rep  # noqa: E402
from cross_dataset_transfer import MODS, clf, fit_tr  # noqa: E402
from results_lib import add_number, save_fig, save_table  # noqa: E402

OLD, BRAIN, BROAD = "#8a8a85", "#eb6834", "#2a78d6"
ENCODERS = [  # (key, label, colour, is_control)
    ("handcrafted", "Handcrafted", OLD, False),
    ("untrained", "Untrained CNN", OLD, False),
    ("medical", "MedicalNet", OLD, False),
    ("selfsup", "SwinUNETR", OLD, False),
    ("brainiac", "BrainIAC", BRAIN, False),
    ("brainfm", "BrainFM", BRAIN, False),
    ("sammed3d", "SAM-Med3D", BROAD, False),
    ("3dino", "3DINO", BROAD, False),
    ("3dino_rand", "3DINO, random weights", BROAD, True),
    ("3dino_z", "3DINO, z-score input", BROAD, True),
    ("medical_pct", "MedicalNet, percentile input", OLD, True),
]


def offsets(X, y):
    """Sequence means minus their average (the dataset's class-balanced centre)."""
    mu = {m: X[y == m].mean(0) for m in np.unique(y)}
    c = np.mean(list(mu.values()), 0)
    return {m: v - c for m, v in mu.items()}, c


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def agreement(off, ref_offs):
    """Mean cosine between one dataset's offsets and the mean offset of the reference datasets."""
    sims = []
    for m, v in off.items():
        refs = [o[m] for o in ref_offs if m in o]
        if refs:
            sims.append(cos(v, np.mean(refs, 0)))
    return float(np.mean(sims)) if sims else np.nan


def probe_bacc(Xtr, ytr, Xte, yte):
    return balanced_accuracy_score(yte, clf().fit(Xtr, ytr).predict(Xte))


def run(rep):
    d, cols = load_rep(rep)
    d = d[d["modality"].isin(MODS)]
    X, y, ds = d[cols].to_numpy(float), d["modality"].to_numpy(), d["dataset"].to_numpy()
    multi = [s for s in sorted(set(ds)) if len(set(y[ds == s])) > 1]
    rows = []
    for held in multi:
        tr, te = ds != held, ds == held
        Xtr, Xte = fit_tr(X[tr], X[te])
        ytr, yte, dtr = y[tr], y[te], ds[tr]

        tr_off = {s: offsets(Xtr[dtr == s], ytr[dtr == s]) for s in multi if s != held}
        held_off, held_c = offsets(Xte, yte)
        pool = np.mean([agreement(tr_off[s][0], [tr_off[e][0] for e in tr_off if e != s]) for s in tr_off])

        Ctr_free, Ctr_orac = Xtr.copy(), Xtr.copy()
        for s in set(dtr):
            m = dtr == s
            Ctr_free[m] -= Xtr[m].mean(0)
            Ctr_orac[m] -= tr_off[s][1] if s in tr_off else Xtr[m].mean(0)  # single-sequence datasets: plain mean

        rows.append(dict(
            representation=rep, dataset=held,
            consistency=agreement(held_off, [o[0] for o in tr_off.values()]),
            pool_consistency=pool,
            before=probe_bacc(Xtr, ytr, Xte, yte),
            centre_free=probe_bacc(Ctr_free, ytr, Xte - Xte.mean(0), yte),
            centre_oracle=probe_bacc(Ctr_orac, ytr, Xte - held_c, yte),
        ))
    return pd.DataFrame(rows)


def summarize(df):
    rows = []
    for rep, g in df.groupby("representation", sort=False):
        r = dict(representation=rep)
        for k in ["consistency", "pool_consistency", "before", "centre_free", "centre_oracle"]:
            r[k] = g[k].mean()
        for k in ["centre_free", "centre_oracle"]:
            r[f"p_{k}"] = wilcoxon(g[k], g["before"]).pvalue
        r["rho_within"] = spearmanr(g["consistency"], g["before"])[0]
        rows.append(r)
    return pd.DataFrame(rows)


def figure(df, s):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for key, label, col, ctrl in ENCODERS:
        g = df[df.representation == key]
        ax1.scatter(g.consistency, g.before, s=18, alpha=0.8, zorder=3,
                    facecolor="none" if ctrl else col, edgecolor=col, lw=0.8)
    rho, p = spearmanr(df.consistency, df.before)
    ax1.text(0.03, 0.96, f"Spearman $\\rho$ = {rho:.2f} (n = {len(df)})", transform=ax1.transAxes,
             va="top", fontsize=9)
    ax1.set_xlabel("Sequence-direction consistency of held-out dataset")
    ax1.set_ylabel("Transfer to held-out dataset (balanced accuracy)")
    ax1.grid(color="#e5e5e5", lw=0.6, zorder=0)
    ax1.set_title("(a)", loc="left", fontsize=10)

    order = [e for e in ENCODERS][::-1]
    yy = np.arange(len(order))
    for i, (key, label, col, ctrl) in enumerate(order):
        r = s[s.representation == key].iloc[0]
        vals = [r.before, r.centre_free, r.centre_oracle]
        ax2.plot([min(vals), max(vals)], [i, i], color="#cccccc", lw=1, zorder=1)
        ax2.scatter(r.before, i, color=col, s=36, zorder=3, label="before" if i == 0 else None)
        ax2.scatter(r.centre_free, i, marker="s", facecolor="white", edgecolor=col, s=30, zorder=3,
                    label="label-free centring" if i == 0 else None)
        ax2.scatter(r.centre_oracle, i, marker="D", color="#222222", s=16, zorder=3,
                    label="class-balanced centring (oracle)" if i == 0 else None)
    ax2.set_yticks(yy, [e[1] for e in order], fontsize=8.5)
    ax2.set_xlabel("Transfer, mean over ten multi-sequence datasets")
    ax2.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    ax2.set_title("(b)", loc="left", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig_sequence_consistency")
    plt.close(fig)


def main():
    df = pd.concat([run(k) for k, *_ in ENCODERS], ignore_index=True)
    s = summarize(df)
    save_table(df, "sequence_consistency")
    save_table(s, "sequence_consistency_summary")
    print(s.round(3).to_string(index=False))

    rho, p = spearmanr(df.consistency, df.before)
    rho_r, p_r = spearmanr(s.consistency, s.before)
    rho_pool, p_pool = spearmanr(s.pool_consistency, s.before)
    print(f"rep x dataset rho={rho:.2f} p={p:.2g}; mean within-rep rho={s.rho_within.mean():.2f}; "
          f"rep-level rho={rho_r:.2f} p={p_r:.3f}; pool (no target) rho={rho_pool:.2f} p={p_pool:.3f}")
    add_number("Sequence-direction consistency vs LODO transfer",
               f"rep x held-out dataset Spearman rho={rho:.2f} (p={p:.1g}, n={len(df)}); mean within-representation "
               f"rho={s.rho_within.mean():.2f}; representation level rho={rho_r:.2f} (p={p_r:.3f}, n={len(s)}); "
               f"training-pool consistency (no target data) rho={rho_pool:.2f} (p={p_pool:.3f})")
    add_number("Centring inside LODO (balanced accuracy, 10 multi-sequence datasets; before / label-free / oracle)",
               "; ".join(f"{r.representation} {r.before:.2f}/{r.centre_free:.2f}/{r.centre_oracle:.2f}"
                         for r in s.itertuples()))
    figure(df, s)


if __name__ == "__main__":
    main()

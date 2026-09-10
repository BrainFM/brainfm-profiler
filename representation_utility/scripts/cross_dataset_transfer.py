"""Step 6: cross-dataset transfer of modality classification + label-free retrieval.

Writes results/tables/{lodo_transfer,within_dataset,pairwise_transfer,retrieval_purity}.csv
and results/figures/fig2_transfer.{png,pdf}.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import REPS, features, load_rep  # noqa: E402
from results_lib import add_number, save_fig, save_table  # noqa: E402

MODS = ["T1", "T1c", "T2", "FLAIR"]  # PD dropped (IXI-only)
FIG_REP = "medical"
N_PCA = 100


def fit_tr(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ncomp = min(N_PCA, Xtr.shape[0] - 1, Xtr.shape[1])
    if Xtr.shape[1] > ncomp:
        pca = PCA(ncomp, random_state=SEED).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    return Xtr, Xte


def clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def lodo(d, cols):
    d = d[d["modality"].isin(MODS)]
    X, y, ds = d[cols].to_numpy(float), d["modality"].to_numpy(), d["dataset"].to_numpy()
    rows = []
    for held in sorted(set(ds)):
        tr, te = ds != held, ds == held
        Xtr, Xte = fit_tr(X[tr], X[te])
        pred = clf().fit(Xtr, y[tr]).predict(Xte)
        k = len(set(y[te]))
        rows.append(dict(dataset=held, n=int(te.sum()), n_modalities=k,
                         accuracy=accuracy_score(y[te], pred),
                         balanced_acc=balanced_accuracy_score(y[te], pred) if k > 1 else np.nan))
    return pd.DataFrame(rows)


def within(d, cols):
    d = d[d["modality"].isin(MODS)]
    rows = []
    for ds, g in d.groupby("dataset"):
        if g["modality"].nunique() < 2:
            continue
        X, y, grp = g[cols].to_numpy(float), g["modality"].to_numpy(), g["patient"].to_numpy()
        accs = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, grp):
            Xtr, Xte = fit_tr(X[tr], X[te])
            accs.append(balanced_accuracy_score(y[te], clf().fit(Xtr, y[tr]).predict(Xte)))
        rows.append(dict(dataset=ds, within_bacc=float(np.mean(accs))))
    return pd.DataFrame(rows)


def pairwise(d, cols):
    d = d[d["modality"].isin(MODS)]
    src = [s for s in sorted(set(d["dataset"])) if d[d["dataset"] == s]["modality"].nunique() >= 3]
    rows = []
    for a in src:
        ga = d[d["dataset"] == a]
        for b in sorted(set(d["dataset"])):
            gb = d[d["dataset"] == b]
            Xtr, Xte = fit_tr(ga[cols].to_numpy(float), gb[cols].to_numpy(float))
            pred = clf().fit(Xtr, ga["modality"].to_numpy()).predict(Xte)
            rows.append(dict(source=a, target=b, accuracy=accuracy_score(gb["modality"].to_numpy(), pred)))
    return pd.DataFrame(rows)


def retrieval(d, cols, k=10):
    X = features(d, cols)
    _, nbr = NearestNeighbors(n_neighbors=k + 25, metric="cosine").fit(X).kneighbors(X)
    ds, pat, mod = d["dataset"].to_numpy(), d["patient"].to_numpy(), d["modality"].to_numpy()
    dpur, mpur = np.zeros(len(X)), np.zeros(len(X))
    for i in range(len(X)):
        cand = [j for j in nbr[i][1:] if pat[j] != pat[i]][:k]
        dpur[i] = np.mean([ds[j] == ds[i] for j in cand])
        mpur[i] = np.mean([mod[j] == mod[i] for j in cand])
    gfrac_ds = d["dataset"].value_counts(normalize=True)
    gfrac_mod = d["modality"].value_counts(normalize=True)
    dd = d.assign(dpur=dpur, mpur=mpur)
    rows = []
    for name, g in dd.groupby("dataset"):
        mchance = sum(g["modality"].value_counts(normalize=True).get(m, 0) * gfrac_mod.get(m, 0)
                      for m in gfrac_mod.index)
        rows.append(dict(dataset=name, dataset_purity=float(g["dpur"].mean()),
                         dataset_chance=float(gfrac_ds[name]),
                         modality_purity=float(g["mpur"].mean()), modality_chance=float(mchance)))
    return pd.DataFrame(rows)


def main():
    idx = pd.read_parquet(OUT / "file_index.parquet")
    reps = {r: load_rep(r) for r in REPS}

    lodo_all, ret_all = [], []
    within_all, pair = None, None
    for r, (d, cols) in reps.items():
        l = lodo(d, cols).assign(representation=r)
        lodo_all.append(l)
        ret_all.append(retrieval(d, cols).assign(representation=r))
        print(f"{r}: LODO modality accuracy mean={l['accuracy'].mean():.3f} "
              f"(min {l['accuracy'].min():.3f} @ {l.loc[l['accuracy'].idxmin(),'dataset']})")
    within_all = within(*reps[FIG_REP])
    pair = pairwise(*reps[FIG_REP]).assign(representation=FIG_REP)

    lodo_df = pd.concat(lodo_all, ignore_index=True)
    ret_df = pd.concat(ret_all, ignore_index=True)

    lodo_wide = lodo_df.pivot(index="dataset", columns="representation", values="accuracy")
    lodo_wide["n_modalities"] = lodo_df.groupby("dataset")["n_modalities"].first()
    lodo_wide["transfer_score"] = lodo_wide[FIG_REP]  # feeds DRUS (Step 7)
    lodo_wide = lodo_wide.reset_index().merge(idx.groupby("dataset")["domain"].first().reset_index())
    save_table(lodo_wide.sort_values("transfer_score"), "lodo_transfer")
    save_table(within_all, "within_dataset")
    save_table(pair, "pairwise_transfer")
    save_table(ret_df, "retrieval_purity")

    _figure(lodo_wide, within_all, ret_df[ret_df["representation"] == FIG_REP])
    _numbers(lodo_df, within_all, ret_df)


def _figure(lodo_wide, within_all, ret):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    w = lodo_wide.sort_values("transfer_score")
    ax1.barh(w["dataset"], w[FIG_REP], color=["#4c72b0" if n > 1 else "#c44e52" for n in w["n_modalities"]])
    ax1.axvline(0.25, ls="--", c="k", lw=1)
    ax1.set_xlabel(f"leave-one-dataset-out modality accuracy ({FIG_REP})")
    ax1.set_title("Cross-dataset transfer per dataset\n(red = single-modality test set)")
    ax1.set_xlim(0, 1)

    g = within_all.merge(lodo_wide[["dataset", FIG_REP]], on="dataset")
    ax2.scatter(g["within_bacc"], g[FIG_REP], s=45)
    for _, r in g.iterrows():
        ax2.annotate(r["dataset"], (r["within_bacc"], r[FIG_REP]), fontsize=6,
                     xytext=(3, 2), textcoords="offset points")
    ax2.plot([0, 1], [0, 1], "k:", lw=1)
    ax2.set_xlabel("within-dataset modality accuracy")
    ax2.set_ylabel("leave-one-dataset-out modality accuracy")
    ax2.set_title(f"The transfer gap ({FIG_REP}, multi-modality datasets)")
    ax2.set_xlim(0.5, 1)
    ax2.set_ylim(0, 1)
    save_fig(fig, "fig2_transfer")
    plt.close(fig)


def _numbers(lodo_df, within_all, ret_df):
    for r in REPS:
        s = lodo_df[lodo_df["representation"] == r]
        add_number(f"LODO modality transfer ({r})",
                   f"mean accuracy {s['accuracy'].mean():.2f}, worst {s['accuracy'].min():.2f} "
                   f"({s.loc[s['accuracy'].idxmin(), 'dataset']})")
    g = within_all.merge(lodo_df[lodo_df["representation"] == FIG_REP][["dataset", "accuracy"]], on="dataset")
    add_number("Transfer gap (medical)",
               f"within-dataset modality accuracy mean {g['within_bacc'].mean():.2f} collapses to "
               f"{g['accuracy'].mean():.2f} leave-one-dataset-out; "
               f"{(lodo_df[lodo_df.representation == 'medical']['accuracy'] < 0.3).sum()}/17 datasets transfer below 0.30")
    mm = ret_df[(ret_df.representation == FIG_REP) & (ret_df.dataset.isin(within_all.dataset))]
    add_number("Retrieval purity, multi-modality datasets (medical, k=10)",
               "neighbours share dataset far more than modality, e.g. "
               + "; ".join(f"{r.dataset} {r.dataset_purity:.2f} vs {r.modality_purity:.2f}"
                           for _, r in mm.nsmallest(3, "modality_purity").iterrows()))


if __name__ == "__main__":
    main()

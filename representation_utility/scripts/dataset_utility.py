"""Step 7: Dataset Representation-Learning Utility Score (DRUS) and its validation.

Writes results/tables/{drus_ranking,drus_validation}.csv and
results/figures/fig3_drus.{png,pdf}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import load_rep  # noqa: E402
from results_lib import RESULTS, add_number, save_fig, save_table  # noqa: E402

REP = "medical"
N_PCA = 100
MODS = ["T1", "T1c", "T2", "FLAIR"]
PLUS = ["emb_spread", "n_modalities", "log_n_subjects", "domain_rarity"]
MINUS = ["aniso_frac", "orient_entropy", "redundancy"]


def contribution(emb):
    """Per dataset X: mean drop in held-out modality accuracy when X is removed from the pool."""
    d = emb[emb["modality"].isin(MODS)].reset_index(drop=True)
    X = StandardScaler().fit_transform(d.filter(like="emb_").to_numpy(float))
    X = PCA(N_PCA, random_state=SEED).fit_transform(X)
    y, ds = d["modality"].to_numpy(), d["dataset"].to_numpy()
    datasets = sorted(set(ds))
    mm = [h for h in datasets if len(set(y[ds == h])) >= 2]

    def acc(tr, te):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        return accuracy_score(y[te], clf.predict(X[te]))

    contrib = {x: [] for x in datasets}
    for h in mm:
        te = ds == h
        base = acc(ds != h, te)
        for x in datasets:
            if x != h:
                contrib[x].append(base - acc((ds != h) & (ds != x), te))
    return pd.Series({x: float(np.mean(v)) for x, v in contrib.items()})


def descriptors(idx, emb):
    X = StandardScaler().fit_transform(emb.filter(like="emb_").to_numpy(float))
    X = StandardScaler().fit_transform(PCA(N_PCA, random_state=SEED).fit_transform(X))
    emb = emb.assign(**{f"z{i}": X[:, i] for i in range(X.shape[1])})
    zc = [f"z{i}" for i in range(X.shape[1])]

    cent = emb.groupby("dataset")[zc].mean()
    cn = cent.to_numpy()
    cn = cn / np.linalg.norm(cn, axis=1, keepdims=True)
    cos = cn @ cn.T
    np.fill_diagonal(cos, -1)
    redundancy = pd.Series(cos.max(1), index=cent.index)

    rows = []
    dom_counts = idx.groupby("domain")["dataset"].nunique()
    for ds, g in idx.groupby("dataset"):
        e = emb[emb["dataset"] == ds][zc].to_numpy()
        rows.append(dict(
            dataset=ds,
            emb_spread=float(np.sqrt(((e - e.mean(0)) ** 2).sum(1).mean())),
            n_modalities=int(g["modality"].nunique()),
            log_n_subjects=float(np.log10(g["patient"].nunique())),
            domain_rarity=float(1.0 / dom_counts[g["domain"].iloc[0]]),
            aniso_frac=float((g["anisotropy_cat"] != "Isotropic").mean()),
            orient_entropy=float(entropy(g["axcodes"].value_counts(normalize=True))),
            redundancy=float(redundancy[ds]),
        ))
    return pd.DataFrame(rows).set_index("dataset")


def drus(df, weights=None):
    z = (df - df.mean()) / df.std(ddof=0)
    w = {c: 1.0 for c in PLUS + MINUS} if weights is None else weights
    return sum(w[c] * z[c] for c in PLUS) - sum(w[c] * z[c] for c in MINUS)


def main():
    idx = pd.read_parquet(OUT / "file_index.parquet")
    d, _ = load_rep(REP)
    desc = descriptors(idx, d)
    desc["DRUS"] = drus(desc[PLUS + MINUS])

    lodo = pd.read_csv(RESULTS / "tables" / "lodo_transfer.csv").set_index("dataset")
    desc["transfer_score"] = lodo["transfer_score"]
    desc["contribution"] = contribution(d)          # primary validation target
    tgt = "contribution"

    out = desc.reset_index().sort_values("DRUS", ascending=False)
    save_table(out.round(4), "drus_ranking")

    # --- validation ---
    vrows = []
    for t in ["contribution", "transfer_score"]:
        rho, p = spearmanr(desc["DRUS"], desc[t])
        vrows.append(dict(test=f"DRUS vs {t}", subset="all (n=17)", value=rho, p=p))
    for c in PLUS + MINUS:
        rho, p = spearmanr(desc[c], desc[tgt])
        vrows.append(dict(test=f"component: {c}", subset=f"vs {tgt}", value=rho, p=p))
    base = spearmanr(desc["DRUS"], desc[tgt]).statistic
    for c in PLUS + MINUS:
        w = {x: (0.0 if x == c else 1.0) for x in PLUS + MINUS}
        rho = spearmanr(drus(desc[PLUS + MINUS], w), desc[tgt]).statistic
        vrows.append(dict(test=f"ablate {c}", subset=f"vs {tgt}", value=rho - base, p=np.nan))
    rng = np.random.default_rng(SEED)
    rhos = [spearmanr(drus(desc[PLUS + MINUS], {c: rng.uniform(0.5, 1.5) for c in PLUS + MINUS}),
                      desc[tgt]).statistic for _ in range(300)]
    vrows.append(dict(test="weight sensitivity (300x)", subset=f"vs {tgt}",
                      value=float(np.mean(rhos)), p=float(np.std(rhos))))
    taus = []
    full_rank = desc["DRUS"].rank()
    for ds in desc.index:
        r = drus(desc.drop(ds)[PLUS + MINUS]).rank()
        taus.append(spearmanr(full_rank.drop(ds), r).statistic)
    vrows.append(dict(test="LODO ranking stability", subset="all",
                      value=float(np.mean(taus)), p=float(np.min(taus))))

    val = pd.DataFrame(vrows)
    save_table(val.round(3), "drus_validation")

    _figure(desc, tgt)
    r_c = spearmanr(desc["DRUS"], desc["contribution"])
    add_number("DRUS validation",
               f"Spearman rho(DRUS, dataset contribution) = {r_c.statistic:+.2f} (p={r_c.pvalue:.2f}, n=17); "
               f"weight-sensitivity mean rho {np.mean(rhos):+.2f} +/- {np.std(rhos):.2f}; "
               f"LODO ranking stability rho >= {np.min(taus):.2f}")
    add_number("DRUS top / bottom",
               "top: " + ", ".join(out["dataset"].head(4)) + " | bottom: "
               + ", ".join(out["dataset"].tail(4)))
    print(val.round(3).to_string(index=False))


def _figure(desc, tgt):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rho = spearmanr(desc["DRUS"], desc[tgt]).statistic
    ax1.scatter(desc["DRUS"], desc[tgt], s=45, c="#4c72b0")
    for ds, r in desc.iterrows():
        ax1.annotate(ds, (r["DRUS"], r[tgt]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax1.axhline(0, c="k", lw=0.8, ls=":")
    ax1.set_xlabel("DRUS")
    ax1.set_ylabel("dataset contribution\n(mean drop in held-out modality accuracy when removed)")
    ax1.set_title(f"DRUS vs measured dataset contribution  (Spearman rho = {rho:+.2f}, n=17)")

    comp = [(c, spearmanr(desc[c], desc[tgt]).statistic) for c in PLUS + MINUS]
    comp.sort(key=lambda t: t[1])
    ax2.barh([c for c, _ in comp], [v for _, v in comp],
             color=["#c44e52" if v < 0 else "#55a868" for _, v in comp])
    ax2.axvline(0, c="k", lw=0.8)
    ax2.set_xlabel("Spearman correlation with dataset contribution")
    ax2.set_title("Which cheap descriptor predicts contribution?")
    save_fig(fig, "fig3_drus")
    plt.close(fig)


if __name__ == "__main__":
    main()

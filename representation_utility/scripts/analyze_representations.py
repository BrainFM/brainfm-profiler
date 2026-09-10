"""Step 5: how much acquisition vs content information each frozen representation encodes.

Writes results/tables/variance_explained.csv, results/tables/decodability.csv,
results/figures/fig1_decodability.{png,pdf}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from results_lib import add_number, save_fig, save_table  # noqa: E402

ID_COLS = ["image_id", "dataset", "domain", "patient", "modality"]
REPS = ["handcrafted", "untrained", "medical", "selfsup"]
N_PCA = 100
N_PERM = 100


def load_rep(name):
    d = pd.read_parquet(OUT / f"emb_{name}.parquet")
    cols = [c for c in d.columns if c not in ID_COLS]
    return d, cols


def features(d, cols):
    X = StandardScaler().fit_transform(d[cols].to_numpy(float))
    if X.shape[1] > N_PCA:
        X = PCA(N_PCA, random_state=SEED).fit_transform(X)
        X = StandardScaler().fit_transform(X)
    return X


def eta_sq(X, y):
    grand = X.mean(0)
    ss_tot = ((X - grand) ** 2).sum()
    ss_between = sum(len(X[y == c]) * ((X[y == c].mean(0) - grand) ** 2).sum() for c in np.unique(y))
    return float(ss_between / ss_tot)


def cv_score(X, y, groups, n_splits=5, n_repeats=3, seed=SEED, auc=True):
    accs, aucs = [], []
    for rep in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y, groups):
            clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
            if auc:
                try:
                    aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te]),
                                              multi_class="ovr", average="macro", labels=clf.classes_))
                except Exception:  # noqa: BLE001
                    aucs.append(np.nan)
    return float(np.mean(accs)), float(np.std(accs)), (float(np.nanmean(aucs)) if auc else np.nan)


def perm_null(X, y, groups, n_perm=N_PERM, seed=SEED):
    def one(i):
        rng = np.random.default_rng(seed + i + 1)
        yp = y.copy()
        rng.shuffle(yp)
        return cv_score(X, yp, groups, n_splits=3, n_repeats=1, seed=seed + i, auc=False)[0]
    null = Parallel(n_jobs=-1)(delayed(one)(i) for i in range(n_perm))
    return np.array(null)


def build_labels(idx):
    """Return dict factor -> (row mask, label series)."""
    ax = idx["axcodes"].copy()
    ax[ax.groupby(ax).transform("size") < 50] = "OTHER"
    ixi = idx["dataset"] == "IXI"
    site = idx["image_id"].str.split("_").str[1].str.split("-").str[1].where(ixi)
    return {
        "dataset": (idx.index, idx["dataset"]),
        "modality (no PD)": (idx.index[idx["modality"] != "PD"], idx["modality"]),
        "orientation": (idx.index, ax),
        "anisotropy": (idx.index, idx["anisotropy_cat"]),
        "site (IXI only)": (idx.index[ixi], site),
    }


def main():
    idx = pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True)
    factors = build_labels(idx)

    var_rows, dec_rows = [], []
    for name in REPS:
        d, cols = load_rep(name)
        d = idx[["image_id"]].merge(d, on="image_id")  # align row order to idx
        X_full = features(d, cols)
        for fac, (mask, labels) in factors.items():
            m = np.asarray(sorted(set(mask) & set(np.where(labels.notna())[0])))
            X, y, g = X_full[m], labels.iloc[m].to_numpy(), idx.loc[m, "patient"].to_numpy()
            k = len(np.unique(y))

            var_rows.append(dict(factor=fac, representation=name, n_classes=k, eta_sq=eta_sq(X, y)))

            acc, acc_sd, auc = cv_score(X, y, g)
            null = perm_null(X, y, g)
            dec_rows.append(dict(
                representation=name, factor=fac, n=len(y), n_classes=k,
                chance=1.0 / k, balanced_acc=acc, balanced_acc_sd=acc_sd, macro_auc=auc,
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                p_value=float((null >= acc).mean()),
            ))
            print(f"{name} {fac:18s} k={k:2d} bAcc={acc:.3f}±{acc_sd:.3f} AUC={auc:.3f} "
                  f"null={null.mean():.3f} p={dec_rows[-1]['p_value']:.3f}")

    var = pd.DataFrame(var_rows).pivot(index="factor", columns="representation", values="eta_sq").reset_index()
    save_table(var, "variance_explained")
    dec = pd.DataFrame(dec_rows)
    save_table(dec, "decodability")

    _figure(dec)
    _numbers(dec)


def _figure(dec):
    import matplotlib.pyplot as plt

    facs = list(dict.fromkeys(dec["factor"]))
    reps = REPS
    x = np.arange(len(facs))
    w = 0.8 / len(reps)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, r in enumerate(reps):
        sub = dec[dec["representation"] == r].set_index("factor").loc[facs]
        ax.bar(x + (j - (len(reps) - 1) / 2) * w, sub["balanced_acc"], w,
               yerr=sub["balanced_acc_sd"], capsize=2, label=r)
    for i, f in enumerate(facs):
        ch = dec[dec["factor"] == f]["chance"].iloc[0]
        ax.plot([i - 0.45, i + 0.45], [ch, ch], "k--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(facs, rotation=15, ha="right")
    ax.set_ylabel("balanced accuracy (patient-level CV)")
    ax.set_title("Decodability of acquisition and content factors from frozen representations")
    ax.set_ylim(0, 1)
    ax.legend(title="representation")
    save_fig(fig, "fig1_decodability")
    plt.close(fig)


def _numbers(dec):
    for f in ["dataset", "modality (no PD)"]:
        s = dec[dec["factor"] == f].set_index("representation")
        add_number(
            f"Decodability: {f}",
            "; ".join(f"{r} bAcc={s.loc[r, 'balanced_acc']:.2f}" for r in REPS)
            + f" (chance {s['chance'].iloc[0]:.2f}, all permutation p<{max(dec['p_value'].max(), 0.01):.2f})")


if __name__ == "__main__":
    main()

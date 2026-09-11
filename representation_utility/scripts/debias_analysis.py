"""Representation-level re-analysis after INLP removal.

Uses the exact feature space and projection P saved by debias.py so that
before/after differ only by the P multiplication:
  - decodability before/after, with a permutation null on 'after'
  - LODO modality transfer before/after, vs a random-projection control of the
    same rank (rules out "improvement is just dimensionality reduction")
  - retrieval purity (dataset vs modality) before/after
  - a non-linear (MLP) residual probe for dataset before/after (how much
    non-linear signal survives linear removal)
For `medical` only (compute budget): an ablation over the retention threshold,
and a stability check (P refit on an independent patient split).

Writes results/tables/debias_{decodability,transfer,retrieval,residual,
ablation_threshold,stability}.csv and results/figures/fig_debias_tradeoff.*.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import REMOVE_ORDER, RETAIN, factor_frame, fit_feature_pipeline, inlp_rounds, nullspace_of, probe_acc  # noqa: E402
from results_lib import save_fig, save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
MODS = ["T1", "T1c", "T2", "FLAIR"]
N_PERM = 20


# ------------------------------------------------------------------ setup ---
def load(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)
    npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows, eval_rows = npz["fit_rows"], npz["eval_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    return ff, Z, npz["P"], fit_rows, eval_rows, int(npz["d"]), dict(zip(npz["factor_order"], npz["dims_removed"]))


def random_projection(d, k, seed):
    """Null space of k random directions -- same rank deficiency as a learned P."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(k, d))
    Q, _ = np.linalg.qr(W.T)
    return np.eye(d) - Q @ Q.T


# ------------------------------------------------------------ decodability ---
def decodability_perm(Z, ff, eval_rows, P, factors):
    rows = []
    sub = ff.iloc[eval_rows].reset_index(drop=True)
    for fac in factors:
        m = sub[fac].notna().to_numpy()
        y = sub.loc[m, fac].astype(str).to_numpy()
        g = sub.loc[m, "patient"].to_numpy()
        Zf = Z[eval_rows][m]
        k = len(np.unique(y))
        chance = 1.0 / k
        row = dict(factor=fac, chance=chance, n=int(m.sum()), n_classes=k)
        for tag, Zx in (("before", Zf), ("after", Zf @ P)):
            accs = []
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Zx, y, g):
                a, _ = probe_acc(Zx[tr], y[tr], Zx[te], y[te])
                accs.append(a)
            row[tag] = float(np.mean(accs))
        # permutation null on the after condition only
        rng = np.random.default_rng(SEED)
        null = []
        Zx = Zf @ P
        for i in range(N_PERM):
            yp = y.copy()
            rng.shuffle(yp)
            tr, te = next(StratifiedGroupKFold(3, shuffle=True, random_state=SEED + i).split(Zx, yp, g))
            a, _ = probe_acc(Zx[tr], yp[tr], Zx[te], yp[te])
            null.append(a)
        row["after_null_mean"] = float(np.mean(null))
        row["after_p"] = float((np.array(null) >= row["after"]).mean())
        rows.append(row)
        print(f"    {fac:24s} chance={chance:.2f} before={row['before']:.2f} after={row['after']:.2f} "
              f"null={row['after_null_mean']:.2f} p={row['after_p']:.2f}")
    return rows


# ------------------------------------------------------------------- LODO ---
def lodo_transfer(Z, ff, P, Prand):
    d = ff[ff["modality"].isin(MODS)]
    rows_idx = d.index.to_numpy()
    y_all, ds_all = d["modality"].to_numpy(), d["dataset"].to_numpy()
    out = []
    for held in sorted(set(ds_all)):
        tr, te = ds_all != held, ds_all == held
        if te.sum() == 0 or len(set(y_all[tr])) < 2:
            continue
        k = len(set(y_all[te]))
        for tag, M in (("before", np.eye(P.shape[0])), ("after", P), ("random", Prand)):
            Zx = Z[rows_idx] @ M
            sc = StandardScaler().fit(Zx[tr])
            pred = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                sc.transform(Zx[tr]), y_all[tr]).predict(sc.transform(Zx[te]))
            out.append(dict(dataset=held, n_modalities=k, condition=tag,
                            accuracy=accuracy_score(y_all[te], pred)))
    return pd.DataFrame(out)


# --------------------------------------------------------------- retrieval ---
def retrieval_purity(Z, ff, P, k=10):
    ds, pat, mod = ff["dataset"].to_numpy(), ff["patient"].to_numpy(), ff["modality"].to_numpy()
    has_mod = pd.notna(mod)
    rows = []
    for tag, Zx in (("before", Z), ("after", Z @ P)):
        _, nbr = NearestNeighbors(n_neighbors=k + 25, metric="cosine").fit(Zx).kneighbors(Zx)
        dpur = np.full(len(Zx), np.nan)
        mpur = np.full(len(Zx), np.nan)
        for i in range(len(Zx)):
            cand = [j for j in nbr[i][1:] if pat[j] != pat[i]][:k]
            if not cand:
                continue
            dpur[i] = np.mean([ds[j] == ds[i] for j in cand])
            if has_mod[i]:
                mc = [j for j in cand if has_mod[j]]
                if mc:
                    mpur[i] = np.mean([mod[j] == mod[i] for j in mc])
        rows.append(dict(condition=tag, dataset_purity=float(np.nanmean(dpur)),
                         modality_purity=float(np.nanmean(mpur))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- residual ---
def nonlinear_residual(Z, ff, eval_rows, P):
    sub = ff.iloc[eval_rows].reset_index(drop=True)
    y, g = sub["dataset"].to_numpy(), sub["patient"].to_numpy()
    Zf = Z[eval_rows]
    rows = []
    for tag, Zx in (("before", Zf), ("after", Zf @ P)):
        accs = []
        for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(Zx, y, g):
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=SEED, early_stopping=True)
            clf.fit(Zx[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(Zx[te])))
        rows.append(dict(factor="dataset", probe="MLP", condition=tag, balanced_acc=float(np.mean(accs))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ per-rep run ---
def analyze_rep(rep):
    print(f"\n=== {rep} ===")
    ff, Z, P, fit_rows, eval_rows, d, dims_removed = load(rep)
    total_k = int(max(dims_removed.values())) if dims_removed else 0  # cumulative, not additive
    Prand = random_projection(d, max(total_k, 1), SEED)

    print("  decodability (before/after/null):")
    dec_rows = decodability_perm(Z, ff, eval_rows, P, [RETAIN, *REMOVE_ORDER])
    for r in dec_rows:
        r["representation"] = rep

    lodo = lodo_transfer(Z, ff, P, Prand)
    lodo["representation"] = rep
    print(f"  LODO modality transfer: before={lodo[lodo.condition=='before'].accuracy.mean():.3f} "
          f"after={lodo[lodo.condition=='after'].accuracy.mean():.3f} "
          f"random-control={lodo[lodo.condition=='random'].accuracy.mean():.3f}  (k_removed={total_k})")

    ret = retrieval_purity(Z, ff, P)
    ret["representation"] = rep
    print(f"  retrieval purity: {ret.to_dict('records')}")

    res = nonlinear_residual(Z, ff, eval_rows, P)
    res["representation"] = rep
    print(f"  non-linear residual (dataset, MLP): "
          f"before={res[res.condition=='before'].balanced_acc.iloc[0]:.3f} "
          f"after={res[res.condition=='after'].balanced_acc.iloc[0]:.3f}")

    return dec_rows, lodo, ret, res, total_k


# ------------------------------------------------- ablation over threshold ---
def fit_operator(Z, ff, fit_rows, d, retain_drop):
    in_fit = np.isin(np.arange(len(ff)), fit_rows)
    mod_m = ff["modality"].notna().to_numpy() & in_fit
    mod_y, mod_g = ff.loc[mod_m, "modality"].to_numpy(), ff.loc[mod_m, "patient"].to_numpy()
    mod_base, _ = probe_acc(*_grp_split(Z[mod_m], mod_y, mod_g))
    rowspaces = []
    import debias as _db
    old = _db.RETAIN_DROP
    _db.RETAIN_DROP = retain_drop
    try:
        for fac in REMOVE_ORDER:
            m = ff[fac].notna().to_numpy() & in_fit
            y, g = ff.loc[m, fac].astype(str).to_numpy(), ff.loc[m, "patient"].to_numpy()
            k = len(np.unique(y))
            inlp_rounds(Z[m], y, g, rowspaces, d, 1.0 / k, Z[mod_m], mod_y, mod_g, mod_base)
    finally:
        _db.RETAIN_DROP = old
    return nullspace_of(rowspaces, d)


def _grp_split(Z, y, g):
    tr, va = next(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(Z, y, g))
    return Z[tr], y[tr], Z[va], y[va]


def ablation_threshold(rep="medical", thresholds=(0.03, 0.05, 0.08, 0.15)):
    print(f"\n=== ablation over retention threshold ({rep}) ===")
    ff, Z, _, fit_rows, eval_rows, d, _ = load(rep)
    rows = []
    for th in thresholds:
        P = fit_operator(Z, ff, fit_rows, d, th)
        k_removed = d - np.linalg.matrix_rank(P, tol=1e-6)
        dec = decodability_perm(Z, ff, eval_rows, P, ["dataset", "modality"])
        lodo = lodo_transfer(Z, ff, P, random_projection(d, max(int(k_removed), 1), SEED))
        rows.append(dict(
            threshold=th, dims_removed=int(k_removed),
            dataset_after=dec[0]["after"], modality_after=dec[1]["after"],
            lodo_after=lodo[lodo.condition == "after"].accuracy.mean(),
            lodo_before=lodo[lodo.condition == "before"].accuracy.mean(),
        ))
        print(f"  threshold={th:.2f} dims_removed={k_removed:3d} "
              f"dataset={rows[-1]['dataset_after']:.2f} modality={rows[-1]['modality_after']:.2f} "
              f"LODO={rows[-1]['lodo_after']:.2f}")
    return pd.DataFrame(rows)


def stability_check(rep="medical", retain_drop=0.05, seed2=SEED + 1):
    print(f"\n=== stability check ({rep}) ===")
    ff, Z, P1, fit_rows1, eval_rows1, d, dims1 = load(rep)
    rng = np.random.default_rng(seed2)
    pats = ff["patient"].unique()
    fit_pats2 = set(rng.choice(pats, int(0.6 * len(pats)), replace=False))
    fit_rows2 = np.where(ff["patient"].isin(fit_pats2).to_numpy())[0]
    P2 = fit_operator(Z, ff, fit_rows2, d, retain_drop)
    k1, k2 = d - np.linalg.matrix_rank(P1, tol=1e-6), d - np.linalg.matrix_rank(P2, tol=1e-6)
    lodo1 = lodo_transfer(Z, ff, P1, np.eye(d))
    lodo2 = lodo_transfer(Z, ff, P2, np.eye(d))
    a1 = lodo1[lodo1.condition == "after"].accuracy.mean()
    a2 = lodo2[lodo2.condition == "after"].accuracy.mean()
    print(f"  split A: dims_removed={k1} LODO_after={a1:.3f}")
    print(f"  split B: dims_removed={k2} LODO_after={a2:.3f}")
    return pd.DataFrame([dict(split="A", dims_removed=int(k1), lodo_after=a1),
                        dict(split="B", dims_removed=int(k2), lodo_after=a2)])


# ------------------------------------------------------------------- main ---
def main():
    all_dec, all_lodo, all_ret, all_res, ks = [], [], [], [], {}
    for rep in REPS:
        dec, lodo, ret, res, k = analyze_rep(rep)
        all_dec += dec
        all_lodo.append(lodo)
        all_ret.append(ret)
        all_res.append(res)
        ks[rep] = k

    save_table(pd.DataFrame(all_dec), "debias_decodability")
    save_table(pd.concat(all_lodo, ignore_index=True), "debias_transfer")
    save_table(pd.concat(all_ret, ignore_index=True), "debias_retrieval")
    save_table(pd.concat(all_res, ignore_index=True), "debias_residual")

    ab = ablation_threshold()
    save_table(ab, "debias_ablation_threshold")

    st = stability_check()
    save_table(st, "debias_stability")

    _figure(ab)


def _figure(ab):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(ab["dims_removed"], ab["dataset_after"], "o-", label="dataset decodability (after)")
    ax1.plot(ab["dims_removed"], ab["modality_after"], "s-", label="modality decodability (after)")
    ax1.plot(ab["dims_removed"], ab["lodo_after"], "^-", label="modality LODO transfer (after)")
    ax1.axhline(ab["lodo_before"].iloc[0], color="gray", ls="--", lw=1, label="modality LODO (before)")
    ax1.set_xlabel("dimensions removed (medical, 100-d space)")
    ax1.set_ylabel("balanced / accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.set_title("Retention-threshold ablation (medical)")
    save_fig(fig, "fig_debias_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    main()

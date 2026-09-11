"""Iterative nullspace projection (INLP) of the non-biological subspace.

For each representation, fit one projection matrix P on a patient-disjoint split
that removes the linearly decodable directions for a graded sequence of
non-biological factors, then report decodability before / after on the held-out
split. P (and the feature pipeline it lives in) is saved for Step 11.

INLP: Ravfogel et al., "Null It Out", ACL 2020.

Usage: python debias.py [--reps handcrafted untrained medical selfsup]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank
from scipy.linalg import orth
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402

# graded removal order; modality is the retention check, not a target
REMOVE_ORDER = ["orientation", "voxel spacing", "template space", "intensity scale",
                "bias field", "motion/ghosting (EFC)", "dataset"]
RETAIN = "modality"
N_PCA = 100
MAX_ROUNDS = 40        # INLP rounds per factor
PER_ROUND_RANK = 1     # directions removed per round (SVD-truncated probe rowspace)
ACC_EPS = 0.02         # stop when factor bAcc <= chance + eps
RETAIN_DROP = 0.05     # abort a factor if modality bAcc falls this far below baseline
MAX_DIMS_FRAC = 0.5    # never remove more than this fraction of the space
FIT_FRAC = 0.6


# ------------------------------------------------------------------ factors ---
def factor_frame(idx):
    """One tidy frame: image_id + every factor column as a category, NaN where undefined."""
    ax = idx["axcodes"].copy()
    ax[ax.groupby(ax).transform("size") < 50] = "OTHER"
    ts = idx["template_space"].copy()
    ts[ts.groupby(ts).transform("size") < 50] = "other"
    return pd.DataFrame({
        "image_id": idx["image_id"].to_numpy(),
        "patient": idx["patient"].to_numpy(),
        "modality": idx["modality"].where(idx["modality"] != "PD"),
        "orientation": ax.to_numpy(),
        "voxel spacing": idx["anisotropy_cat"].to_numpy(),
        "template space": ts.to_numpy(),
        "intensity scale": idx["intensity_scale"].astype(object).to_numpy(),
        "bias field": idx["biasfield_cat"].astype(object).to_numpy(),
        "motion/ghosting (EFC)": idx["efc_cat"].astype(object).to_numpy(),
        "dataset": idx["dataset"].to_numpy(),
    })


# ------------------------------------------------------------ feature space ---
def fit_feature_pipeline(X_raw, fit_rows):
    """StandardScaler -> PCA(100) -> StandardScaler, all fit on fit_rows only."""
    sc1 = StandardScaler().fit(X_raw[fit_rows])
    Z = sc1.transform(X_raw)
    pca = None
    if Z.shape[1] > N_PCA:
        pca = PCA(N_PCA, random_state=SEED).fit(Z[fit_rows])
        Z = pca.transform(Z)
    sc2 = StandardScaler().fit(Z[fit_rows])
    Z = sc2.transform(Z)
    return Z, (sc1, pca, sc2)


# -------------------------------------------------------------------- INLP ---
def rowspace_proj(W):
    if np.allclose(W, 0):
        return np.zeros((W.shape[1], W.shape[1]))
    B = orth(W.T)
    return B @ B.T


def topr_rowspace(W, r):
    """Projection onto the top-r right singular directions of a probe's weights."""
    _, _, Vt = np.linalg.svd(np.atleast_2d(W), full_matrices=False)
    B = Vt[:r].T
    return B @ B.T


def nullspace_of(rowspaces, d):
    if not rowspaces:
        return np.eye(d)
    return np.eye(d) - rowspace_proj(np.sum(rowspaces, axis=0))


def probe_acc(Ztr, ytr, Zva, yva):
    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
    clf.fit(Ztr, ytr)
    return balanced_accuracy_score(yva, clf.predict(Zva)), clf


def inlp_rounds(Z, y, groups, rowspaces, d, chance, Zm, ym, gm, mod_base):
    """Add rounds for one factor to the shared `rowspaces`; return (n_rounds, final_acc)."""
    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    tr, va = next(gkf.split(Z, y, groups))
    mtr, mva = next(gkf.split(Zm, ym, gm))
    n, acc = 0, chance
    for _ in range(MAX_ROUNDS):
        P = nullspace_of(rowspaces, d)
        if d - matrix_rank(P, tol=1e-6) >= MAX_DIMS_FRAC * d:
            break
        acc, clf = probe_acc(Z[tr] @ P, y[tr], Z[va] @ P, y[va])
        if acc <= chance + ACC_EPS:
            break
        cand = rowspaces + [topr_rowspace(clf.coef_, PER_ROUND_RANK)]
        Pc = nullspace_of(cand, d)
        m_acc, _ = probe_acc(Zm[mtr] @ Pc, ym[mtr], Zm[mva] @ Pc, ym[mva])
        if m_acc < mod_base - RETAIN_DROP:            # would hurt content -> stop this factor
            break
        rowspaces[:] = cand
        n += 1
    return n, acc


# ----------------------------------------------------------------- per rep ---
def decodability(Z, ff, eval_rows, P):
    """5x1 patient-grouped CV bAcc for every factor on the eval split, raw vs P-projected."""
    out = {}
    sub = ff.iloc[eval_rows].reset_index(drop=True)
    for fac in [RETAIN, *REMOVE_ORDER]:
        m = sub[fac].notna().to_numpy()
        y = sub.loc[m, fac].astype(str).to_numpy()
        g = sub.loc[m, "patient"].to_numpy()
        Zf = Z[eval_rows][m]
        k = len(np.unique(y))
        row = {"chance": 1.0 / k}
        for tag, Zx in (("before", Zf), ("after", Zf @ P)):
            accs = []
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Zx, y, g):
                a, _ = probe_acc(Zx[tr], y[tr], Zx[te], y[te])
                accs.append(a)
            row[tag] = float(np.mean(accs))
        out[fac] = row
    return out


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    rng = np.random.default_rng(SEED)
    pats = ff["patient"].unique()
    fit_pats = set(rng.choice(pats, int(FIT_FRAC * len(pats)), replace=False))
    fit_rows = np.where(ff["patient"].isin(fit_pats).to_numpy())[0]
    eval_rows = np.where(~ff["patient"].isin(fit_pats).to_numpy())[0]

    Z, pipe = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    in_fit = np.isin(np.arange(len(ff)), fit_rows)
    mod_m = ff["modality"].notna().to_numpy() & in_fit
    mod_y = ff.loc[mod_m, "modality"].to_numpy()
    mod_g = ff.loc[mod_m, "patient"].to_numpy()
    mod_base, _ = probe_acc(*_split(Z[mod_m], mod_y, mod_g))
    print(f"[{rep}] d={d}  modality baseline bAcc={mod_base:.3f}  "
          f"(fit {len(fit_rows)} / eval {len(eval_rows)} rows)")

    rowspaces, k_hist = [], {}
    for fac in REMOVE_ORDER:
        m = ff[fac].notna().to_numpy() & in_fit
        y = ff.loc[m, fac].astype(str).to_numpy()
        g = ff.loc[m, "patient"].to_numpy()
        k = len(np.unique(y))
        n, acc = inlp_rounds(Z[m], y, g, rowspaces, d, 1.0 / k,
                             Z[mod_m], mod_y, mod_g, mod_base)
        P = nullspace_of(rowspaces, d)
        dims = d - matrix_rank(P, tol=1e-6)
        k_hist[fac] = int(dims)
        print(f"  {fac:24s} rounds={n:2d}  factor bAcc->{acc:.2f} (chance {1/k:.2f})  "
              f"cumulative dims removed={dims}/{d}")

    P = nullspace_of(rowspaces, d)
    dec = decodability(Z, ff, eval_rows, P)
    print(f"  {'factor':24s} {'chance':>7} {'before':>8} {'after':>8}")
    for fac, r in dec.items():
        print(f"  {fac:24s} {r['chance']:7.2f} {r['before']:8.2f} {r['after']:8.2f}")

    np.savez(OUT / f"debias_P_{rep}.npz",
             P=P, dims_removed=np.array([k_hist[f] for f in REMOVE_ORDER]),
             factor_order=np.array(REMOVE_ORDER), d=d, fit_rows=fit_rows, eval_rows=eval_rows,
             decode_before=np.array([dec[f]["before"] for f in [RETAIN, *REMOVE_ORDER]]),
             decode_after=np.array([dec[f]["after"] for f in [RETAIN, *REMOVE_ORDER]]),
             decode_factors=np.array([RETAIN, *REMOVE_ORDER]),
             decode_chance=np.array([dec[f]["chance"] for f in [RETAIN, *REMOVE_ORDER]]))
    print(f"  saved outputs/debias_P_{rep}.npz")
    return dec


def _split(Z, y, g):
    tr, va = next(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(Z, y, g))
    return Z[tr], y[tr], Z[va], y[va]


if __name__ == "__main__":
    reps = ["handcrafted", "untrained", "medical", "selfsup"]
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    for r in reps:
        run(r)
        print()

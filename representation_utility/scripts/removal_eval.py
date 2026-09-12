"""Callable-based generalization of debias_analysis.py's four evaluation
functions, so a non-matrix removal method (CORAL's dataset-conditional
affine map, the adversarial network's non-linear g_theta) can be scored with
the exact same protocol as INLP/LEACE, without touching debias_analysis.py
(whose numbers are already compiled into the dissertation).

A Transform is any callable (Z, dataset_labels) -> Z' of matching shape.
For a plain linear method, `matrix_transform(P)` reproduces `Z @ P` exactly.

Every function here mirrors its debias_analysis.py counterpart 1:1 in logic
(same CV scheme, same permutation-null design, same neighbour-purity
definition) -- only the "apply P" step is replaced by "call transform".
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import SEED  # noqa: E402
from debias import probe_acc  # noqa: E402

MODS = ["T1", "T1c", "T2", "FLAIR"]
N_PERM = 20

Transform = Callable[[np.ndarray, np.ndarray], np.ndarray]


def matrix_transform(P: np.ndarray) -> Transform:
    """Wrap a d x d matrix as a Transform, for parity with debias_analysis.py."""
    return lambda Z, ds: Z @ P


def identity_transform(Z: np.ndarray, ds: np.ndarray) -> np.ndarray:
    return Z


# ------------------------------------------------------------ decodability ---
def decodability_perm(Z, ff, eval_rows, transform: Transform, factors, seed=SEED, n_perm=N_PERM):
    """Same as debias_analysis.decodability_perm, with `transform` replacing `Z @ P`."""
    rows = []
    sub = ff.iloc[eval_rows].reset_index(drop=True)
    for fac in factors:
        m = sub[fac].notna().to_numpy()
        y = sub.loc[m, fac].astype(str).to_numpy()
        g = sub.loc[m, "patient"].to_numpy()
        ds = sub.loc[m, "dataset"].to_numpy()
        Zf = Z[eval_rows][m]
        k = len(np.unique(y))
        chance = 1.0 / k
        row = dict(factor=fac, chance=chance, n=int(m.sum()), n_classes=k)
        for tag, Zx in (("before", Zf), ("after", transform(Zf, ds))):
            accs = []
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Zx, y, g):
                a, _ = probe_acc(Zx[tr], y[tr], Zx[te], y[te])
                accs.append(a)
            row[tag] = float(np.mean(accs))
        rng = np.random.default_rng(seed)
        null = []
        Zx = transform(Zf, ds)
        for i in range(n_perm):
            yp = y.copy()
            rng.shuffle(yp)
            tr, te = next(StratifiedGroupKFold(3, shuffle=True, random_state=seed + i).split(Zx, yp, g))
            a, _ = probe_acc(Zx[tr], yp[tr], Zx[te], yp[te])
            null.append(a)
        row["after_null_mean"] = float(np.mean(null))
        row["after_p"] = float((np.array(null) >= row["after"]).mean())
        rows.append(row)
        print(f"    {fac:24s} chance={chance:.2f} before={row['before']:.2f} after={row['after']:.2f} "
              f"null={row['after_null_mean']:.2f} p={row['after_p']:.2f}")
    return rows


# ------------------------------------------------------------------- LODO ---
def lodo_transfer(Z, ff, transform_after: Transform, transform_random: Transform):
    """Same as debias_analysis.lodo_transfer, with two Transforms replacing P/Prand."""
    d = ff[ff["modality"].isin(MODS)]
    rows_idx = d.index.to_numpy()
    y_all, ds_all = d["modality"].to_numpy(), d["dataset"].to_numpy()
    Zr = Z[rows_idx]
    conditions = (
        ("before", identity_transform(Zr, ds_all)),
        ("after", transform_after(Zr, ds_all)),
        ("random", transform_random(Zr, ds_all)),
    )
    out = []
    for held in sorted(set(ds_all)):
        tr, te = ds_all != held, ds_all == held
        if te.sum() == 0 or len(set(y_all[tr])) < 2:
            continue
        k = len(set(y_all[te]))
        for tag, Zx in conditions:
            sc = StandardScaler().fit(Zx[tr])
            pred = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                sc.transform(Zx[tr]), y_all[tr]).predict(sc.transform(Zx[te]))
            out.append(dict(dataset=held, n_modalities=k, condition=tag,
                            accuracy=accuracy_score(y_all[te], pred)))
    return pd.DataFrame(out)


# --------------------------------------------------------------- retrieval ---
def retrieval_purity(Z, ff, transform: Transform, k=10):
    """Same as debias_analysis.retrieval_purity, with `transform` replacing `Z @ P`."""
    ds, pat, mod = ff["dataset"].to_numpy(), ff["patient"].to_numpy(), ff["modality"].to_numpy()
    has_mod = pd.notna(mod)
    rows = []
    for tag, Zx in (("before", Z), ("after", transform(Z, ds))):
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
def nonlinear_residual(Z, ff, eval_rows, transform: Transform, seed=SEED):
    """Same as debias_analysis.nonlinear_residual, with `transform` replacing `Z @ P`."""
    sub = ff.iloc[eval_rows].reset_index(drop=True)
    y, g = sub["dataset"].to_numpy(), sub["patient"].to_numpy()
    ds = sub["dataset"].to_numpy()
    Zf = Z[eval_rows]
    rows = []
    for tag, Zx in (("before", Zf), ("after", transform(Zf, ds))):
        accs = []
        for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Zx, y, g):
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed, early_stopping=True)
            clf.fit(Zx[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(Zx[te])))
        rows.append(dict(factor="dataset", probe="MLP", condition=tag, balanced_acc=float(np.mean(accs))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ random ---
def random_projection(d, k, seed):
    """Null space of k random directions -- identical to debias_analysis.random_projection."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(k, d))
    Q, _ = np.linalg.qr(W.T)
    return np.eye(d) - Q @ Q.T

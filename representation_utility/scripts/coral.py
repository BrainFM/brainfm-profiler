"""CORAL-style distribution alignment (Sun & Saenko, "Deep CORAL", 2016):
instead of erasing a labeled direction (INLP/LEACE), whiten each dataset's own
embedding covariance and recolor it to a shared reference covariance/mean.
A fundamentally different removal philosophy -- alignment of whole
distributions, not erasure of one direction -- which might succeed where
erasure failed, since Section 2.4/2.7's finding is that the confound is
*diffuse* across the embedding rather than concentrated in a few directions.

Per dataset d: T_d(Z) = (Z - mu_d) @ W_d @ C_ref + mu_ref, where W_d whitens
using dataset d's own (shrinkage-regularized) covariance and C_ref recolors to
the pooled fit-split reference covariance. This is dataset-conditional and
affine, not a single global projection matrix -- unlike INLP/LEACE's P, so it
is evaluated through removal_eval.py's callable-based Transform interface.

Small-sample fix: per-dataset covariance in a 100-d space from as few as ~56
fit-split scans (UMF-PD) is badly underdetermined, so each Sigma_d is shrunk
toward the (well-estimated, n=2533) pooled reference: Sigma_d_reg =
(1-alpha) Sigma_d + alpha Sigma_ref, alpha = d / (d + n_d) -- a simple
shrinkage-to-target heuristic (not full Ledoit-Wolf optimal shrinkage).

Transductive by design (flagged, not glossed over): a held-out (LODO)
dataset's own Sigma_d/mu_d are still computed from *that dataset's own*
fit-split rows -- unlabeled (no modality labels used), but a genuinely
different information regime from INLP/LEACE's P, which never looks at the
target dataset at all. Worth being explicit about when comparing methods.

Random control: recolor each dataset to a different, randomly-assigned OTHER
dataset's own (mu, Sigma) instead of the shared correct reference -- same
operation, wrong target -- isolating whether alignment to the *correct*
reference matters versus alignment per se.

Reuses INLP's saved fit_rows/eval_rows (outputs/debias_P_<rep>.npz) so every
method is compared on identical held-out patients.

Writes outputs/coral_<rep>.npz (mu_ref, Sigma_ref stats, per-dataset mu/W,
dataset order, fit_rows, eval_rows, d).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import factor_frame, fit_feature_pipeline  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
EPS = 1e-6


def sqrt_inv_sqrt(Sigma, eps=EPS):
    """Return (W, C): W whitens (Sigma^-1/2), C colors (Sigma^1/2)."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T
    C = vecs @ np.diag(vals ** 0.5) @ vecs.T
    return W, C


def fit_dataset_stats(Z, ds_labels, datasets):
    """Pooled reference + per-dataset (mean, shrinkage-regularized covariance)."""
    d = Z.shape[1]
    mu_ref = Z.mean(0)
    Sigma_ref = np.cov(Z, rowvar=False)
    mus, Sigmas, ns = {}, {}, {}
    for ds in datasets:
        Zd = Z[ds_labels == ds]
        n_d = len(Zd)
        mu_d = Zd.mean(0)
        Sigma_d = np.cov(Zd, rowvar=False) if n_d > 1 else Sigma_ref.copy()
        alpha = d / (d + n_d)  # shrink harder for smaller datasets
        Sigmas[ds] = (1 - alpha) * Sigma_d + alpha * Sigma_ref
        mus[ds] = mu_d
        ns[ds] = n_d
    return mu_ref, Sigma_ref, mus, Sigmas, ns


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows, eval_rows = npz["fit_rows"], npz["eval_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    ds_fit = ff["dataset"].to_numpy()[fit_rows]
    datasets = sorted(set(ds_fit))
    mu_ref, Sigma_ref, mus, Sigmas, ns = fit_dataset_stats(Z[fit_rows], ds_fit, datasets)

    W_by_ds = {ds: sqrt_inv_sqrt(Sigmas[ds])[0] for ds in datasets}
    _, C_ref = sqrt_inv_sqrt(Sigma_ref)

    # derangement for the random-target control: each dataset recolors to a
    # different OTHER dataset's own (mu, Sigma) instead of the shared mu_ref/C_ref
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(datasets)
    while any(a == b for a, b in zip(datasets, perm)) and len(datasets) > 1:
        perm = rng.permutation(datasets)
    random_target = dict(zip(datasets, perm))
    W_target_by_ds = {ds: sqrt_inv_sqrt(Sigmas[random_target[ds]])[1] for ds in datasets}  # C of the wrong target
    mu_target_by_ds = {ds: mus[random_target[ds]] for ds in datasets}

    print(f"[{rep}] d={d}  {len(datasets)} datasets  "
          f"fit-split sizes {min(ns.values())}-{max(ns.values())} (shrinkage alpha "
          f"{d/(d+max(ns.values())):.2f}-{d/(d+min(ns.values())):.2f})")

    np.savez(OUT / f"coral_{rep}.npz",
             datasets=np.array(datasets, dtype=object),
             mu_ref=mu_ref, C_ref=C_ref,
             mu_by_ds=np.array([mus[ds] for ds in datasets]),
             W_by_ds=np.array([W_by_ds[ds] for ds in datasets]),
             mu_target_by_ds=np.array([mu_target_by_ds[ds] for ds in datasets]),
             C_target_by_ds=np.array([W_target_by_ds[ds] for ds in datasets]),
             random_target=np.array([random_target[ds] for ds in datasets], dtype=object),
             d=d, fit_rows=fit_rows, eval_rows=eval_rows)
    print(f"  saved outputs/coral_{rep}.npz")


if __name__ == "__main__":
    reps = REPS
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    for r in reps:
        run(r)

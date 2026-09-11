"""LEACE: closed-form, minimal-collateral-damage linear concept erasure
(Belrose et al., "LEACE: Perfect Linear Concept Erasure in Closed Form",
NeurIPS 2023), applied to erase `dataset` identity -- the single most
aggressive INLP arm (Step 10's "upper bound") -- so INLP and LEACE can be
compared head to head on the *same* fit/eval split.

Unlike INLP (iterative, greedy, rank-1 per round with a retention guard),
LEACE computes the erasure in one closed-form step that provably minimises the
Frobenius-norm change to the representation among all linear operators that
make the concept linearly undecodable. If it still damages modality transfer
as much as INLP did, the entanglement is not an INLP artefact.

Writes outputs/leace_P_<rep>.npz (same schema as debias.py's npz, factor_order
= ["dataset"]) and prints dims removed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import factor_frame, fit_feature_pipeline  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
TARGET_FACTOR = "dataset"


def leace_projection(X, y, eps=1e-6):
    """Closed-form LEACE projection P (apply as X @ P) erasing `y` from `X`."""
    classes = np.unique(y)
    Z = np.stack([(y == c).astype(float) for c in classes], axis=1)  # one-hot
    Xc = X - X.mean(0)
    Zc = Z - Z.mean(0)
    n = len(X)
    Sigma = (Xc.T @ Xc) / n
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T       # whitening (symmetric)
    Winv = vecs @ np.diag(vals ** 0.5) @ vecs.T      # coloring (inverse)
    cross = (Xc.T @ Zc) / n                          # d x k
    proj = W @ cross                                  # whitened cross-covariance
    U, S, _ = np.linalg.svd(proj, full_matrices=False)
    r = int((S > eps * max(S.max(), eps)).sum())
    Ur = U[:, :r]
    P_tilde = np.eye(X.shape[1]) - Ur @ Ur.T          # erase in whitened space
    P = Winv @ P_tilde @ W                            # unwhiten
    return P, r


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    # reuse the exact fit/eval split INLP used, for a fair comparison
    inlp_npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows, eval_rows = inlp_npz["fit_rows"], inlp_npz["eval_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    in_fit = np.isin(np.arange(len(ff)), fit_rows)
    m = ff[TARGET_FACTOR].notna().to_numpy() & in_fit
    y = ff.loc[m, TARGET_FACTOR].astype(str).to_numpy()
    P, r = leace_projection(Z[m], y)
    print(f"[{rep}] LEACE erased '{TARGET_FACTOR}' ({len(np.unique(y))} classes): "
          f"rank removed = {r}/{d}")

    np.savez(OUT / f"leace_P_{rep}.npz", P=P, dims_removed=np.array([r]),
             factor_order=np.array([TARGET_FACTOR]), d=d,
             fit_rows=fit_rows, eval_rows=eval_rows)
    return r


if __name__ == "__main__":
    reps = REPS
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    for r in reps:
        run(r)

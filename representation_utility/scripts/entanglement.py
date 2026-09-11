"""Direct measurement of why INLP removal backfires: how aligned is each
non-biological factor's discriminative subspace with modality's, in each
representation?

For each representation, on the same fit split used by debias.py: fit a linear
probe for the factor and one for modality, take each probe's weight subspace,
and compute the principal angles between the two subspaces (Björck & Golub).
A small angle (large overlap = cos(angle)) means the two signals live in
close to the same directions, so no linear operator can remove one without
damaging the other -- the direct, operator-free explanation for Step 11's
result.

Writes results/tables/entanglement.csv, results/figures/fig_entanglement.*.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orth, subspace_angles
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import REMOVE_ORDER, RETAIN, factor_frame, fit_feature_pipeline  # noqa: E402
from results_lib import save_fig, save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]


def subspace(Z, y):
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Z, y)
    return orth(np.atleast_2d(clf.coef_).T)


def random_subspace_angle(d, r1, r2, n=200, seed=SEED):
    """Expected min principal angle between two random subspaces of these ranks, as a reference."""
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(n):
        A = orth(rng.normal(size=(d, r1)))
        B = orth(rng.normal(size=(d, r2)))
        mins.append(subspace_angles(A, B)[0])
    return float(np.degrees(np.mean(mins)))


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)
    npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows = npz["fit_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    in_fit = np.isin(np.arange(len(ff)), fit_rows)
    mod_m = ff[RETAIN].notna().to_numpy() & in_fit
    B_mod = subspace(Z[mod_m], ff.loc[mod_m, RETAIN].to_numpy())

    rows = []
    for fac in REMOVE_ORDER:
        m = ff[fac].notna().to_numpy() & in_fit
        y = ff.loc[m, fac].astype(str).to_numpy()
        B_fac = subspace(Z[m], y)
        angles = np.degrees(subspace_angles(B_fac, B_mod))
        ref = random_subspace_angle(d, B_fac.shape[1], B_mod.shape[1])
        rows.append(dict(representation=rep, factor=fac, rank_factor=B_fac.shape[1],
                         rank_modality=B_mod.shape[1], min_angle_deg=float(angles[0]),
                         overlap=float(np.cos(np.radians(angles[0]))),
                         random_ref_angle_deg=ref))
        print(f"  [{rep}] {fac:24s} min angle to modality = {angles[0]:6.1f} deg "
              f"(random subspaces: ~{ref:.1f} deg)  overlap={rows[-1]['overlap']:.2f}")
    return rows


def main():
    all_rows = []
    for rep in REPS:
        print(f"=== {rep} ===")
        all_rows += run(rep)
    df = pd.DataFrame(all_rows)
    save_table(df, "entanglement")
    _figure(df)


def _figure(df):
    import matplotlib.pyplot as plt
    facs = list(dict.fromkeys(df["factor"]))
    reps = REPS
    x = np.arange(len(facs))
    w = 0.8 / len(reps)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, r in enumerate(reps):
        sub = df[df["representation"] == r].set_index("factor").loc[facs]
        ax.bar(x + (j - (len(reps) - 1) / 2) * w, sub["min_angle_deg"], w, label=r)
    ref = df.groupby("factor")["random_ref_angle_deg"].first().loc[facs]
    for i, f in enumerate(facs):
        ax.plot([i - 0.45, i + 0.45], [ref.loc[f]] * 2, "k--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(facs, rotation=15, ha="right")
    ax.set_ylabel("min principal angle to modality subspace (degrees)")
    ax.set_title("Entanglement of each factor's discriminative subspace with modality\n"
                 "(dashed = expected angle between random subspaces of the same rank)")
    ax.legend(title="representation", fontsize=8)
    save_fig(fig, "fig_entanglement")
    plt.close(fig)


if __name__ == "__main__":
    main()

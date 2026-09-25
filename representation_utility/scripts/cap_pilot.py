"""CAP E3: pilot gates -> results/tables/cap_pilot_gates.csv, results/figures/fig_cap_pilot.*

Gate 1 (consistency): per factor, mean pairwise cosine of d_i = z(x') - z(x) over pairs from
different images (levels pooled). Pass if >= 0.5 and above the 99th percentile of the null
(cosines between differences of different factors, different images); needs >= 2 of 4 factors
in both reps.
Gate 2 (signal): S = span of the top-2 uncentred singular vectors of each factor's differences
(rank <= 8). Real offsets = dataset mean minus pool mean over all 4167 originals (17 vectors).
Explained = offset energy inside S. Pass if >= 2x the mean of 1000 random same-rank subspaces and
above their 95th percentile, in at least one rep.
Controls: max |d| for intensity scaling / orientation flip must be < 5% of the median distance
between dataset means, else the gates are not judged. Volumes whose header voxel size disagrees
with the affine are excluded from this check (the existing pipeline reads the header, but
reorientation resets zooms from the affine); their factor pairs are unaffected.

Usage: python cap_pilot.py [--reps untrained medical]
"""
from __future__ import annotations
import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from cap_simulate import CONTROLS, FACTORS  # noqa: E402
from debias import fit_feature_pipeline  # noqa: E402
from results_lib import save_fig, save_table  # noqa: E402

COS_MIN, NULL_Q, G1_MIN_FACTORS = 0.5, 0.99, 2
TOP_K, N_RAND, RAND_Q, RAND_MULT = 2, 1000, 0.95, 2.0
CONTROL_MAX_RATIO = 0.05


def differences(rep):
    """Frame of d_i = z(variant) - z(original) with factor/level/image_id."""
    df = pd.read_parquet(OUT / f"cap_pairs_{rep}.parquet")
    zc = [c for c in df.columns if c.startswith("z_")]
    orig = df[df["factor"] == "original"].set_index("image_id")[zc]
    var = df[df["factor"] != "original"]
    D = var[zc].to_numpy() - orig.loc[var["image_id"], zc].to_numpy()
    return var[["image_id", "dataset", "factor", "level"]].reset_index(drop=True), D


def unit(D):
    n = np.linalg.norm(D, axis=1, keepdims=True)
    return D / np.where(n > 0, n, 1), n.ravel()


def mean_cos(U, ids):
    """Mean cosine over pairs from different images."""
    C = U @ U.T
    diff = ids[:, None] != ids[None, :]
    iu = np.triu(np.ones_like(C, bool), 1) & diff
    return float(C[iu].mean()), int(iu.sum())


def null_cos(U, ids, fac):
    """All cosines between differences of different factors and different images."""
    C = U @ U.T
    m = (fac[:, None] != fac[None, :]) & (ids[:, None] != ids[None, :])
    return C[np.triu(m, 1)]


def header_mismatch(tol=1e-3):
    """Pilot volumes whose header voxel size disagrees with the affine (reorientation changes their zooms)."""
    import nibabel as nib
    pilot = pd.read_csv(OUT / "cap_pilot_ids.csv")
    bad = []
    for i, p in zip(pilot["image_id"], pilot["path"]):
        img = nib.load(p)
        if not np.allclose(img.header.get_zooms()[:3], np.linalg.norm(img.affine[:3, :3], axis=0), atol=tol):
            bad.append(i)
    return bad


def dataset_offsets(rep):
    """17 dataset-mean-minus-pool-mean vectors and dataset means, in the shared z space."""
    idx = pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True)
    emb = pd.read_parquet(OUT / f"emb_{rep}.parquet")
    cols = [c for c in emb.columns if c.startswith("emb_")]
    X = idx[["image_id"]].merge(emb, on="image_id")[cols].to_numpy(float)
    fit_rows = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)["fit_rows"]
    Z, _ = fit_feature_pipeline(X, fit_rows)
    ds = idx["dataset"].to_numpy()
    names = sorted(set(ds))
    means = np.stack([Z[ds == d].mean(0) for d in names])
    return means - Z.mean(0), means


def energy_in(O, Q):
    return float(((O @ Q) ** 2).sum() / (O ** 2).sum())


def run(rep, rng):
    meta, D = differences(rep)
    U, norms = unit(D)
    rows, extra = [], {}

    # controls
    _, means = dataset_offsets(rep)
    med = float(np.median([np.linalg.norm(a - b) for a, b in combinations(means, 2)]))
    bad = meta["image_id"].isin(header_mismatch()).to_numpy()
    for c in CONTROLS:
        m = (meta["factor"] == c).to_numpy()
        r_all = float(norms[m].max() / med)
        r = float(norms[m & ~bad].max() / med)
        rows.append(dict(rep=rep, gate="control", item=c, value=r, threshold=CONTROL_MAX_RATIO,
                         passed=r < CONTROL_MAX_RATIO, n=int((m & ~bad).sum()),
                         note=f"max |d| / median between-dataset distance ({med:.2f}); excludes "
                              f"{int((m & bad).sum())} header/affine voxel-size mismatch volumes "
                              f"(with them: {r_all:.3f})"))

    # gate 1
    fm = meta["factor"].isin(FACTORS).to_numpy() & (norms > 0)
    null = null_cos(U[fm], meta.loc[fm, "image_id"].to_numpy(), meta.loc[fm, "factor"].to_numpy())
    thr = float(np.quantile(null, NULL_Q))
    extra["null"], extra["null_thr"], extra["cos"] = null, thr, {}
    n_pass = 0
    for f in FACTORS:
        m = fm & (meta["factor"] == f).to_numpy()
        mc, npair = mean_cos(U[m], meta.loc[m, "image_id"].to_numpy())
        ok = mc >= COS_MIN and mc > thr
        n_pass += ok
        extra["cos"][f] = mc
        rows.append(dict(rep=rep, gate="G1_consistency", item=f, value=mc, threshold=max(COS_MIN, thr),
                         passed=ok, n=int(m.sum()), note=f"null p99={thr:.3f}; {npair} pairs; levels pooled"))
        for lvl in meta.loc[m, "level"].unique():
            ml = m & (meta["level"] == lvl).to_numpy()
            if ml.sum() > 1:
                v, _ = mean_cos(U[ml], meta.loc[ml, "image_id"].to_numpy())
                rows.append(dict(rep=rep, gate="G1_by_level", item=f"{f}:{lvl}", value=v, threshold=np.nan,
                                 passed=np.nan, n=int(ml.sum()), note="descriptive only"))
    rows.append(dict(rep=rep, gate="G1_consistency", item="GATE (factors passing)", value=n_pass,
                     threshold=G1_MIN_FACTORS, passed=n_pass >= G1_MIN_FACTORS, n=len(FACTORS), note=""))

    # gate 2
    O, _ = dataset_offsets(rep)
    basis = []
    for f in FACTORS:
        m = fm & (meta["factor"] == f).to_numpy()
        _, _, Vt = np.linalg.svd(D[m], full_matrices=False)
        basis.append(Vt[:TOP_K])
        rows.append(dict(rep=rep, gate="G2_by_factor", item=f,
                         value=energy_in(O, np.linalg.qr(Vt[:TOP_K].T)[0]), threshold=np.nan, passed=np.nan,
                         n=int(m.sum()), note=f"offset energy in this factor's top-{TOP_K} alone"))
    Q, s, _ = np.linalg.svd(np.vstack(basis).T, full_matrices=False)
    Q = Q[:, s > 1e-8 * s[0]]
    k, d = Q.shape[1], O.shape[1]
    obs = energy_in(O, Q)
    rand = np.array([energy_in(O, np.linalg.qr(rng.normal(size=(d, k)))[0]) for _ in range(N_RAND)])
    ok = obs >= RAND_MULT * rand.mean() and obs > np.quantile(rand, RAND_Q)
    extra.update(obs=obs, rand=rand)
    rows.append(dict(rep=rep, gate="G2_signal", item=f"explained fraction (rank {k})", value=obs,
                     threshold=max(RAND_MULT * rand.mean(), np.quantile(rand, RAND_Q)), passed=ok, n=len(O),
                     note=f"random mean={rand.mean():.3f}, p95={np.quantile(rand, RAND_Q):.3f}, "
                          f"ratio={obs / rand.mean():.2f}, p={(rand >= obs).mean():.3f}"))
    return rows, extra


def figure(extras, reps):
    fig, axes = plt.subplots(len(reps), 2, figsize=(9, 3.2 * len(reps)), squeeze=False)
    for i, rep in enumerate(reps):
        e = extras[rep]
        ax = axes[i, 0]
        names = [f.replace("_", " ") for f in FACTORS]
        ax.bar(names, [e["cos"][f] for f in FACTORS], color="0.45")
        ax.axhline(COS_MIN, color="k", ls="--", lw=1, label="pass mark 0.5")
        ax.axhline(e["null_thr"], color="tab:red", ls=":", lw=1.2, label="null 99th pct")
        ax.set(ylim=(min(0, min(e["cos"].values()) - 0.05), 1), ylabel="mean pairwise cosine",
               title=f"{rep}: consistency (Gate 1)")
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(axis="x", labelsize=8)
        ax = axes[i, 1]
        ax.hist(e["rand"], bins=40, color="0.75", label="random subspaces")
        ax.axvline(e["obs"], color="k", lw=2, label="factor subspace")
        ax.axvline(RAND_MULT * e["rand"].mean(), color="tab:red", ls="--", lw=1, label="2x random mean")
        ax.set(xlabel="fraction of dataset-offset energy explained", ylabel="count", xlim=(0, None),
               title=f"{rep}: signal (Gate 2)")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig_cap_pilot")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["untrained", "medical"])
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    all_rows, extras = [], {}
    for rep in args.reps:
        rows, extras[rep] = run(rep, rng)
        all_rows += rows
    t = pd.DataFrame(all_rows)
    save_table(t, "cap_pilot_gates")
    figure(extras, args.reps)

    ctrl_ok = t.loc[t["gate"] == "control", "passed"].all()
    g1 = t[(t["gate"] == "G1_consistency") & t["item"].str.startswith("GATE")]["passed"].all()
    g2 = t[t["gate"] == "G2_signal"]["passed"].any()
    pd.set_option("display.width", 200)
    print(t[t["gate"] != "G1_by_level"].to_string(index=False))
    print(f"\ncontrols ok={ctrl_ok}  gate1={g1}  gate2={g2}  ->  "
          f"{'NOT JUDGED (fix controls)' if not ctrl_ok else ('GO' if g1 and g2 else 'NO-GO')}")


if __name__ == "__main__":
    main()

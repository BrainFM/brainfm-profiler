"""Cheap check: does cross-representation embedding-space distance between two
datasets predict their measured pairwise transfer accuracy (pairwise_transfer.csv,
already computed in Step 6 -- no new transfer experiments here)?

Motivated by a proposal to build a "representation shift" taxonomy (persistent /
absorbed / emergent / amplified) across untrained -> medical -> selfsup and use it
to predict transferability. Before investing in that taxonomy, this checks the
one load-bearing claim as cheaply as possible: distances computed from embeddings
already on disk, correlated against transfer numbers already on disk.

Distance is a per-dataset-centroid distance in each representation's standardized
PCA-100 space, normalised by the pooled within-dataset spread (a Cohen's-d-like
effect size: "how many typical within-dataset spreads apart are these two
datasets"), so magnitudes are comparable across representations of different
raw dimensionality (26 / 1024 / 512 / 768). Significance uses a Mantel-style
permutation (relabel the 17 datasets jointly, not per-pair) because the pairs
share nodes and are not independent.

Writes results/tables/geometry_transferability.csv (distance matrices, long form)
and results/tables/geometry_transferability_corr.csv (the actual check).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import features, load_rep  # noqa: E402
from results_lib import save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
N_PERM = 2000


def dataset_centroids_and_spread(X, ds):
    """Per-dataset centroid and pooled within-dataset spread (mean squared
    distance to own centroid, n-weighted) in the given feature space."""
    names = sorted(set(ds))
    cents = {}
    spread_num, spread_den = 0.0, 0
    for name in names:
        m = ds == name
        c = X[m].mean(0)
        cents[name] = c
        spread_num += ((X[m] - c) ** 2).sum()
        spread_den += m.sum()
    pooled_spread = spread_num / spread_den  # trace of pooled within-dataset covariance
    return names, cents, pooled_spread


def distance_matrix(names, cents, pooled_spread):
    n = len(names)
    D = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            D[i, j] = np.linalg.norm(cents[a] - cents[b]) / np.sqrt(pooled_spread)
    return pd.DataFrame(D, index=names, columns=names)


def mantel_perm_test(D, pairs, obs_r, seed=SEED, n_perm=N_PERM):
    """Permute the 17 dataset labels jointly (not per-pair) and recompute the
    correlation each time -- respects the shared-node structure of the pairs."""
    rng = np.random.default_rng(seed)
    names = list(D.index)
    y = pairs["accuracy"].to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(names)
        Dp = pd.DataFrame(D.loc[perm, perm].to_numpy(), index=names, columns=names)
        x = np.array([Dp.loc[s, t] for s, t in zip(pairs["source"], pairs["target"])])
        null[i] = spearmanr(x, y).statistic
    return float((np.abs(null) >= abs(obs_r)).mean())


def main():
    idx = pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True)
    pairs = pd.read_csv(OUT.parent / "results" / "tables" / "pairwise_transfer.csv")
    pairs = pairs[pairs["source"] != pairs["target"]].reset_index(drop=True)  # self-pairs are trivial (n_modalities check inflates them)

    mats = {}
    long_rows = []
    for rep in REPS:
        d, cols = load_rep(rep)
        d = idx[["image_id"]].merge(d, on="image_id")
        X = features(d, cols)
        ds = idx["dataset"].to_numpy()
        names, cents, spread = dataset_centroids_and_spread(X, ds)
        D = distance_matrix(names, cents, spread)
        mats[rep] = D
        for a in names:
            for b in names:
                long_rows.append(dict(representation=rep, dataset_a=a, dataset_b=b, distance=D.loc[a, b]))
        print(f"[{rep}] pooled within-dataset spread={spread:.3f}, d_eff={X.shape[1]}")

    save_table(pd.DataFrame(long_rows), "geometry_transferability")

    # --- the actual check: distance (per representation) vs measured pairwise transfer accuracy
    corr_rows = []
    for rep in REPS:
        D = mats[rep]
        x = np.array([D.loc[s, t] for s, t in zip(pairs["source"], pairs["target"])])
        y = pairs["accuracy"].to_numpy()
        r = spearmanr(x, y).statistic
        p = mantel_perm_test(D, pairs, r)
        corr_rows.append(dict(feature=f"distance_{rep}", spearman_r=r, mantel_p=p, n_pairs=len(pairs)))
        print(f"  distance in {rep:12s} space vs transfer accuracy: rho={r:+.3f}  mantel p={p:.3f}")

    # amplification ratio relative to untrained (the proposal's "learning amplification" quantity)
    for rep in ["handcrafted", "medical", "selfsup"]:
        Dr, D0 = mats[rep], mats["untrained"]
        ratio = Dr / (D0 + 1e-6)
        x = np.array([ratio.loc[s, t] for s, t in zip(pairs["source"], pairs["target"])])
        y = pairs["accuracy"].to_numpy()
        r = spearmanr(x, y).statistic
        p = mantel_perm_test(ratio, pairs, r)
        corr_rows.append(dict(feature=f"amplification_{rep}_over_untrained", spearman_r=r, mantel_p=p, n_pairs=len(pairs)))
        print(f"  amplification ({rep}/untrained) vs transfer accuracy: rho={r:+.3f}  mantel p={p:.3f}")

    save_table(pd.DataFrame(corr_rows), "geometry_transferability_corr")


if __name__ == "__main__":
    main()

"""Does a dataset's transfer isolation follow from cheap acquisition metadata?

Different in kind from DRUS (scripts/dataset_utility.py, Step 7): DRUS combined
7 general-purpose descriptors into one equal-weighted composite score and
validated it against *contribution* (near-zero for almost every dataset --
a target with almost no signal) -- result: rho ~ -0.10, n=17, null.

This script instead uses the *specific* acquisition-outlier properties already
named in cross_dataset_transfer.tex as explaining which datasets are isolated
(unusual/missing template-space code, high native anisotropy, skull-stripped
in a mostly non-stripped pool), and validates directly against transfer_score
/ the isolated flag -- a target with real spread (0.00-1.00).

Reported honestly at n=17: a contingency table is the primary result (does
"isolated" coincide with "has >=1 outlier flag"), per-feature Spearman
correlation against transfer_score is secondary (compared head-to-head with
DRUS's own component correlations), and a leave-one-dataset-out logistic
probe is reported last, explicitly labeled exploratory given n=17.

Writes results/tables/isolation_predictor.csv, results/figures/fig_isolation_predictor.*,
and NUMBERS.md entries.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from analyze_representations import add_qc_factors  # noqa: E402
from debias import factor_frame  # noqa: E402
from results_lib import add_number, save_fig, save_table  # noqa: E402

ISOLATED_THRESHOLD = 0.30  # matches guidelines.py's existing threshold
RARE_COUNT = 50  # matches factor_frame's own rare-orientation-code cutoff

# Already asserted as fact in the compiled dissertation
# (cross_dataset_transfer.tex: "brain-only stripped volumes in a mostly
# non-stripped pool (NFBS, BrainMetShare)"; UPENN-GBM confirmed unstripped
# in PLAN.md). Not a guess -- every other dataset in this corpus is raw
# clinical/research data, not skull-stripped.
SKULL_STRIPPED = {"NFBS": True, "BrainMetShare": True}


def dataset_features(idx: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: the acquisition-outlier features."""
    ff = factor_frame(idx)  # reuses the same rare-orientation-code collapse as debias.py

    global_mode_space = idx["template_space"].mode().iloc[0]

    rows = []
    for ds, g in idx.groupby("dataset"):
        fg = ff[ff["dataset"] == ds]
        rows.append(dict(
            dataset=ds,
            aniso_frac=float((g["anisotropy_cat"] != "Isotropic").mean()),
            template_space_mismatch=float((g["template_space"] != global_mode_space).mean()),
            orientation_rarity=float((fg["orientation"] == "OTHER").mean()),
            skull_stripped=SKULL_STRIPPED.get(ds, False),
        ))
    return pd.DataFrame(rows).set_index("dataset")


def main():
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    feats = dataset_features(idx)

    lodo = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "tables" / "lodo_transfer.csv").set_index("dataset")
    feats["transfer_score"] = lodo["transfer_score"]
    feats["isolated"] = feats["transfer_score"] < ISOLATED_THRESHOLD

    # binary outlier flags (continuous features thresholded at their own
    # corpus-wide upper tercile, except skull_stripped which is already boolean)
    aniso_flag = feats["aniso_frac"] >= feats["aniso_frac"].quantile(2 / 3)
    space_flag = feats["template_space_mismatch"] >= feats["template_space_mismatch"].quantile(2 / 3)
    feats["n_outlier_flags"] = (aniso_flag.astype(int) + space_flag.astype(int)
                                 + feats["skull_stripped"].astype(int))

    out = feats.reset_index().sort_values("transfer_score")
    save_table(out.round(4), "isolation_predictor")

    # 1. primary: contingency table, isolated vs >=1 flag
    has_flag = feats["n_outlier_flags"] >= 1
    tp = int((feats["isolated"] & has_flag).sum())
    fn = int((feats["isolated"] & ~has_flag).sum())
    fp = int((~feats["isolated"] & has_flag).sum())
    tn = int((~feats["isolated"] & ~has_flag).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"contingency (isolated vs >=1 outlier flag): TP={tp} FN={fn} FP={fp} TN={tn}  "
          f"sensitivity={sens:.2f} specificity={spec:.2f}")
    if fn:
        print("  isolated but no flag:", list(feats.index[feats["isolated"] & ~has_flag]))
    if fp:
        print("  flagged but not isolated:", list(feats.index[~feats["isolated"] & has_flag]))
    add_number("isolation predictor contingency",
               f"outlier-flag rule vs measured isolation (n=17): sensitivity={sens:.2f}, "
               f"specificity={spec:.2f} (TP={tp} FN={fn} FP={fp} TN={tn})")

    # 2. secondary: per-feature Spearman vs transfer_score
    corr_rows = []
    for c in ["aniso_frac", "template_space_mismatch", "orientation_rarity", "skull_stripped"]:
        rho, p = spearmanr(feats[c].astype(float), feats["transfer_score"])
        corr_rows.append(dict(feature=c, spearman_rho=rho, p=p))
    corr = pd.DataFrame(corr_rows)
    print(corr.round(3).to_string(index=False))
    save_table(corr.round(4), "isolation_predictor_correlations")

    # 3. tertiary, explicitly exploratory: LODO-CV logistic probe (3 features, no tuning)
    X = StandardScaler().fit_transform(
        feats[["aniso_frac", "template_space_mismatch", "skull_stripped"]].astype(float).to_numpy())
    y = feats["isolated"].to_numpy().astype(int)
    preds = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    loo_acc = float((preds == y).mean())
    majority_acc = float(max(y.mean(), 1 - y.mean()))
    print(f"[exploratory, n=17] LODO-CV logistic (3 features) accuracy={loo_acc:.2f} "
          f"vs majority-class baseline={majority_acc:.2f}")
    add_number("isolation predictor LODO-CV (exploratory)",
               f"3-feature logistic, leave-one-dataset-out accuracy={loo_acc:.2f} "
               f"vs majority-class baseline={majority_acc:.2f} (n=17, illustrative only)")

    _figure(feats)


def _figure(feats):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = np.where(feats["isolated"], "#c44e52", "#4c72b0")
    ax.scatter(feats["n_outlier_flags"] + np.random.default_rng(0).uniform(-0.08, 0.08, len(feats)),
               feats["transfer_score"], c=colors, s=55)
    for ds, r in feats.iterrows():
        ax.annotate(ds, (r["n_outlier_flags"], r["transfer_score"]), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    ax.axhline(ISOLATED_THRESHOLD, color="k", lw=0.8, ls=":", label=f"isolated threshold ({ISOLATED_THRESHOLD})")
    ax.set_xlabel("number of acquisition-outlier flags (0-3)")
    ax.set_ylabel("LODO modality transfer score")
    ax.set_title("Outlier-flag count vs. measured transfer isolation")
    ax.legend(fontsize=8)
    save_fig(fig, "fig_isolation_predictor")
    plt.close(fig)


if __name__ == "__main__":
    main()

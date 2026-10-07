"""Sex as a second content label: LODO transfer, sex-direction consistency, centring and removal.

Only the 10 datasets with sex labels take part (outputs/content_labels.parquet). Inside every
leave-one-dataset-out split (feature space fitted on the 9 training datasets):
  consistency        cosine between the held-out dataset's male-female direction and the training mean;
                     directions are taken inside each sequence and averaged, so sequence mix does not leak in
  pool_consistency   the same score among the training datasets only
  centre_free/oracle label-free and class-balanced centring, as for the sequence
  LEACE, CORAL       dataset-signal removal with a same-rank random projection / wrong-dataset control

Writes results/tables/sex_consistency{,_summary}.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import load_rep  # noqa: E402
from cross_dataset_transfer import fit_tr  # noqa: E402
from leace import leace_projection  # noqa: E402
from removal_eval import random_projection  # noqa: E402
from removal_lodo import coral_maps  # noqa: E402
from results_lib import add_number, save_table  # noqa: E402
from sequence_consistency import ENCODERS, cos  # noqa: E402

MIN_PER_SEX = 3


def sex_clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def sex_direction(X, sex, seq):
    """Male minus female mean, computed inside each sequence and averaged; also the class-balanced centre."""
    diffs, centres = [], []
    for s in np.unique(seq):
        m, f = (seq == s) & (sex == "M"), (seq == s) & (sex == "F")
        if m.sum() >= MIN_PER_SEX and f.sum() >= MIN_PER_SEX:
            diffs.append(X[m].mean(0) - X[f].mean(0))
            centres.append((X[m].mean(0) + X[f].mean(0)) / 2)
    return np.mean(diffs, 0), np.mean(centres, 0)


def bacc(Xtr, ytr, Xte, yte):
    return balanced_accuracy_score(yte, sex_clf().fit(Xtr, ytr).predict(Xte))


def run(rep, lab):
    d, cols = load_rep(rep)
    d = d.merge(lab, on="image_id").dropna(subset=["sex"])
    X = d[cols].to_numpy(float)
    y, ds, seq = d["sex"].to_numpy(), d["dataset"].to_numpy(), d["modality"].to_numpy()
    rows = []
    for held in sorted(set(ds)):
        tr, te = ds != held, ds == held
        Ftr, Fte = fit_tr(X[tr], X[te])
        ytr, yte, dtr = y[tr], y[te], ds[tr]

        dirs = {s: sex_direction(Ftr[dtr == s], ytr[dtr == s], seq[tr][dtr == s]) for s in set(dtr)}
        ref = np.mean([v[0] for v in dirs.values()], 0)
        h_dir, h_c = sex_direction(Fte, yte, seq[te])
        pool = np.mean([cos(dirs[s][0], np.mean([dirs[e][0] for e in dirs if e != s], 0)) for s in dirs])

        Cf, Co = Ftr.copy(), Ftr.copy()
        for s in set(dtr):
            m = dtr == s
            Cf[m] -= Ftr[m].mean(0)
            Co[m] -= dirs[s][1]

        F = np.empty((len(X), Ftr.shape[1]))
        F[tr], F[te] = Ftr, Fte
        sc = StandardScaler().fit(F[tr])
        Z = sc.transform(F)
        P, r = leace_projection(Z[tr], ds[tr])
        coral = coral_maps(Z[tr], ds[tr], Z, ds)
        rem = {"LEACE": (Z @ P, Z @ random_projection(Z.shape[1], r, SEED)),
               "CORAL": (coral(Z, ds, False), coral(Z, ds, True))}

        row = dict(representation=rep, dataset=held, n=int(te.sum()),
                   consistency=cos(h_dir, ref), pool_consistency=pool,
                   before=bacc(Ftr, ytr, Fte, yte),
                   centre_free=bacc(Cf, ytr, Fte - Fte.mean(0), yte),
                   centre_oracle=bacc(Co, ytr, Fte - h_c, yte))
        for k, (after, ctrl) in rem.items():
            row[k] = bacc(after[tr], ytr, after[te], yte)
            row[f"{k}_control"] = bacc(ctrl[tr], ytr, ctrl[te], yte)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    lab = pd.read_parquet(OUT / "content_labels.parquet")[["image_id", "sex"]]
    df = pd.concat([run(k, lab) for k, *_ in ENCODERS], ignore_index=True)
    keys = ["consistency", "pool_consistency", "before", "centre_free", "centre_oracle",
            "LEACE", "LEACE_control", "CORAL", "CORAL_control"]
    rows = []
    for rep, g in df.groupby("representation", sort=False):
        r = dict(representation=rep, **{k: g[k].mean() for k in keys})
        for k in ["centre_free", "centre_oracle", "LEACE", "CORAL"]:
            r[f"p_{k}"] = wilcoxon(g[k], g["before"]).pvalue
        r["rho_within"] = spearmanr(g["consistency"], g["before"])[0]
        rows.append(r)
    s = pd.DataFrame(rows)
    save_table(df, "sex_consistency")
    save_table(s, "sex_consistency_summary")
    print(s.round(3).to_string(index=False))

    rho, p = spearmanr(df.consistency, df.before)
    rho_r, p_r = spearmanr(s.consistency, s.before)
    rho_pool, p_pool = spearmanr(s.pool_consistency, s.before)
    print(f"rep x dataset rho={rho:.2f} p={p:.2g}; mean within-rep rho={s.rho_within.mean():.2f} "
          f"(positive {int((s.rho_within > 0).sum())}/{len(s)}); rep-level rho={rho_r:.2f} p={p_r:.3f}; "
          f"pool rho={rho_pool:.2f} p={p_pool:.3f}")
    add_number("Sex-direction consistency vs LODO sex transfer",
               f"rep x held-out dataset Spearman rho={rho:.2f} (p={p:.1g}, n={len(df)}); mean within-representation "
               f"rho={s.rho_within.mean():.2f}; representation level rho={rho_r:.2f} (p={p_r:.3f}); "
               f"pool consistency rho={rho_pool:.2f} (p={p_pool:.3f})")
    add_number("Sex transfer inside LODO (bAcc; before / centre free / centre oracle / LEACE (ctrl) / CORAL (ctrl))",
               "; ".join(f"{r.representation} {r.before:.2f}/{r.centre_free:.2f}/{r.centre_oracle:.2f}/"
                         f"{r.LEACE:.2f} ({r.LEACE_control:.2f})/{r.CORAL:.2f} ({r.CORAL_control:.2f})"
                         for r in s.itertuples()))


if __name__ == "__main__":
    main()

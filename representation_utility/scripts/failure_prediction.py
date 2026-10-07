"""Label-free prediction of which held-out datasets the sequence probe will fail on.

Inside every leave-one-dataset-out split, three scores are computed without held-out labels and compared
with the measured balanced accuracy on that dataset:
  cluster_consistency  k-means on the held-out features (k = number of sequences, known from file names),
                       cluster offsets matched to the training sequence offsets, mean matched cosine
  avg_confidence       mean maximum softmax probability of the probe
  atc                  average thresholded confidence, threshold from patient-grouped CV on the training data

Writes results/tables/failure_prediction{,_summary}.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import SEED  # noqa: E402
from analyze_representations import load_rep  # noqa: E402
from cross_dataset_transfer import MODS, clf, fit_tr  # noqa: E402
from results_lib import add_number, save_table  # noqa: E402
from sequence_consistency import ENCODERS, cos, offsets  # noqa: E402

SCORES = ["cluster_consistency", "avg_confidence", "atc"]


def cluster_consistency(Xte, k, ref):
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xte)
    off, _ = offsets(Xte, lab)
    S = np.array([[cos(off[c], ref[m]) for m in ref] for c in range(k)])
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


def atc(Xtr, ytr, gtr, Xte, model):
    conf, corr = [], []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr):
        m = clf().fit(Xtr[a], ytr[a])
        p = m.predict_proba(Xtr[b])
        conf.append(p.max(1))
        corr.append(m.classes_[p.argmax(1)] == ytr[b])
    conf, corr = np.concatenate(conf), np.concatenate(corr)
    t = np.quantile(conf, 1 - corr.mean())
    return float((model.predict_proba(Xte).max(1) > t).mean())


def run(rep):
    d, cols = load_rep(rep)
    d = d[d["modality"].isin(MODS)]
    X, y = d[cols].to_numpy(float), d["modality"].to_numpy()
    ds, g = d["dataset"].to_numpy(), d["patient"].to_numpy()
    multi = [s for s in sorted(set(ds)) if len(set(y[ds == s])) > 1]
    rows = []
    for held in multi:
        tr, te = ds != held, ds == held
        Xtr, Xte = fit_tr(X[tr], X[te])
        ytr, yte, dtr = y[tr], y[te], ds[tr]
        model = clf().fit(Xtr, ytr)
        tr_off = [offsets(Xtr[dtr == s], ytr[dtr == s])[0] for s in multi if s != held]
        ref = {m: np.mean([o[m] for o in tr_off if m in o], 0) for m in MODS}
        rows.append(dict(representation=rep, dataset=held,
                         bacc=balanced_accuracy_score(yte, model.predict(Xte)),
                         cluster_consistency=cluster_consistency(Xte, len(set(yte)), ref),
                         avg_confidence=float(model.predict_proba(Xte).max(1).mean()),
                         atc=atc(Xtr, ytr, g[tr], Xte, model)))
    return pd.DataFrame(rows)


def main():
    df = pd.concat([run(k) for k, *_ in ENCODERS], ignore_index=True)
    within = pd.DataFrame({k: [spearmanr(gg[k], gg.bacc)[0] for _, gg in df.groupby("representation")]
                           for k in SCORES}, index=sorted(df.representation.unique()))
    summ = pd.DataFrame([dict(score=k,
                              rho_within_mean=within[k].mean(),
                              rho_pooled=spearmanr(df[k], df.bacc)[0],
                              rho_representation=spearmanr(df.groupby("representation")[k].mean(),
                                                           df.groupby("representation").bacc.mean())[0])
                         for k in SCORES])
    save_table(df, "failure_prediction")
    save_table(summ, "failure_prediction_summary")
    print(within.round(2))
    print(summ.round(3).to_string(index=False))
    tests = []
    for b in ["avg_confidence", "atc"]:
        p = wilcoxon(within["cluster_consistency"], within[b]).pvalue
        n = int((within["cluster_consistency"] > within[b]).sum())
        tests.append(f"vs {b}: higher in {n}/{len(within)} representations, Wilcoxon p={p:.3g}")
        print(tests[-1])
    mae = (df.atc - df.bacc).abs().mean()
    print(f"ATC absolute error as an accuracy estimate: {mae:.3f}")
    add_number("Label-free failure prediction (within-representation Spearman rho with held-out sequence bAcc)",
               "; ".join(f"{r.score} {r.rho_within_mean:.2f} (pooled {r.rho_pooled:.2f}, representation level "
                         f"{r.rho_representation:.2f})" for r in summ.itertuples())
               + "; cluster consistency " + "; ".join(tests) + f"; ATC mean absolute error {mae:.2f}")


if __name__ == "__main__":
    main()

"""E6: LODO sex classification and age regression from frozen embeddings.

Only datasets with labels take part (outputs/content_labels.parquet). Same
feature pipeline as the modality LODO (scaler + PCA-100 fit on the training
datasets). Age is scored three ways because dataset age ranges barely overlap:
MAE, MAE of predicting the training mean, and Pearson r inside the held-out
dataset (ranking, ignores a constant offset).

Writes results/tables/lodo_age_sex.csv, lodo_age_sex_summary.csv.
Usage: python lodo_age_sex.py [--reps untrained medical ...]
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import REPS, load_rep  # noqa: E402
from cross_dataset_transfer import fit_tr  # noqa: E402
from results_lib import add_number, save_table  # noqa: E402

ALPHAS = np.logspace(-1, 4, 11)


def subject(d):
    return d["dataset"] + ":" + d["patient"].str.replace(r"_T\d+$", "", regex=True)


def sex_clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def age_fit_predict(Xtr, ytr, Xte):
    return np.clip(RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte), ytr.min(), ytr.max())


def _age_scores(y, pred, ytr_mean):
    r = pearsonr(y, pred)[0] if np.std(y) > 0 and np.std(pred) > 0 else np.nan
    return dict(mae=mean_absolute_error(y, pred),
                mae_mean_baseline=mean_absolute_error(y, np.full_like(y, ytr_mean)),
                pearson_r=r)


def lodo(d, cols, task):
    y_col = "sex" if task == "sex" else "age"
    d = d.dropna(subset=[y_col])
    X, y, ds = d[cols].to_numpy(float), d[y_col].to_numpy(), d["dataset"].to_numpy()
    rows = []
    for held in sorted(set(ds)):
        tr, te = ds != held, ds == held
        Xtr, Xte = fit_tr(X[tr], X[te])
        if task == "sex":
            pred = sex_clf().fit(Xtr, y[tr]).predict(Xte)
            rows.append(dict(dataset=held, n=int(te.sum()), bacc=balanced_accuracy_score(y[te], pred)))
        else:
            yt = y.astype(float)
            pred = age_fit_predict(Xtr, yt[tr], Xte)
            rows.append(dict(dataset=held, n=int(te.sum()), **_age_scores(yt[te], pred, yt[tr].mean())))
    return pd.DataFrame(rows)


def within(d, cols, task):
    y_col = "sex" if task == "sex" else "age"
    d = d.dropna(subset=[y_col])
    rows = []
    for ds, g in d.groupby("dataset"):
        X, y, grp = g[cols].to_numpy(float), g[y_col].to_numpy(), subject(g).to_numpy()
        if len(set(grp)) < 10 or (task == "sex" and len(set(y)) < 2):
            continue
        if task == "sex":
            split = StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, grp)
        else:
            split = GroupKFold(5).split(X, y, grp)
        preds = np.empty(len(y), dtype=object if task == "sex" else float)
        base = np.empty(len(y))
        for tr, te in split:
            Xtr, Xte = fit_tr(X[tr], X[te])
            if task == "sex":
                preds[te] = sex_clf().fit(Xtr, y[tr]).predict(Xte)
            else:
                preds[te] = age_fit_predict(Xtr, y[tr].astype(float), Xte)
                base[te] = y[tr].astype(float).mean()
        if task == "sex":
            rows.append(dict(dataset=ds, n=len(y), bacc=balanced_accuracy_score(y, preds)))
        else:
            yt = y.astype(float)
            rows.append(dict(dataset=ds, n=len(y), mae=mean_absolute_error(yt, preds),
                             mae_mean_baseline=mean_absolute_error(yt, base),
                             pearson_r=pearsonr(yt, preds)[0]))
    return pd.DataFrame(rows)


def main():
    reps = REPS
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    lab = pd.read_parquet(OUT / "content_labels.parquet")[["image_id", "age", "sex"]]
    lab = lab[lab["age"].notna() | lab["sex"].notna()]

    parts = []
    for r in reps:
        d, cols = load_rep(r)
        d = d.merge(lab, on="image_id")
        for task in ("sex", "age"):
            for setting, fn in (("lodo", lodo), ("within", within)):
                parts.append(fn(d, cols, task).assign(representation=r, task=task, setting=setting))
        print(f"{r} done ({len(d)} labelled volumes)")
    res = pd.concat(parts, ignore_index=True)
    lead = ["representation", "task", "setting", "dataset", "n"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    save_table(res, "lodo_age_sex")

    rows = []
    for (r, task, setting), g in res.groupby(["representation", "task", "setting"], sort=False):
        row = dict(representation=r, task=task, setting=setting, n_datasets=len(g))
        if task == "sex":
            row["bacc_mean"] = g["bacc"].mean()
            row["bacc_min"] = g["bacc"].min()
            row["p_vs_chance"] = wilcoxon(g["bacc"] - 0.5).pvalue if len(g) > 5 else np.nan
        else:
            row["mae_mean"] = g["mae"].mean()
            row["mae_mean_baseline"] = g["mae_mean_baseline"].mean()
            row["n_beats_baseline"] = int((g["mae"] < g["mae_mean_baseline"]).sum())
            row["pearson_r_mean"] = g["pearson_r"].mean()
        rows.append(row)
    summ = pd.DataFrame(rows)
    save_table(summ, "lodo_age_sex_summary")
    print(summ.to_string())
    _numbers(summ)


def _numbers(s):
    def pick(task, setting):
        return s[(s.task == task) & (s.setting == setting)].set_index("representation")

    sl, sw = pick("sex", "lodo"), pick("sex", "within")
    add_number("LODO sex (E6, balanced accuracy, mean over held-out datasets; within-dataset in brackets)",
               "; ".join(f"{r} {sl.loc[r, 'bacc_mean']:.2f} ({sw.loc[r, 'bacc_mean']:.2f})" for r in sl.index)
               + f" (chance 0.50, n={int(sl['n_datasets'].iloc[0])} datasets)")
    al, aw = pick("age", "lodo"), pick("age", "within")
    add_number("LODO age (E6, MAE in years vs predict-training-mean baseline; within-dataset r in brackets)",
               "; ".join(f"{r} {al.loc[r, 'mae_mean']:.1f} vs {al.loc[r, 'mae_mean_baseline']:.1f}, "
                         f"r={al.loc[r, 'pearson_r_mean']:.2f} ({aw.loc[r, 'pearson_r_mean']:.2f})"
                         for r in al.index)
               + f" (n={int(al['n_datasets'].iloc[0])} datasets)")


if __name__ == "__main__":
    main()

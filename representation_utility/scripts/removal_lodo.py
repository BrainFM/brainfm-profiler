"""Dataset-signal removal inside each leave-one-dataset-out fold.

Per held-out dataset: fit the benchmark feature space (scaler + PCA-100) and
each removal method (INLP, LEACE, CORAL, adversarial) and its control on the
other 16 datasets only, then score the benchmark sequence probe on the
held-out dataset. "Before" equals lodo_transfer_fm.csv.

Writes results/tables/removal_lodo.csv (per fold: accuracy, balanced accuracy, macro F1,
dataset and sequence decodability inside the training datasets), removal_lodo_summary.csv,
and outputs/removal_lodo_predictions.parquet (per-scan predictions).
Usage: python removal_lodo.py [--reps 3dino] [--no-decode]; then --merge after one run per rep
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.linalg import matrix_rank
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEVICE, OUT, SEED  # noqa: E402
from adversarial_removal import FeatureHead, train_adversarial  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from coral import fit_dataset_stats, sqrt_inv_sqrt  # noqa: E402
from cross_dataset_transfer import MODS, clf, fit_tr  # noqa: E402
from debias import REMOVE_ORDER, _split, factor_frame, inlp_rounds, nullspace_of, probe_acc  # noqa: E402
from leace import leace_projection  # noqa: E402
from removal_eval import random_projection  # noqa: E402
from results_lib import TABLES, save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup", "brainiac", "brainfm", "3dino", "sammed3d"]
METHODS = ["INLP", "LEACE", "CORAL", "Adversarial"]


def inlp(Z, ff):
    """Graded INLP on the training rows (same settings as debias.run); returns P and rank removed."""
    d = Z.shape[1]
    mm = ff["modality"].notna().to_numpy()
    mod_y, mod_g = ff.loc[mm, "modality"].to_numpy(), ff.loc[mm, "patient"].to_numpy()
    mod_base, _ = probe_acc(*_split(Z[mm], mod_y, mod_g))
    rowspaces = []
    for fac in REMOVE_ORDER:
        m = ff[fac].notna().to_numpy()
        y = ff.loc[m, fac].astype(str).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        inlp_rounds(Z[m], y, ff.loc[m, "patient"].to_numpy(), rowspaces, d, 1.0 / len(np.unique(y)),
                    Z[mm], mod_y, mod_g, mod_base)
    P = nullspace_of(rowspaces, d)
    return P, int(d - matrix_rank(P, tol=1e-6))


def coral_maps(Z_tr, ds_tr, Z_all, ds_all):
    """CORAL transform and its wrong-target control; held-out stats come from its unlabelled rows."""
    datasets = sorted(set(ds_all))
    mu_ref, Sigma_ref, _, _, _ = fit_dataset_stats(Z_tr, ds_tr, sorted(set(ds_tr)))
    _, _, mus, Sigmas, _ = fit_dataset_stats(Z_all, ds_all, datasets)
    W = {ds: sqrt_inv_sqrt(Sigmas[ds])[0] for ds in datasets}
    _, C_ref = sqrt_inv_sqrt(Sigma_ref)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(datasets)
    while any(a == b for a, b in zip(datasets, perm)):
        perm = rng.permutation(datasets)
    target = dict(zip(datasets, perm))
    C_t = {ds: sqrt_inv_sqrt(Sigmas[target[ds]])[1] for ds in datasets}

    def apply(Z, ds, rand):
        out = np.empty_like(Z)
        for k in set(ds):
            m = ds == k
            Zw = (Z[m] - mus[k]) @ W[k]
            out[m] = Zw @ C_t[k] + mus[target[k]] if rand else Zw @ C_ref + mu_ref
        return out
    return apply


def net_map(net):
    net.eval()

    def f(Z):
        with torch.no_grad():
            return net(torch.tensor(Z, dtype=torch.float32, device=DEVICE)).cpu().numpy().astype(Z.dtype)
    return f


def removals(Z, ff, tr):
    """{method: (after, control, dims)} in standardized space, all fitted on training rows."""
    d = Z.shape[1]
    ds = ff["dataset"].to_numpy()
    fft = ff[tr].reset_index(drop=True)
    out = {}

    P, k = inlp(Z[tr], fft)
    out["INLP"] = (Z @ P, Z @ random_projection(d, k, SEED), k)

    P, r = leace_projection(Z[tr], ds[tr])
    out["LEACE"] = (Z @ P, Z @ random_projection(d, r, SEED), r)

    f = coral_maps(Z[tr], ds[tr], Z, ds)
    out["CORAL"] = (f(Z, ds, False), f(Z, ds, True), None)

    ds_le = LabelEncoder().fit(ds[tr])
    mod_le = LabelEncoder().fit(fft["modality"])
    g = train_adversarial(Z[tr], ds_le.transform(ds[tr]), np.ones(tr.sum(), bool),
                          mod_le.transform(fft["modality"]), d, len(ds_le.classes_), len(mod_le.classes_))
    torch.manual_seed(SEED + 1)
    out["Adversarial"] = (net_map(g)(Z), net_map(FeatureHead(d).to(DEVICE))(Z), None)
    return out


def decode(Z, lab, groups):
    """3-fold patient-grouped balanced accuracy of a label inside the training datasets."""
    accs = []
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(Z, lab, groups):
        sc = StandardScaler().fit(Z[a])
        m = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(Z[a]), lab[a])
        accs.append(balanced_accuracy_score(lab[b], m.predict(sc.transform(Z[b]))))
    return float(np.mean(accs))


def run(rep, do_decode=True):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    emb, cols = load_rep(rep)
    X = idx[["image_id"]].merge(emb, on="image_id", how="left")[cols].to_numpy(float)
    keep = ff["modality"].isin(MODS).to_numpy()
    ff, X = ff[keep].reset_index(drop=True), X[keep]
    y, ds, pat = ff["modality"].to_numpy(), ff["dataset"].to_numpy(), ff["patient"].to_numpy()

    rows, preds = [], []
    for held in sorted(set(ds)):
        tr, te = ds != held, ds == held
        Ftr, Fte = fit_tr(X[tr], X[te])
        F = np.empty((len(X), Ftr.shape[1]))
        F[tr], F[te] = Ftr, Fte
        sc = StandardScaler().fit(F[tr])
        Z = sc.transform(F)
        back = lambda A: sc.inverse_transform(A)  # noqa: E731
        k = len(set(y[te]))

        def score(Fx, method, cond, dims=None):
            pred = clf().fit(Fx[tr], y[tr]).predict(Fx[te])
            r = dict(representation=rep, dataset=held, n=int(te.sum()), n_modalities=k, method=method,
                     condition=cond, dims_removed=dims, accuracy=accuracy_score(y[te], pred),
                     balanced_acc=balanced_accuracy_score(y[te], pred) if k > 1 else np.nan,
                     macro_f1=f1_score(y[te], pred, average="macro", labels=sorted(set(y[te]))))
            if do_decode:
                r["dataset_bacc"] = decode(Fx[tr], ds[tr], pat[tr])
                r["sequence_bacc"] = decode(Fx[tr], y[tr], pat[tr])
            rows.append(r)
            preds.append(pd.DataFrame(dict(representation=rep, dataset=held, method=method, condition=cond,
                                           image_id=ff.loc[te, "image_id"].to_numpy(),
                                           true=y[te], pred=pred)))

        score(F, "none", "before")
        for method, (after, ctrl, dims) in removals(Z, ff, tr).items():
            score(back(after), method, "after", dims)
            score(back(ctrl), method, "control", dims)
        b = [r for r in rows if r["dataset"] == held]
        print(f"[{rep}] {held:14s} " + " ".join(
            f"{r['method'][:4]}/{r['condition'][:3]}={r['accuracy']:.2f}" for r in b), flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(df):
    out = []
    base_all = df[df.method == "none"]
    for (rep, method), g in df[df.method != "none"].groupby(["representation", "method"], sort=False):
        base = base_all[base_all.representation == rep].set_index("dataset")
        row = dict(representation=rep, method=method, dims_removed_mean=g.dims_removed.mean())
        multi, single = g[g.n_modalities > 1], g[g.n_modalities == 1]
        for metric in ("balanced_acc", "accuracy", "macro_f1"):
            a = multi[multi.condition == "after"].set_index("dataset")[metric]
            c = multi[multi.condition == "control"].set_index("dataset")[metric]
            b = base.loc[a.index, metric]
            row.update({f"multi_{metric}_before": b.mean(), f"multi_{metric}_after": a.mean(),
                        f"multi_{metric}_control": c.mean(),
                        f"multi_{metric}_p_after_vs_before": wilcoxon(a, b).pvalue if (a != b).any() else 1.0,
                        f"multi_{metric}_p_after_vs_control": wilcoxon(a, c).pvalue if (a != c).any() else 1.0})
        row.update(single_accuracy_before=base.loc[base.n_modalities == 1, "accuracy"].mean(),
                   single_accuracy_after=single[single.condition == "after"].accuracy.mean(),
                   single_accuracy_control=single[single.condition == "control"].accuracy.mean(),
                   all17_accuracy_before=base.accuracy.mean(),
                   all17_accuracy_after=g[g.condition == "after"].accuracy.mean(),
                   all17_accuracy_control=g[g.condition == "control"].accuracy.mean())
        for lab in ("dataset_bacc", "sequence_bacc"):
            if lab in df:
                row.update({f"{lab}_before": base[lab].mean(),
                            f"{lab}_after": g[g.condition == "after"][lab].mean(),
                            f"{lab}_control": g[g.condition == "control"][lab].mean()})
        out.append(row)
    return pd.DataFrame(out)


def merge():
    """Combine per-rep runs (from --reps <one rep>) into the full tables."""
    df = pd.concat([pd.read_csv(TABLES / f"removal_lodo_{r}.csv") for r in REPS], ignore_index=True)
    pd.concat([pd.read_parquet(OUT / f"removal_lodo_predictions_{r}.parquet") for r in REPS],
              ignore_index=True).to_parquet(OUT / "removal_lodo_predictions.parquet")
    save_table(df, "removal_lodo")
    s = summarize(df)
    save_table(s.round(4), "removal_lodo_summary")
    print(s[["representation", "method", "multi_balanced_acc_before", "multi_balanced_acc_after",
             "multi_balanced_acc_control"]].round(3).to_string(index=False))


def main():
    if "--merge" in sys.argv:
        return merge()
    reps = REPS
    if "--reps" in sys.argv:
        reps = [a for a in sys.argv[sys.argv.index("--reps") + 1:] if not a.startswith("--")]
    do_decode = "--no-decode" not in sys.argv
    tag = "_" + "_".join(reps) if "--reps" in sys.argv else ""
    res = [run(r, do_decode) for r in reps]
    df = pd.concat([r[0] for r in res], ignore_index=True)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(OUT / f"removal_lodo_predictions{tag}.parquet")
    save_table(df.round(4), f"removal_lodo{tag}")
    s = summarize(df)
    save_table(s.round(4), f"removal_lodo_summary{tag}")
    print(s[["representation", "method", "multi_balanced_acc_before", "multi_balanced_acc_after",
             "multi_balanced_acc_control"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

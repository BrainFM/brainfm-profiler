"""Benchmark with the new brain FMs: re-run the Chapter 2 analyses for every
representation, reusing the existing functions. Writes *_fm tables only; the
existing tables are left as they are.

  decodability_fm.csv       factor decodability + permutation null (computed only
                            for reps missing from decodability.csv or an earlier
                            decodability_fm.csv, rest copied)
  lodo_transfer_fm.csv      LODO modality accuracy per held-out dataset
  retrieval_purity_fm.csv   k=10 dataset/modality purity
  pairwise_transfer_fm.csv  source -> target modality transfer, per rep
  geometry_transfer_fm.csv  distance vs transfer: against the medical pairwise
                            transfer used in Chapter 2, and against each rep's own
  lodo_age_sex_fm.csv       E6 sex/age LODO + within-dataset (+ _summary)
  benchmark_fm_summary.csv  one row per rep

Usage: python benchmark_fm.py [--reps handcrafted untrained ... 3dino brainfm]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from analyze_representations import (add_qc_factors, build_labels, cv_score, features,  # noqa: E402
                                     load_rep, perm_null)
from cross_dataset_transfer import lodo, pairwise, retrieval  # noqa: E402
from geometry_transferability import (dataset_centroids_and_spread, distance_matrix,  # noqa: E402
                                      mantel_perm_test)
import lodo_age_sex as las  # noqa: E402
from results_lib import TABLES, add_number, save_table  # noqa: E402

BASE = ["handcrafted", "untrained", "medical", "selfsup"]
NEW = ["brainiac", "3dino", "brainfm", "sammed3d", "3dino_rand", "3dino_z", "untrained_pct", "medical_pct"]


def decodability(reps, idx):
    old = pd.read_csv(TABLES / "decodability.csv")
    if (TABLES / "decodability_fm.csv").exists():
        prev = pd.read_csv(TABLES / "decodability_fm.csv")
        old = pd.concat([old, prev[~prev["representation"].isin(old["representation"])]], ignore_index=True)
    keep = old[old["representation"].isin(reps)]
    todo = [r for r in reps if r not in set(old["representation"])]
    factors = build_labels(add_qc_factors(idx))
    rows = []
    for name in todo:
        d, cols = load_rep(name)
        d = idx[["image_id"]].merge(d, on="image_id")
        X_full = features(d, cols)
        for fac, (mask, labels) in factors.items():
            m = np.asarray(sorted(set(mask) & set(np.where(labels.notna())[0])))
            X, y, g = X_full[m], labels.iloc[m].to_numpy(), idx.loc[m, "patient"].to_numpy()
            k = len(np.unique(y))
            acc, acc_sd, auc = cv_score(X, y, g)
            null = perm_null(X, y, g)
            rows.append(dict(representation=name, factor=fac, n=len(y), n_classes=k, chance=1.0 / k,
                             balanced_acc=acc, balanced_acc_sd=acc_sd, macro_auc=auc,
                             null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                             p_value=float((null >= acc).mean())))
            print(f"{name} {fac:22s} bAcc={acc:.3f} null={null.mean():.3f} p={rows[-1]['p_value']:.3f}", flush=True)
    return pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)


def geometry(reps, idx, own_pairs):
    med_pairs = pd.read_csv(TABLES / "pairwise_transfer.csv")
    rows = []
    for rep in reps:
        d, cols = load_rep(rep)
        d = idx[["image_id"]].merge(d, on="image_id")
        X = features(d, cols)
        D = distance_matrix(*dataset_centroids_and_spread(X, idx["dataset"].to_numpy()))
        for target, pairs in (("medical_transfer", med_pairs), ("own_transfer", own_pairs[rep])):
            pairs = pairs[pairs["source"] != pairs["target"]].reset_index(drop=True)
            x = np.array([D.loc[s, t] for s, t in zip(pairs["source"], pairs["target"])])
            r = spearmanr(x, pairs["accuracy"].to_numpy()).statistic
            rows.append(dict(representation=rep, transfer=target, spearman_r=r,
                             mantel_p=mantel_perm_test(D, pairs, r), n_pairs=len(pairs)))
            print(f"{rep} distance vs {target}: rho={r:+.3f} p={rows[-1]['mantel_p']:.3f}", flush=True)
    return pd.DataFrame(rows)


def age_sex(reps):
    lab = pd.read_parquet(OUT / "content_labels.parquet")[["image_id", "age", "sex"]]
    lab = lab[lab["age"].notna() | lab["sex"].notna()]
    parts = []
    for r in reps:
        d, cols = load_rep(r)
        d = d.merge(lab, on="image_id")
        for task in ("sex", "age"):
            for setting, fn in (("lodo", las.lodo), ("within", las.within)):
                parts.append(fn(d, cols, task).assign(representation=r, task=task, setting=setting))
    return pd.concat(parts, ignore_index=True)


def main():
    reps = BASE + [r for r in NEW if (OUT / f"emb_{r}.parquet").exists()]
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    print("reps:", reps)
    idx = pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True)

    lodo_rows, ret_rows, own_pairs = [], [], {}
    for r in reps:
        d, cols = load_rep(r)
        lodo_rows.append(lodo(d, cols).assign(representation=r))
        ret_rows.append(retrieval(d, cols).assign(representation=r))
        own_pairs[r] = pairwise(d, cols).assign(representation=r)
        print(f"{r}: LODO modality mean {lodo_rows[-1]['accuracy'].mean():.3f}", flush=True)
    lodo_df = pd.concat(lodo_rows, ignore_index=True)
    ret_df = pd.concat(ret_rows, ignore_index=True)
    save_table(lodo_df, "lodo_transfer_fm")
    save_table(ret_df, "retrieval_purity_fm")
    save_table(pd.concat(own_pairs.values(), ignore_index=True), "pairwise_transfer_fm")

    geo = geometry(reps, idx, own_pairs)
    save_table(geo, "geometry_transfer_fm")

    asx = age_sex(reps)
    save_table(asx, "lodo_age_sex_fm")

    dec = decodability(reps, idx)
    save_table(dec, "decodability_fm")

    summ = summary(reps, dec, lodo_df, ret_df, geo, asx)
    save_table(summ, "benchmark_fm_summary")
    print(summ.to_string())
    _numbers(summ)


def summary(reps, dec, lodo_df, ret_df, geo, asx):
    rows = []
    for r in reps:
        dv = dec[dec["representation"] == r].set_index("factor")["balanced_acc"]
        lo = lodo_df[lodo_df["representation"] == r]
        rt = ret_df[ret_df["representation"] == r]
        gm = geo[(geo["representation"] == r) & (geo["transfer"] == "medical_transfer")].iloc[0]
        go = geo[(geo["representation"] == r) & (geo["transfer"] == "own_transfer")].iloc[0]
        a = asx[asx["representation"] == r]
        sex_l = a[(a.task == "sex") & (a.setting == "lodo")]["bacc"].mean()
        sex_w = a[(a.task == "sex") & (a.setting == "within")]["bacc"].mean()
        age_l = a[(a.task == "age") & (a.setting == "lodo")]
        age_w = a[(a.task == "age") & (a.setting == "within")]
        rows.append(dict(
            representation=r,
            dataset_bacc=dv.get("dataset"), modality_bacc=dv.get("modality (no PD)"),
            lodo_modality_mean=lo["accuracy"].mean(), lodo_modality_min=lo["accuracy"].min(),
            n_below_030=int((lo["accuracy"] < 0.3).sum()),
            dataset_purity=rt["dataset_purity"].mean(), modality_purity=rt["modality_purity"].mean(),
            geo_rho_medical=gm["spearman_r"], geo_p_medical=gm["mantel_p"],
            geo_rho_own=go["spearman_r"], geo_p_own=go["mantel_p"],
            sex_lodo=sex_l, sex_within=sex_w,
            age_mae_lodo=age_l["mae"].mean(), age_mae_baseline=age_l["mae_mean_baseline"].mean(),
            age_r_lodo=age_l["pearson_r"].mean(), age_r_within=age_w["pearson_r"].mean()))
    return pd.DataFrame(rows)


def _numbers(s):
    s = s.set_index("representation")
    fmt = lambda col, f="{:.2f}": "; ".join(f"{r} " + f.format(s.loc[r, col]) for r in s.index)  # noqa: E731
    add_number("FM benchmark: dataset decodability (bAcc, chance 0.06)", fmt("dataset_bacc"))
    add_number("FM benchmark: modality decodability (bAcc, no PD, chance 0.25)", fmt("modality_bacc"))
    add_number("FM benchmark: LODO modality transfer (mean accuracy)", fmt("lodo_modality_mean"))
    add_number("FM benchmark: retrieval dataset purity (k=10, chance 0.06)", fmt("dataset_purity"))
    add_number("FM benchmark: distance vs medical pairwise transfer (Spearman rho)", fmt("geo_rho_medical", "{:+.2f}"))
    add_number("FM benchmark: LODO sex bAcc (within-dataset in the E6 table)", fmt("sex_lodo"))
    add_number("FM benchmark: LODO age r inside held-out dataset", fmt("age_r_lodo"))


if __name__ == "__main__":
    main()

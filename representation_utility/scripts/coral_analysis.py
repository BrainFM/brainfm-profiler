"""Evaluate CORAL distribution alignment with the same protocol as INLP/LEACE
(decodability + permutation null, LODO modality transfer vs. a random-target
control, retrieval purity, non-linear residual probe), via removal_eval.py's
callable-based Transform interface -- CORAL is a dataset-conditional affine
map, not a single global projection matrix, so it cannot reuse
debias_analysis.py's `Z @ P` functions directly.

Writes results/tables/coral_{decodability,transfer,retrieval,residual}.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import RETAIN, REMOVE_ORDER, factor_frame, fit_feature_pipeline  # noqa: E402
from removal_eval import decodability_perm, lodo_transfer, nonlinear_residual, retrieval_purity  # noqa: E402
from results_lib import save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]


def load_coral(rep):
    npz = np.load(OUT / f"coral_{rep}.npz", allow_pickle=True)
    datasets = list(npz["datasets"])
    idx_map = {ds: i for i, ds in enumerate(datasets)}
    mu_ref, C_ref = npz["mu_ref"], npz["C_ref"]
    mu_by_ds, W_by_ds = npz["mu_by_ds"], npz["W_by_ds"]
    mu_target_by_ds, C_target_by_ds = npz["mu_target_by_ds"], npz["C_target_by_ds"]

    def correct_transform(Z, ds):
        out = np.empty_like(Z)
        for name in np.unique(ds):
            i = idx_map[name]
            m = ds == name
            out[m] = (Z[m] - mu_by_ds[i]) @ W_by_ds[i] @ C_ref + mu_ref
        return out

    def random_target_transform(Z, ds):
        out = np.empty_like(Z)
        for name in np.unique(ds):
            i = idx_map[name]
            m = ds == name
            out[m] = (Z[m] - mu_by_ds[i]) @ W_by_ds[i] @ C_target_by_ds[i] + mu_target_by_ds[i]
        return out

    return correct_transform, random_target_transform, npz["fit_rows"], npz["eval_rows"], int(npz["d"])


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    transform, transform_rand, fit_rows, eval_rows, d = load_coral(rep)
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)

    print(f"\n=== {rep} (CORAL) ===")
    print("  decodability (before/after/null):")
    dec_rows = decodability_perm(Z, ff, eval_rows, transform, [RETAIN, *REMOVE_ORDER])
    for r in dec_rows:
        r["representation"] = rep

    lodo = lodo_transfer(Z, ff, transform, transform_rand)
    lodo["representation"] = rep
    print(f"  LODO modality transfer: before={lodo[lodo.condition=='before'].accuracy.mean():.3f} "
          f"after={lodo[lodo.condition=='after'].accuracy.mean():.3f} "
          f"random-target-control={lodo[lodo.condition=='random'].accuracy.mean():.3f}")

    ret = retrieval_purity(Z, ff, transform)
    ret["representation"] = rep
    print(f"  retrieval purity: {ret.to_dict('records')}")

    res = nonlinear_residual(Z, ff, eval_rows, transform)
    res["representation"] = rep
    print(f"  non-linear residual (dataset, MLP): "
          f"before={res[res.condition=='before'].balanced_acc.iloc[0]:.3f} "
          f"after={res[res.condition=='after'].balanced_acc.iloc[0]:.3f}")

    return dec_rows, lodo, ret, res


def main():
    all_dec, all_lodo, all_ret, all_res = [], [], [], []
    for rep in REPS:
        dec, lodo, ret, res = run(rep)
        all_dec += dec
        all_lodo.append(lodo)
        all_ret.append(ret)
        all_res.append(res)

    save_table(pd.DataFrame(all_dec), "coral_decodability")
    save_table(pd.concat(all_lodo, ignore_index=True), "coral_transfer")
    save_table(pd.concat(all_ret, ignore_index=True), "coral_retrieval")
    save_table(pd.concat(all_res, ignore_index=True), "coral_residual")


if __name__ == "__main__":
    main()

"""Compare INLP vs LEACE vs a matched-rank random projection for erasing
`dataset` identity, on the same fit/eval split, for every representation.

Writes results/tables/leace_comparison.csv.
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
from debias import factor_frame, fit_feature_pipeline  # noqa: E402
from debias_analysis import decodability_perm, lodo_transfer, random_projection  # noqa: E402
from results_lib import save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]


def load_P(rep, tag):
    npz = np.load(OUT / f"{tag}_P_{rep}.npz", allow_pickle=True)
    return npz["P"], npz["fit_rows"], npz["eval_rows"], int(npz["d"])


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    P_inlp, fit_rows, eval_rows, d = load_P(rep, "debias")
    P_leace, _, _, _ = load_P(rep, "leace")
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)

    k_leace = d - np.linalg.matrix_rank(P_leace, tol=1e-6)
    k_inlp = d - np.linalg.matrix_rank(P_inlp, tol=1e-6)
    P_rand16 = random_projection(d, max(int(k_leace), 1), 0)

    rows = []
    for tag, P in (("INLP", P_inlp), ("LEACE", P_leace), (f"random(k={k_leace})", P_rand16)):
        dec = decodability_perm(Z, ff, eval_rows, P, ["dataset", "modality"])
        lodo = lodo_transfer(Z, ff, P, np.eye(d))
        rows.append(dict(representation=rep, method=tag,
                         dims_removed=int(d - np.linalg.matrix_rank(P, tol=1e-6)),
                         dataset_after=dec[0]["after"], modality_after=dec[1]["after"],
                         lodo_before=lodo[lodo.condition == "before"].accuracy.mean(),
                         lodo_after=lodo[lodo.condition == "after"].accuracy.mean()))
        print(f"  [{rep}] {tag:16s} k={rows[-1]['dims_removed']:3d}  "
              f"dataset={rows[-1]['dataset_after']:.2f}  modality={rows[-1]['modality_after']:.2f}  "
              f"LODO {rows[-1]['lodo_before']:.2f}->{rows[-1]['lodo_after']:.2f}")
    return rows


def main():
    all_rows = []
    for rep in REPS:
        print(f"=== {rep} ===")
        all_rows += run(rep)
    save_table(pd.DataFrame(all_rows), "leace_comparison")


if __name__ == "__main__":
    main()

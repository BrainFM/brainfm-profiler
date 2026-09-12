"""Consolidate all four removal/alignment methods (INLP, LEACE, CORAL,
adversarial) into one table: one row per (representation x method), LODO
modality transfer before/after/random-control. The single table anchoring an
expanded write-up of "what happens when you try to remove the acquisition
confound," across two linear methods, one distribution-alignment method, and
one non-linear method.

Writes results/tables/removal_comparison.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from results_lib import RESULTS, save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]


def _agg(df, rep, method, dims_removed, method_type):
    d = df[df.representation == rep]
    return dict(
        representation=rep, method=method, method_type=method_type,
        dims_removed=dims_removed,
        lodo_before=float(d[d.condition == "before"].accuracy.mean()),
        lodo_after=float(d[d.condition == "after"].accuracy.mean()),
        lodo_random_control=float(d[d.condition == "random"].accuracy.mean()),
    )


def main():
    tables = RESULTS / "tables"
    debias_t = pd.read_csv(tables / "debias_transfer.csv")
    coral_t = pd.read_csv(tables / "coral_transfer.csv")
    adv_t = pd.read_csv(tables / "adversarial_transfer.csv")

    rows = []
    for rep in REPS:
        npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
        dims_removed = int(max(dict(zip(npz["factor_order"], npz["dims_removed"])).values()))
        rows.append(_agg(debias_t, rep, "INLP", dims_removed, "linear (iterative, retention-guarded)"))

        leace_npz = np.load(OUT / f"leace_P_{rep}.npz", allow_pickle=True)
        d_full = int(leace_npz["d"])
        k_leace = d_full - np.linalg.matrix_rank(leace_npz["P"], tol=1e-6)
        # leace_analysis.py's own comparison already has before/after/random for LEACE
        leace_cmp = pd.read_csv(tables / "leace_comparison.csv")
        r = leace_cmp[(leace_cmp.representation == rep) & (leace_cmp.method == "LEACE")].iloc[0]
        rand_row = leace_cmp[(leace_cmp.representation == rep) & (leace_cmp.method.str.startswith("random"))].iloc[0]
        rows.append(dict(representation=rep, method="LEACE", method_type="linear (closed-form, minimal-damage)",
                         dims_removed=int(k_leace), lodo_before=float(r.lodo_before),
                         lodo_after=float(r.lodo_after), lodo_random_control=float(rand_row.lodo_after)))

        rows.append(_agg(coral_t, rep, "CORAL", None, "affine (distribution alignment, full-rank)"))
        rows.append(_agg(adv_t, rep, "Adversarial (DANN)", None, "non-linear (learned, full-rank)"))

    out = pd.DataFrame(rows)
    save_table(out.round(4), "removal_comparison")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Dataset-signal removal (INLP, LEACE, CORAL, adversarial) on the broad FMs.

Fits each method for the new reps with the existing run() functions (these only
write per-rep files: debias_P_/leace_P_/coral_<rep>.npz, adversarial_<rep>.pt),
then scores them with the existing analysis functions: same fit/eval split,
same random controls, same LODO protocol as removal_comparison.csv.
`--check` re-scores an old rep from its existing files without refitting, to
confirm the numbers match removal_comparison.csv.

Writes results/tables/removal_fm.csv only.
Usage: python removal_fm.py [--reps 3dino sammed3d] [--check medical]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
import adversarial_removal  # noqa: E402
import coral  # noqa: E402
import coral_analysis  # noqa: E402
import debias  # noqa: E402
import debias_analysis  # noqa: E402
import leace  # noqa: E402
import leace_analysis  # noqa: E402
from results_lib import TABLES, add_number, save_table  # noqa: E402

MULTI = None


def _multi_datasets():
    global MULTI
    if MULTI is None:
        l = pd.read_csv(TABLES / "lodo_transfer_fm.csv")
        MULTI = set(l.loc[l["n_modalities"] >= 2, "dataset"])
    return MULTI


def _dataset_after(dec_rows):
    r = next(r for r in dec_rows if r["factor"] == "dataset")
    return r["before"], r["after"]


def _row(rep, method, lodo, dims, dec_rows=None):
    by = lambda c: lodo[lodo.condition == c]  # noqa: E731
    multi = lodo[lodo["dataset"].isin(_multi_datasets())]
    mb = lambda c: multi[multi.condition == c].accuracy.mean()  # noqa: E731
    row = dict(representation=rep, method=method, dims_removed=dims,
               lodo_before=by("before").accuracy.mean(), lodo_after=by("after").accuracy.mean(),
               lodo_random_control=by("random").accuracy.mean(),
               multi_before=mb("before"), multi_after=mb("after"), multi_random_control=mb("random"))
    if dec_rows is not None:
        row["dataset_bacc_before"], row["dataset_bacc_after"] = _dataset_after(dec_rows)
    return row


def fit(rep):
    debias.run(rep)
    leace.run(rep)
    coral.run(rep)


def score(rep, adversarial_fit=True):
    rows = []
    dec, lodo, _, _, k = debias_analysis.analyze_rep(rep)
    rows.append(_row(rep, "INLP", lodo, k, dec))

    lr = leace_analysis.run(rep)
    le = next(r for r in lr if r["method"] == "LEACE")
    rnd = next(r for r in lr if r["method"].startswith("random"))
    rows.append(dict(representation=rep, method="LEACE", dims_removed=le["dims_removed"],
                     lodo_before=le["lodo_before"], lodo_after=le["lodo_after"],
                     lodo_random_control=rnd["lodo_after"],
                     dataset_bacc_after=le["dataset_after"]))

    dec, lodo, _, _ = coral_analysis.run(rep)
    rows.append(_row(rep, "CORAL", lodo, None, dec))

    if adversarial_fit:
        dec, lodo, _, _ = adversarial_removal.run(rep)
        rows.append(_row(rep, "Adversarial (DANN)", lodo, None, dec))
    return rows


def main():
    reps = ["3dino", "sammed3d"]
    if "--reps" in sys.argv:
        reps = [a for a in sys.argv[sys.argv.index("--reps") + 1:] if not a.startswith("--")]
    check = []
    if "--check" in sys.argv:
        check = [sys.argv[sys.argv.index("--check") + 1]]

    rows = []
    for rep in check:
        rows += [dict(r, note="check (existing fit)") for r in score(rep, adversarial_fit=False)]
    for rep in reps:
        fit(rep)
        rows += score(rep)
    out = pd.DataFrame(rows)
    save_table(out.round(4), "removal_fm")
    print(out.round(3).to_string(index=False))

    new = out[out["representation"].isin(reps)]
    add_number("Removal on broad FMs (LODO modality accuracy, before -> after, random control)",
               "; ".join(f"{r.representation} {r.method} {r.lodo_before:.2f}->{r.lodo_after:.2f} "
                         f"(control {r.lodo_random_control:.2f})" for r in new.itertuples()))


if __name__ == "__main__":
    main()

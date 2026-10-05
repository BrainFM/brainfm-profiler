"""Within-dataset sequence decodability for every benchmark representation.

Writes results/tables/within_dataset_fm.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_representations import load_rep  # noqa: E402
from cross_dataset_transfer import within  # noqa: E402
from results_lib import save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup", "brainiac", "brainfm", "3dino", "sammed3d",
        "3dino_rand", "3dino_z", "medical_pct"]


def main():
    rows = []
    for r in REPS:
        d, cols = load_rep(r)
        rows.append(within(d, cols).assign(representation=r))
        print(r, round(rows[-1].within_bacc.mean(), 3), flush=True)
    save_table(pd.concat(rows, ignore_index=True).round(4), "within_dataset_fm")


if __name__ == "__main__":
    main()

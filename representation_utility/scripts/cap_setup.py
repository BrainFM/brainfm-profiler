"""CAP E0: pilot subset (outputs/cap_pilot_ids.csv) and factor levels (configs/cap_levels.yaml).

Usage: python cap_setup.py [--per-dataset 5] [--noise-sample 20] [--workers 6]
"""
from __future__ import annotations
import argparse
import ast
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, ROOT, SEED  # noqa: E402
from io_utils import load_volume  # noqa: E402

LEVELS = {"mild": 0.25, "moderate": 0.50, "strong": 0.90}
THICK_MIN_MM = 1.5  # thick-slice subset used for the slice-thickness levels


def pilot_ids(idx, per_dataset):
    """Pick `per_dataset` patients per dataset, then one volume per patient."""
    rng = np.random.default_rng(SEED)
    rows = []
    for ds in sorted(idx["dataset"].unique()):
        d = idx[idx["dataset"] == ds]
        pats = rng.choice(sorted(d["patient"].unique()), per_dataset, replace=False)
        for p in pats:
            ids = sorted(d.loc[d["patient"] == p, "image_id"])
            rows.append(rng.choice(ids))
    keep = idx[idx["image_id"].isin(rows)]
    return keep[["image_id", "dataset", "patient", "modality", "path", "voxel_spacing"]].sort_values(
        ["dataset", "image_id"]).reset_index(drop=True)


def noise_sigma_rel(path):
    """Noise std relative to tissue mean, from the residual after a 3x3x3 mean filter."""
    x, _ = load_volume(path)
    tissue = x > 0.2 * np.percentile(x[x > 0], 99)
    tissue = ndimage.binary_erosion(tissue, iterations=1)
    r = x - ndimage.uniform_filter(x, 3)
    rt = r[tissue]
    mad = np.median(np.abs(rt - np.median(rt))) * 1.4826
    return float(mad / np.sqrt(26 / 27) / x[tissue].mean())


def _noise_worker(args):
    image_id, path = args
    try:
        return image_id, noise_sigma_rel(path)
    except Exception:  # noqa: BLE001
        return image_id, np.nan


def pct(s):
    return {k: round(float(s.quantile(q)), 4) for k, q in LEVELS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=5)
    ap.add_argument("--noise-sample", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    idx = pd.read_parquet(OUT / "file_index.parquet")
    pilot = pilot_ids(idx, args.per_dataset)
    pilot.to_csv(OUT / "cap_pilot_ids.csv", index=False)
    print(f"cap_pilot_ids.csv: {len(pilot)} volumes, {pilot['patient'].nunique()} patients, "
          f"{pilot['dataset'].nunique()} datasets")

    thick = idx["voxel_spacing"].apply(lambda s: max(ast.literal_eval(s)))
    thick_sub = thick[thick > THICK_MIN_MM]

    qc = pd.read_parquet(OUT / "qc_factors.parquet")

    sample = pd.concat([d.sample(min(len(d), args.noise_sample), random_state=SEED)
                        for _, d in idx.groupby("dataset")])
    with ProcessPoolExecutor(args.workers) as ex:
        noise = dict(ex.map(_noise_worker, zip(sample["image_id"], sample["path"])))
    sample["noise_rel"] = sample["image_id"].map(noise)
    sample[["image_id", "dataset", "noise_rel"]].to_csv(OUT / "cap_noise_estimates.csv", index=False)
    print("noise_rel per dataset (median):",
          sample.groupby("dataset")["noise_rel"].median().round(4).to_dict())

    cfg = {
        "created": "2026-09-25",
        "percentiles": LEVELS,
        "slice_thickness": {
            "unit": "mm, target slice thickness along the slice axis",
            "source": f"file_index voxel_spacing, thickest side, volumes with thickest side > {THICK_MIN_MM} mm "
                      f"(n={len(thick_sub)}); the full-corpus percentiles are "
                      f"{pct(thick)} and would be no-ops (user decision 2026-09-25)",
            "levels": pct(thick_sub),
        },
        "bias_field": {
            "unit": "coefficient of variation of the multiplicative field over the foreground",
            "source": f"qc_factors.parquet biasfield (n={len(qc)})",
            "levels": pct(qc["biasfield"]),
        },
        "noise": {
            "unit": "Rician sigma as a fraction of the tissue mean intensity",
            "source": f"residual-MAD noise estimate on {sample['noise_rel'].notna().sum()} volumes "
                      f"({args.noise_sample} per dataset, seed {SEED}), cap_noise_estimates.csv",
            "levels": pct(sample["noise_rel"].dropna()),
        },
        "skull_strip": {
            "unit": "HD-BET brain mask applied (single level); only volumes with skull present",
            "source": "applies when > 5% of non-zero voxels lie outside the HD-BET mask",
            "levels": {"full": 1.0},
        },
        "ghosting": {
            "unit": "torchio Ghosting intensity (4 ghosts, phase axis A-P); not in the pilot",
            "source": "not calibrated yet; qc EFC percentiles recorded for E4",
            "efc_percentiles": pct(qc["efc"]),
            "levels": {"mild": 0.2, "moderate": 0.4, "strong": 0.8},
        },
        "controls": {"intensity_scale": 3.7, "orientation_flip": "flip axis 0 + swap axes 0/1, affine updated"},
    }
    path = ROOT / "configs" / "cap_levels.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    print(f"wrote {path}")
    for f in ("slice_thickness", "bias_field", "noise"):
        print(f"  {f:16s} {cfg[f]['levels']}")


if __name__ == "__main__":
    main()

"""handcrafted: per-volume image statistics -> outputs/emb_handcrafted.parquet.

Usage: python features_r1.py [--workers N]
"""
from __future__ import annotations
import ast
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, LOG  # noqa: E402
from io_utils import center_crop_pad, load_volume, resample_iso  # noqa: E402

PCTS = [1, 5, 25, 50, 75, 95, 99]
OUT_PATH = OUT / "emb_handcrafted.parquet"


def features(path, meta):
    data, zooms = load_volume(path)
    v = center_crop_pad(resample_iso(data, zooms))
    fg = v[v != 0]
    if fg.size < 100:
        raise ValueError("almost no foreground")

    p = np.percentile(fg, PCTS)
    hist, _ = np.histogram(fg, bins=64)
    hist = hist[hist > 0] / hist.sum()
    entropy = float(-(hist * np.log(hist)).sum())

    gx, gy, gz = np.gradient(v)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)[v != 0]
    lap = ndimage.laplace(v)[v != 0]

    sx, sy, sz = (float(x) for x in ast.literal_eval(meta["voxel_spacing"]))
    nx, ny, nz = (int(x) for x in ast.literal_eval(meta["shape"]))

    return {
        "image_id": meta["image_id"],
        **{f"p{q}": float(p[i]) for i, q in enumerate(PCTS)},
        "mean": float(fg.mean()), "std": float(fg.std()),
        "skew": float(stats.skew(fg)), "kurtosis": float(stats.kurtosis(fg)),
        "entropy": entropy,
        "iqr": float(p[4] - p[2]), "p95_p50": float(p[5] / (p[3] + 1e-6)),
        "fg_frac": float((v != 0).mean()),
        "grad_mean": float(gmag.mean()), "grad_std": float(gmag.std()),
        "lap_var": float(lap.var()),
        "orig_sx": sx, "orig_sy": sy, "orig_sz": sz,
        "aniso_ratio": float(meta["anisotropy_ratio"]),
        "orig_nx": nx, "orig_ny": ny, "orig_nz": nz,
        "orig_nvox": nx * ny * nz,
    }


def _worker(args):
    path, meta = args
    try:
        return features(path, meta)
    except Exception as e:  # noqa: BLE001
        return {"image_id": meta["image_id"], "_error": f"{type(e).__name__}: {e}"}


def main(workers=4):
    idx = pd.read_parquet(OUT / "file_index.parquet")
    done = set()
    if OUT_PATH.exists():
        done = set(pd.read_parquet(OUT_PATH)["image_id"])
    todo = idx[~idx["image_id"].isin(done)]
    print(f"{len(todo)} volumes to process ({len(done)} already done), {workers} workers")

    jobs = [(r["path"], r.to_dict()) for _, r in todo.iterrows()]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_worker, jobs), 1):
            (errors if "_error" in res else rows).append(res)
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}  (errors: {len(errors)})")

    if rows:
        new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            new = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True)
        new = idx[["image_id", "dataset", "domain", "patient", "modality"]].merge(
            new, on="image_id", how="inner")
        new.to_parquet(OUT_PATH, index=False)

    if errors:
        (LOG / "r1_errors.txt").write_text(
            "\n".join(f"{e['image_id']}\t{e['_error']}" for e in errors) + "\n")

    final = pd.read_parquet(OUT_PATH)
    print(f"\nemb_handcrafted.parquet: {len(final)} rows / {len(idx)} index rows, "
          f"{final.isna().any(axis=1).sum()} rows with NaN, {len(errors)} errors")


if __name__ == "__main__":
    w = 4
    if "--workers" in sys.argv:
        w = int(sys.argv[sys.argv.index("--workers") + 1])
    main(w)

"""Per-scan non-biological QC factors -> outputs/qc_factors.parquet.

Three factors, all per scan, no documentation needed:
  template_space  - NIfTI sform/qform code (native / aligned / mni / talairach / other)
  biasfield       - low-frequency intensity non-uniformity (CV of a smoothed foreground field)
  efc             - Entropy Focus Criterion on the foreground (motion / ghosting proxy)

biasfield and efc are continuous; each is also bucketed into corpus-wide terciles
(*_cat = low / medium / high) for the categorical probes in Step 5.

Usage: python qc_factors.py [--workers N]
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, LOG  # noqa: E402
from io_utils import center_crop_pad, load_volume, resample_iso  # noqa: E402

OUT_PATH = OUT / "qc_factors.parquet"

_SPACE = {0: "unknown", 1: "native", 2: "aligned", 3: "talairach", 4: "mni", 5: "other"}
BIAS_SIGMA_MM = 32.0  # low-frequency scale for the bias-field estimate
TARGET_MM = 2.0


def template_space(path):
    """sform takes precedence over qform (NIfTI convention); fall back to qform.

    ANALYZE (.hdr/.img) headers have no sform/qform code -> 'unknown'.
    """
    hdr = nib.load(str(path)).header
    try:
        s, q = int(hdr["sform_code"]), int(hdr["qform_code"])
    except (KeyError, ValueError):
        return "unknown"
    code = s if s > 0 else q
    return _SPACE.get(code, "other")


def bias_field_cv(v, mask):
    """CV of a heavily smoothed foreground intensity field. Higher = stronger inhomogeneity."""
    sigma = BIAS_SIGMA_MM / TARGET_MM
    num = ndimage.gaussian_filter(np.where(mask, v, 0.0), sigma)
    den = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    smooth = num / (den + 1e-6)
    s = smooth[mask]
    return float(s.std() / (s.mean() + 1e-6))


def efc(fg):
    """Entropy Focus Criterion (Atkinson 1997; MRIQC), foreground voxels only."""
    n = fg.size
    efc_max = n * (1.0 / np.sqrt(n)) * np.log(1.0 / np.sqrt(n))
    b_max = np.sqrt((fg ** 2).sum())
    r = fg / b_max
    return float((1.0 / efc_max) * np.sum(r * np.log(r)))


def compute(path, image_id):
    data, zooms = load_volume(path)
    v = center_crop_pad(resample_iso(data, zooms))
    mask = v > 0
    fg = v[mask]
    if fg.size < 100:
        raise ValueError("almost no foreground")
    return {
        "image_id": image_id,
        "template_space": template_space(path),
        "biasfield": bias_field_cv(v, mask),
        "efc": efc(fg),
    }


def _worker(args):
    path, image_id = args
    try:
        return compute(path, image_id)
    except Exception as e:  # noqa: BLE001
        return {"image_id": image_id, "_error": f"{type(e).__name__}: {e}"}


def _terciles(s):
    q1, q2 = s.quantile([1 / 3, 2 / 3])
    return pd.cut(s, [-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"])


def main(workers=4):
    idx = pd.read_parquet(OUT / "file_index.parquet")
    done = set(pd.read_parquet(OUT_PATH)["image_id"]) if OUT_PATH.exists() else set()
    todo = idx[~idx["image_id"].isin(done)]
    print(f"{len(todo)} volumes to process ({len(done)} done), {workers} workers")

    jobs = [(r["path"], r["image_id"]) for _, r in todo.iterrows()]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_worker, jobs), 1):
            (errors if "_error" in res else rows).append(res)
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}  (errors: {len(errors)})")

    if rows:
        new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            new = pd.concat([pd.read_parquet(OUT_PATH).drop(columns=["biasfield_cat", "efc_cat"],
                                                            errors="ignore"), new], ignore_index=True)
        new["biasfield_cat"] = _terciles(new["biasfield"])
        new["efc_cat"] = _terciles(new["efc"])
        new.to_parquet(OUT_PATH, index=False)

    if errors:
        (LOG / "qc_errors.txt").write_text(
            "\n".join(f"{e['image_id']}\t{e['_error']}" for e in errors) + "\n")

    final = pd.read_parquet(OUT_PATH)
    print(f"\nqc_factors.parquet: {len(final)}/{len(idx)} rows, {len(errors)} errors")
    print("template_space:", final["template_space"].value_counts().to_dict())
    print(f"biasfield  terciles at {final['biasfield'].quantile([1/3, 2/3]).round(3).tolist()}")
    print(f"efc        terciles at {final['efc'].quantile([1/3, 2/3]).round(3).tolist()}")


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    main(w)

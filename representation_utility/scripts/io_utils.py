"""Load a NIfTI, put it in canonical orientation, resample to 1 mm, 128^3, normalise."""
from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

OUT_SIZE = 128
TARGET_MM = 2.0  # 128 * 2mm = 256mm box -> fits any head, no clipping


def load_volume(path):
    """Return (data float32, zooms) in canonical (RAS) orientation. Raises on failure."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms


def resample_iso(data, zooms, target_mm=TARGET_MM):
    factors = [z / target_mm for z in zooms]
    if np.allclose(factors, 1.0):
        return data
    return zoom(data, factors, order=1)


def center_crop_pad(data, size=OUT_SIZE):
    out = np.zeros((size, size, size), dtype=np.float32)
    src, dst = [], []
    for n in data.shape:
        if n >= size:
            s = (n - size) // 2
            src.append(slice(s, s + size))
            dst.append(slice(0, size))
        else:
            s = (size - n) // 2
            src.append(slice(0, n))
            dst.append(slice(s, s + n))
    out[tuple(dst)] = data[tuple(src)]
    return out


def normalize(data):
    """Z-score using non-zero voxel stats (MedicalNet convention)."""
    fg = data[data != 0]
    if fg.size == 0:
        raise ValueError("empty volume")
    return (data - fg.mean()) / (fg.std() + 1e-8)


def to_model_input(data, zooms):
    return normalize(center_crop_pad(resample_iso(data, zooms)))


def prepare(path):
    data, zooms = load_volume(path)
    return to_model_input(data, zooms)


def central_axial_slice(data):
    """Middle axial slice (axis 2 is S after canonical reorientation)."""
    return data[:, :, data.shape[2] // 2]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _env import OUT

    df = pd.read_parquet(OUT / "file_index.parquet")
    for _, r in df.groupby("dataset").first().reset_index().iterrows():
        data, zooms = load_volume(r["path"])
        res = center_crop_pad(resample_iso(data, zooms))
        x = normalize(res)
        fg = x[res != 0]
        nz = np.argwhere(res != 0)
        clipped = bool((nz.min(0) <= 0).any() or (nz.max(0) >= OUT_SIZE - 1).any())
        print(f"{r['dataset']:14s} {r['modality']:5s} shape={x.shape} "
              f"fg_mean={fg.mean():+.3f} fg_std={fg.std():.3f} "
              f"finite={np.isfinite(x).all()} clipped={clipped}")

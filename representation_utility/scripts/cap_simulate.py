"""CAP E1: acquisition-factor simulators applied in native space before io_utils.prepare.

Usage:
  python cap_simulate.py masks     # HD-BET brain masks for the pilot volumes
  python cap_simulate.py check     # negative-control identity check + PNG grids
"""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEVICE, FIG, OUT, ROOT  # noqa: E402
from io_utils import central_axial_slice, prepare, to_model_input  # noqa: E402

LEVELS = yaml.safe_load((ROOT / "configs" / "cap_levels.yaml").read_text())
MASK_DIR = OUT / "cap_masks"
STRIP_MIN_OUTSIDE = 0.05
# already skull-stripped (checked visually on the pilot scans); no mask needed
STRIPPED = {"BraTS25-MEN", "BraTS25-MET", "BraTS25-SSA", "BrainMetShare", "ISLES22", "MSLesSeg", "NFBS"}
FACTORS = ["slice_thickness", "bias_field", "noise", "skull_strip"]
CONTROLS = ["intensity_scale", "orientation_flip"]


def seed_for(image_id, factor, level):
    """Deterministic seed per (image, factor, level)."""
    return int(hashlib.md5(f"{image_id}|{factor}|{level}".encode()).hexdigest()[:8], 16)


def load_img(img):
    """Same as io_utils.load_volume, from an in-memory image."""
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, tuple(float(z) for z in img.header.get_zooms()[:3])


def prepare_img(img):
    return to_model_input(*load_img(img))


# ------------------------------------------------------------- factors ---
def _interp_axis(a, axis, pos):
    """Linear interpolation of `a` along `axis` at fractional positions `pos`."""
    pos = np.clip(pos, 0, a.shape[axis] - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, a.shape[axis] - 1)
    w = (pos - lo).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = -1
    return np.take(a, lo, axis) * (1 - w.reshape(shape)) + np.take(a, hi, axis) * w.reshape(shape)


def slice_axis(zooms):
    """Thickest axis if anisotropic, else the superior-inferior axis (axial slices)."""
    return int(np.argmax(zooms)) if max(zooms) / min(zooms) > 1.3 else 2


def slice_thickness(data, zooms, mm, rng=None):
    """Thick slices of `mm` along the slice axis (box profile), resampled back to the native grid."""
    ax = slice_axis(zooms)
    k = mm / zooms[ax]
    if k <= 1.05:
        return None
    n = data.shape[ax]
    blurred = ndimage.uniform_filter1d(data, size=max(int(round(k)), 1), axis=ax, mode="nearest")
    centers = np.arange((k - 1) / 2, n, k)
    thick = _interp_axis(blurred, ax, centers)
    back = _interp_axis(thick, ax, (np.arange(n) - (k - 1) / 2) / k)
    return back.astype(np.float32)


def smooth_logfield(shape, zooms, rng, order=3):
    """Random low-order polynomial log-field in physical coordinates, unit std."""
    grids = np.meshgrid(*[(np.arange(n) - n / 2) * z / 128.0 for n, z in zip(shape, zooms)], indexing="ij")
    f = np.zeros(shape, np.float32)
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                if i + j + k:
                    f += rng.normal() * grids[0] ** i * grids[1] ** j * grids[2] ** k
    return f / (f.std() + 1e-8)


def bias_field(data, zooms, cv, rng):
    """Multiply by a smooth field whose CV over the foreground equals `cv`."""
    fg = data != 0
    f = smooth_logfield(data.shape, zooms, rng)[fg]
    lo, hi = 0.0, 5.0
    for _ in range(40):
        a = (lo + hi) / 2
        e = np.exp(a * f)
        lo, hi = (a, hi) if e.std() / e.mean() < cv else (lo, a)
    field = np.ones_like(data)
    field[fg] = np.exp(a * f) / np.exp(a * f).mean()
    return data * field


def rician_noise(data, zooms, rel_sigma, rng):
    """Rician noise, sigma relative to tissue mean, inside the non-zero support."""
    fg = data != 0
    x = data[fg]
    tissue = x > 0.2 * np.percentile(x, 99)
    s = rel_sigma * np.abs(x[tissue]).mean()
    out = data.copy()
    out[fg] = np.sqrt((x + s * rng.normal(size=x.shape)) ** 2 + (s * rng.normal(size=x.shape)) ** 2)
    return out.astype(np.float32)


def skull_strip(data, mask):
    """Zero everything outside the brain mask; None if the volume is already stripped."""
    fg = data != 0
    outside = (fg & ~mask).sum() / max(fg.sum(), 1)
    if outside < STRIP_MIN_OUTSIDE:
        return None
    return np.where(mask, data, 0).astype(np.float32)


def ghosting(data, zooms, intensity, rng):
    """torchio Ghosting along the A-P axis (not in the pilot)."""
    import torchio as tio
    t = tio.Ghosting(num_ghosts=4, axis=1, intensity=intensity, restore=0.02)
    return t(data[None]).squeeze(0).astype(np.float32)


SIMS = {"slice_thickness": slice_thickness, "bias_field": bias_field, "noise": rician_noise,
        "ghosting": ghosting}


# ------------------------------------------------------------ controls ---
def scaled_img(img, k):
    data, zooms = load_img(img)
    canon = nib.as_closest_canonical(img)
    out = nib.Nifti1Image(data * k, canon.affine, header=canon.header)
    out.header.set_zooms(zooms)
    return out


def flipped_img(img):
    """Same image stored differently: axis 0 flipped, axes 0/1 swapped, affine updated."""
    canon = nib.as_closest_canonical(img)
    data, zooms = load_img(img)
    n0 = data.shape[0]
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, n0 - 1
    swap = np.eye(4)[[1, 0, 2, 3]]
    aff = canon.affine @ flip @ swap
    new = np.swapaxes(data[::-1], 0, 1)
    out = nib.Nifti1Image(np.ascontiguousarray(new), aff, header=canon.header)
    out.header.set_zooms((zooms[1], zooms[0], zooms[2]))
    return out


# ---------------------------------------------------------------- driver ---
def load_mask(image_id):
    p = MASK_DIR / f"{image_id}_bet.nii.gz"
    return np.asanyarray(nib.load(str(p)).dataobj) > 0 if p.exists() else None


def variants(image_id, path):
    """Yield (factor, level, applied, model_input) for the original, every factor level and the controls."""
    img = nib.load(str(path))
    data, zooms = load_img(img)
    yield "original", "none", True, to_model_input(data, zooms)
    for fac in FACTORS:
        for lvl, val in LEVELS[fac]["levels"].items():
            if fac == "skull_strip":
                mask = load_mask(image_id)
                out = None if mask is None else skull_strip(data, mask)
            else:
                out = SIMS[fac](data, zooms, val, np.random.default_rng(seed_for(image_id, fac, lvl)))
            yield fac, lvl, out is not None, None if out is None else to_model_input(out, zooms)
    yield "intensity_scale", "control", True, prepare_img(scaled_img(img, LEVELS["controls"]["intensity_scale"]))
    yield "orientation_flip", "control", True, prepare_img(flipped_img(img))


def run_masks():
    """HD-BET masks on canonical copies of the unstripped pilot volumes."""
    pilot = pd.read_csv(OUT / "cap_pilot_ids.csv")
    pilot = pilot[~pilot["dataset"].isin(STRIPPED)]
    MASK_DIR.mkdir(exist_ok=True)
    todo = pilot[[not (MASK_DIR / f"{i}_bet.nii.gz").exists() for i in pilot["image_id"]]]
    print(f"{len(todo)} masks to compute ({len(pilot) - len(todo)} done)")
    if todo.empty:
        return
    tin, tout = MASK_DIR / "_in", MASK_DIR / "_out"
    tin.mkdir(exist_ok=True)
    for _, r in todo.iterrows():
        canon = nib.as_closest_canonical(nib.load(r["path"]))
        data, _ = load_img(canon)
        nib.save(nib.Nifti1Image(data, canon.affine), tin / f"{r['image_id']}.nii.gz")
    subprocess.run(["hd-bet", "-i", str(tin), "-o", str(tout), "-device", DEVICE,
                    "--save_bet_mask", "--no_bet_image", "--disable_tta"], check=True)
    for f in tout.glob("*_bet.nii.gz"):
        shutil.move(str(f), MASK_DIR / f.name)
    shutil.rmtree(tin)
    shutil.rmtree(tout)
    stats = []
    for _, r in pilot.iterrows():
        data, _ = load_img(nib.load(r["path"]))
        mask = load_mask(r["image_id"])
        fg = data != 0
        stats.append(dict(image_id=r["image_id"], dataset=r["dataset"],
                          outside_frac=float((fg & ~mask).sum() / fg.sum())))
    s = pd.DataFrame(stats)
    s["skull_present"] = s["outside_frac"] >= STRIP_MIN_OUTSIDE
    s.to_csv(OUT / "cap_mask_stats.csv", index=False)
    print(s.groupby("dataset")[["outside_frac", "skull_present"]].mean().round(3))


def run_check():
    """Negative controls must equal the original after prepare; save PNG grids for 2 volumes."""
    import matplotlib.pyplot as plt
    import results_lib  # noqa: F401  (serif fonts)

    pilot = pd.read_csv(OUT / "cap_pilot_ids.csv")
    rows = []
    for _, r in pilot.iterrows():
        img = nib.load(r["path"])
        orig = prepare(r["path"])
        rows.append(dict(image_id=r["image_id"], dataset=r["dataset"],
                         load_img_vs_prepare=float(np.abs(prepare_img(img) - orig).max()),
                         intensity_scale=float(np.abs(prepare_img(scaled_img(img, 3.7)) - orig).max()),
                         orientation_flip=float(np.abs(prepare_img(flipped_img(img)) - orig).max())))
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / "cap_control_check.csv", index=False)
    print("max |difference| after prepare (z-scored units):")
    print(chk[["load_img_vs_prepare", "intensity_scale", "orientation_flip"]].max())

    out = FIG / "cap_checks"
    out.mkdir(exist_ok=True)
    stats = pd.read_csv(OUT / "cap_mask_stats.csv") if (OUT / "cap_mask_stats.csv").exists() else None
    unstripped = stats.loc[stats["skull_present"], "image_id"] if stats is not None else pilot["image_id"]
    picks = [pilot.loc[pilot["dataset"] == "IXI", "image_id"].iloc[0],
             pilot.loc[pilot["dataset"] == "BraTS25-MET", "image_id"].iloc[0]]
    picks = [p for p in picks if p in set(pilot["image_id"])]
    if picks[0] not in set(unstripped):
        picks[0] = unstripped.iloc[0]
    got = {p: {(f, l): x for f, l, _, x in variants(p, pilot.set_index("image_id").loc[p, "path"])}
           for p in picks}
    for fac in FACTORS + CONTROLS:
        keys = [("original", "none")] + [k for k in got[picks[0]] if k[0] == fac]
        fig, axes = plt.subplots(len(picks), len(keys), figsize=(2.2 * len(keys), 2.3 * len(picks)),
                                 squeeze=False)
        for i, p in enumerate(picks):
            for j, k in enumerate(keys):
                x = got[p].get(k)
                ax = axes[i, j]
                if x is not None:
                    ax.imshow(np.rot90(central_axial_slice(x)), cmap="gray", vmin=-2, vmax=4)
                else:
                    ax.text(0.5, 0.5, "not applied", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]), ax.set_yticks([])
                if i == 0:
                    ax.set_title(k[1] if k[0] != "original" else "original", fontsize=9)
            axes[i, 0].set_ylabel(p.split("_", 1)[-1][:22], fontsize=7)
        fig.suptitle(fac.replace("_", " "), fontsize=11)
        fig.savefig(out / f"cap_check_{fac}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved PNG grids to {out}")


if __name__ == "__main__":
    {"masks": run_masks, "check": run_check}[sys.argv[1]]()

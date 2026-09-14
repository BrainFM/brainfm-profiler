"""Extend the image-level analysis (voxel spacing, orientation, intensity)
from the original 14-dataset subset to the 17-dataset corpus used from the
representation-level analysis onward: drop MSSEG-2 (files not available,
already excluded there), add 3D-MR-MS, BGSP, NSK-GBM, UPENN-GBM (metadata
already collected via characterization/collect_metadata.py).

Standalone script, not the notebook: only needs pandas/numpy/matplotlib/scipy,
already present in representation_utility/.venv. Mirrors the logic in
analysis.ipynb's "Voxel spacing", "Orientation", and "Intensity" sections
exactly, so the numbers are directly comparable to what was reported before.

Run: representation_utility/.venv/bin/python characterization/image_level_analysis_17.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kruskal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

ROOT = Path(__file__).resolve().parent
FIG_OUT = ROOT.parent / "Dissertation" / "figures"

# 17-dataset corpus: the 13 "core" + 4 "added" datasets, matching the
# representation-level analysis exactly (MSSEG-2 dropped, not on the drive).
FILES = [
    "brats25met", "brats25ssa", "brats25men", "episurg", "isles22", "msbaghdad",
    "mslesseg", "oasis1", "oasis2", "ixi", "umfpd", "nfbs", "brainmetshare",
    "3dmrms", "bgsp", "nskgbm", "upenngbm",
]

# cosmetic renames so labels match the chapter's existing naming convention
# (the raw metadata 'dataset' column uses the collector's internal names)
RENAME = {
    "BraTS25MET": "BraTS-MET", "BraTS25SSA": "BraTS-SSA", "BraTS25MEN": "BraTS-MEN",
    "MSBaghdad": "MS-60", "UMFPD": "UMFPD",  # already matches existing table spelling
}


def parse_voxel_spacing(vs):
    if isinstance(vs, str):
        return tuple(map(float, vs.strip("()").split(",")))
    return vs


def compute_anisotropy_ratio(vs):
    vs = np.array(vs)
    return np.max(vs) / np.min(vs)


def categorize_anisotropy(ratio, iso_threshold=1.0, high_aniso_threshold=2.0):
    if ratio == iso_threshold:
        return "Isotropic"
    elif ratio < high_aniso_threshold:
        return "Mildly Anisotropic"
    else:
        return "Highly Anisotropic"


def main():
    dfs = []
    for f in FILES:
        d = pd.read_csv(ROOT / "metadata" / f"metadata_{f}.csv")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["dataset"] = df["dataset"].replace(RENAME)

    n_patients = df["patient"].nunique()
    print(f"17-dataset corpus: {n_patients} patients, {df.shape[0]} images, "
          f"{df['dataset'].nunique()} datasets")
    print(sorted(df["dataset"].unique()))

    # ---- Voxel spacing / anisotropy ----
    df["voxel_spacing_parsed"] = df["voxel_spacing"].apply(parse_voxel_spacing)
    df["anisotropy_ratio"] = df["voxel_spacing_parsed"].apply(compute_anisotropy_ratio)
    df["anisotropy_category"] = df["anisotropy_ratio"].apply(categorize_anisotropy)

    print("\nAnisotropy category counts:")
    counts = df["anisotropy_category"].value_counts()
    print(counts)

    df[["spacing_x", "spacing_y", "spacing_z"]] = pd.DataFrame(
        df["voxel_spacing_parsed"].tolist(), index=df.index)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    datasets = sorted(df["dataset"].unique())
    colors = plt.cm.tab20.colors
    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        ax.scatter(subset["spacing_x"], subset["spacing_y"], subset["spacing_z"],
                   color=colors[i % len(colors)], label=dataset, s=50, alpha=0.6)
    ax.set_xlabel("Spacing X (mm)")
    ax.set_ylabel("Spacing Y (mm)")
    ax.set_zlabel("Spacing Z (mm)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(FIG_OUT / "voxel_3d.jpeg", format="jpeg", dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {FIG_OUT / 'voxel_3d.jpeg'}")

    # per-dataset anisotropy breakdown, to identify which datasets are anisotropic
    print("\nPer-dataset anisotropy breakdown (mildly/highly anisotropic datasets):")
    ds_summary = df.groupby("dataset")["anisotropy_category"].value_counts().unstack().fillna(0)
    for ds in ds_summary.index:
        row = ds_summary.loc[ds]
        total = row.sum()
        aniso_frac = (row.get("Mildly Anisotropic", 0) + row.get("Highly Anisotropic", 0)) / total
        if aniso_frac > 0.3:
            print(f"  {ds:16s} iso={row.get('Isotropic',0):.0f} "
                  f"mild={row.get('Mildly Anisotropic',0):.0f} "
                  f"high={row.get('Highly Anisotropic',0):.0f}")

    # ---- Orientation ----
    print("\nOrientation code distribution:")
    summary = (
        df.groupby("axcodes")["dataset"]
        .agg(["count", lambda x: ", ".join(sorted(x.unique()))])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.columns = ["axcodes", "count", "datasets"]
    print(summary.to_string(index=False))
    print(f"\ntop-3 orientation share: {summary['count'].iloc[:3].sum() / summary['count'].sum():.3f}")

    # ---- Intensity ----
    # A handful of NSK-GBM FLAIR/T2 images carry raw, unnormalized intensity
    # (up to ~18,900) against a corpus otherwise under ~400; left unclipped,
    # the y-axis has to stretch to fit them and every other dataset's real
    # spread reads as a flat line near zero. Clip the axis and disclose the
    # excluded points explicitly rather than let them hide the real signal.
    Y_CAP = 450
    n_clipped = int((df["median_value"] > Y_CAP).sum())
    clipped_datasets = sorted(df.loc[df["median_value"] > Y_CAP, "dataset"].unique())
    print(f"\n{n_clipped} points above y={Y_CAP} clipped from the intensity plot, "
          f"all from: {clipped_datasets}")

    plt.figure(figsize=(10, 5))
    order = sorted(df["dataset"].unique())
    positions = {d: i for i, d in enumerate(order)}
    rng = np.random.default_rng(42)
    for d in order:
        vals = df.loc[df["dataset"] == d, "median_value"]
        x = positions[d] + rng.uniform(-0.2, 0.2, size=len(vals))
        plt.scatter(x, vals, alpha=0.4, s=10)
    plt.ylim(-20, Y_CAP)
    plt.xticks(range(len(order)), order, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_OUT / "median_intensity_per_image.jpeg", format="jpeg", dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"saved {FIG_OUT / 'median_intensity_per_image.jpeg'}")

    groups = [g["median_value"].to_numpy() for _, g in df.groupby("dataset")]
    stat, p = kruskal(*groups)
    print(f"\nKruskal-Wallis H = {stat:.3f}, p = {p:.3e}")


if __name__ == "__main__":
    main()

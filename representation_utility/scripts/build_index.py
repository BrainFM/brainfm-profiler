"""Build outputs/file_index.parquet from the metadata CSVs (configs/datasets.yaml)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import METADATA_DIR, OUT, LOG, SEED  # noqa: E402

CFG_DIR = Path(__file__).resolve().parents[1] / "configs"
KEEP_MODALITIES = {"T1", "T1c", "T2", "FLAIR", "PD"}


def parse_spacing(s: str) -> tuple[float, float, float]:
    return tuple(float(x) for x in str(s).strip("()[] ").split(","))


def anisotropy(sp) -> tuple[float, str]:
    sp = np.asarray(sp, dtype=float)
    ratio = float(sp.max() / sp.min())
    if np.isclose(ratio, 1.0):
        cat = "Isotropic"
    elif ratio < 2.0:
        cat = "Mildly Anisotropic"
    else:
        cat = "Highly Anisotropic"
    return ratio, cat


def check_readable(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return "missing"
    try:
        shp = nib.load(str(p)).header.get_data_shape()
        if len(shp) < 3 or min(shp[:3]) < 2:
            return f"bad_shape:{shp}"
    except Exception as e:  # noqa: BLE001
        return f"load_error:{type(e).__name__}"
    return None


def sample_patients(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    patients = df["patient"].astype(str).unique().tolist()
    rng.shuffle(patients)
    kept, n = [], 0
    for pat in patients:
        rows = df[df["patient"].astype(str) == pat]
        kept.append(rows)
        n += len(rows)
        if n >= cap:
            break
    return pd.concat(kept, ignore_index=True)


def main() -> None:
    cfg = yaml.safe_load((CFG_DIR / "datasets.yaml").read_text())
    modmap = yaml.safe_load((CFG_DIR / "modality_map.yaml").read_text())
    modmap = {str(k).lower(): v for k, v in modmap.items()}
    cap = int(cfg["cap_per_dataset"])
    rng = np.random.default_rng(cfg.get("seed", SEED))

    all_rows, summary, bad_lines = [], [], []

    for d in cfg["datasets"]:
        name, csv = d["name"], d["csv"]
        df = pd.read_csv(METADATA_DIR / csv, low_memory=False)
        n_raw = len(df)

        rw = d.get("path_rewrite")
        if rw:
            df["image_path"] = df["image_path"].astype(str).str.replace(rw[0], rw[1], regex=False)

        df["modality_raw"] = df["modality"].astype(str).str.strip().str.lower()
        df["modality"] = df["modality_raw"].map(modmap)
        for u in sorted(set(df.loc[df["modality"].isna(), "modality_raw"])):
            bad_lines.append(f"{name}\tUNKNOWN_MODALITY\t{u}")
        df = df[df["modality"].isin(KEEP_MODALITIES)].copy()
        n_struct = len(df)

        df = sample_patients(df, cap, rng)

        reasons = df["image_path"].map(check_readable)
        for pth, r in zip(df["image_path"][reasons.notna()], reasons[reasons.notna()]):
            bad_lines.append(f"{name}\t{r}\t{pth}")
        df = df[reasons.isna()].copy()
        n_ok = len(df)

        sp = df["voxel_spacing"].map(parse_spacing)
        df["anisotropy_ratio"] = sp.map(lambda x: anisotropy(x)[0])
        df["anisotropy_cat"] = sp.map(lambda x: anisotropy(x)[1])
        df["dataset"] = name
        df["domain"] = d.get("domain", "")
        df["patient"] = df["patient"].astype(str)
        df["image_id"] = df["image_id"].astype(str)
        df = df.rename(columns={"image_path": "path"})

        keep_cols = [
            "dataset", "domain", "patient", "image_id", "modality", "modality_raw",
            "path", "shape", "voxel_spacing", "axcodes", "anisotropy_ratio", "anisotropy_cat",
        ]
        all_rows.append(df[keep_cols])

        summary.append(dict(
            dataset=name, raw_rows=n_raw, structural=n_struct, verified=len(reasons),
            kept=n_ok, patients=df["patient"].nunique(),
            modalities="|".join(f"{k}:{v}" for k, v in sorted(df["modality"].value_counts().items())),
            aniso="|".join(f"{k}:{v}" for k, v in df["anisotropy_cat"].value_counts().items()),
        ))
        print(f"{name:14s} raw={n_raw:5d}  structural={n_struct:5d}  "
              f"kept={n_ok:4d}  patients={df['patient'].nunique():4d}  [{summary[-1]['modalities']}]")

    index = pd.concat(all_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    index.to_parquet(OUT / "file_index.parquet", index=False)
    pd.DataFrame(summary).to_csv(OUT / "index_summary.csv", index=False)
    (LOG / "unreadable.txt").write_text(
        "dataset\treason\tdetail\n" + "\n".join(bad_lines) + ("\n" if bad_lines else "")
    )

    print("\n" + "=" * 60)
    print(f"file_index.parquet : {len(index)} rows, {index['dataset'].nunique()} datasets, "
          f"{index['patient'].nunique()} patients")
    print("modality totals    :", dict(index["modality"].value_counts()))
    print("anisotropy totals  :", dict(index["anisotropy_cat"].value_counts()))
    print(f"unreadable.txt     : {len(bad_lines)} entries")


if __name__ == "__main__":
    main()

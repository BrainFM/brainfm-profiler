"""Append datasets with no metadata CSV (configs/datasets_extra.yaml) to file_index.parquet."""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, LOG, SEED  # noqa: E402
from build_index import anisotropy, KEEP_MODALITIES  # noqa: E402

CFG_DIR = Path(__file__).resolve().parents[1] / "configs"


# path -> (patient, raw_modality), or None to skip
def parse_mr_ms(p: Path):
    if p.parent.name == "raw":
        return None
    m = re.match(r"(patient\d+)_([A-Za-z0-9]+)\.nii\.gz$", p.name)
    return (m.group(1), m.group(2).lower()) if m else None


def parse_bids_t1(p: Path):
    if p.parent.name != "anat" or "_T1w" not in p.name:
        return None
    m = re.match(r"(sub-[A-Za-z0-9]+)_", p.name)
    return (m.group(1), "t1w") if m else None


def parse_nsk_gbm(p: Path):
    if p.parent.name != "anat":
        return None
    m = re.match(r"(sub-\d+)_([A-Za-z0-9]+)\.nii\.gz$", p.name)
    if not m or m.group(2).lower() not in {"flair", "t1ce", "t1w", "t2w"}:
        return None
    return m.group(1), m.group(2).lower()


def parse_upenn(p: Path):
    m = re.match(r"(UPENN-GBM-\d+)_11_([A-Za-z0-9]+)_unstripped\.nii\.gz$", p.name)
    return (m.group(1), m.group(2).lower()) if m else None


PARSERS = {"mr_ms": parse_mr_ms, "bids_t1": parse_bids_t1, "nsk_gbm": parse_nsk_gbm, "upenn": parse_upenn}


def header_info(path: str):
    try:
        img = nib.load(path)
        shp = tuple(int(x) for x in img.header.get_data_shape()[:3])
        if len(shp) < 3 or min(shp) < 2:
            return None
        zooms = tuple(round(float(z), 6) for z in img.header.get_zooms()[:3])
        return shp, zooms, "".join(nib.aff2axcodes(img.affine))
    except Exception:  # noqa: BLE001
        return None


def sample_patients(df, cap, rng):
    if len(df) <= cap:
        return df
    pats = df["patient"].unique().tolist()
    rng.shuffle(pats)
    kept, n = [], 0
    for pat in pats:
        rows = df[df["patient"] == pat]
        kept.append(rows)
        n += len(rows)
        if n >= cap:
            break
    return pd.concat(kept, ignore_index=True)


def main() -> None:
    cfg = yaml.safe_load((CFG_DIR / "datasets_extra.yaml").read_text())
    modmap = yaml.safe_load((CFG_DIR / "modality_map.yaml").read_text())
    modmap = {str(k).lower(): v for k, v in modmap.items()}
    cap = int(cfg["cap_per_dataset"])
    rng = np.random.default_rng(cfg.get("seed", SEED))

    new_rows, bad = [], []

    for d in cfg["datasets"]:
        name, root, parser = d["name"], Path(d["root"]), PARSERS[d["parser"]]
        if not root.is_dir():
            bad.append(f"{name}\tROOT_MISSING\t{root}")
            print(f"{name:12s}  ROOT MISSING: {root}")
            continue

        recs = []
        for p in sorted(root.rglob("*.nii.gz")):
            if {"old", "raw", "derivatives"} & {x.lower() for x in p.parts}:
                continue
            if any(t in p.name.lower() for t in ("_seg", "gt", "mask", "brainmask", "_swi", "roi", "label")):
                continue
            parsed = parser(p)
            if parsed is None:
                continue
            patient, mod_raw = parsed
            mod = modmap.get(mod_raw)
            if mod not in KEEP_MODALITIES:
                continue
            recs.append(dict(patient=patient, modality=mod, modality_raw=mod_raw, path=str(p)))

        df = pd.DataFrame(recs)
        if df.empty:
            bad.append(f"{name}\tNO_FILES_MATCHED\t{root}")
            print(f"{name:12s}  no files matched")
            continue

        n_found = len(df)
        df["image_id"] = df["path"].map(lambda s: Path(s).name.replace(".nii.gz", ""))
        df = sample_patients(df, cap, rng)

        info = df["path"].map(header_info)
        for pth in df["path"][info.isna()]:
            bad.append(f"{name}\tunreadable\t{pth}")
        df = df[info.notna()].copy()
        info = info[info.notna()]
        df["shape"] = [str(t[0]) for t in info]
        df["voxel_spacing"] = [str(t[1]) for t in info]
        df["axcodes"] = [t[2] for t in info]
        df["anisotropy_ratio"] = [anisotropy(t[1])[0] for t in info]
        df["anisotropy_cat"] = [anisotropy(t[1])[1] for t in info]
        df["dataset"] = name
        df["domain"] = d.get("domain", "")

        cols = ["dataset", "domain", "patient", "image_id", "modality", "modality_raw",
                "path", "shape", "voxel_spacing", "axcodes", "anisotropy_ratio", "anisotropy_cat"]
        new_rows.append(df[cols])
        print(f"{name:12s}  found={n_found:5d}  kept={len(df):4d}  patients={df['patient'].nunique():4d}  "
              f"[{'|'.join(f'{k}:{v}' for k, v in sorted(df['modality'].value_counts().items()))}]")

    extra = pd.concat(new_rows, ignore_index=True)
    idx_path = OUT / "file_index.parquet"
    base = pd.read_parquet(idx_path)
    base = base[~base["dataset"].isin(extra["dataset"].unique())]   # idempotent re-run
    combined = pd.concat([base, extra], ignore_index=True)
    assert combined["image_id"].duplicated().sum() == 0, "duplicate image_id after merge"
    combined.to_parquet(idx_path, index=False)
    (LOG / "unreadable_extra.txt").write_text(
        "dataset\treason\tdetail\n" + "\n".join(bad) + ("\n" if bad else "")
    )

    print("\n" + "=" * 60)
    print(f"file_index.parquet : {len(combined)} rows, {combined['dataset'].nunique()} datasets, "
          f"{combined['patient'].nunique()} patients")
    print("per dataset:\n", combined.groupby("dataset").size().to_string())
    print("modality totals :", dict(combined["modality"].value_counts()))


if __name__ == "__main__":
    main()

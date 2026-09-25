"""E6: age and sex per volume, where the dataset provides them -> outputs/content_labels.parquet.

Sources are the datasets' own demographic tables. Files the drive does not hold
(OASIS-2, IXI, UPENN-GBM) are kept in representation_utility/metadata/.
No labels: BraTS25-MEN/MET/SSA, BrainMetShare, EPISURG (none published),
BGSP and NFBS (demographics sit behind separate data-use terms).
"""
from __future__ import annotations
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DATASETS_DIR, OUT, ROOT  # noqa: E402
from results_lib import add_number, save_table  # noqa: E402

META = ROOT / "metadata"
MSLESSEG = DATASETS_DIR / "MSLesSeg/27919209/MSLesSeg Dataset/info_dataset/clinical_data.csv"


def _sex(v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip().lower()
    if s in ("f", "female", "2"):
        return "F"
    if s in ("m", "male", "1"):
        return "M"
    return None


def _age(v):
    if pd.isna(v):
        return np.nan
    m = re.search(r"\d+(?:[.,]\d+)?", str(v))
    return float(m.group(0).replace(",", ".")) if m else np.nan


def by_patient(g, table, key, age, sex, src):
    t = table.set_index(key)
    k = g["patient"]
    return pd.DataFrame({"image_id": g["image_id"], "age": k.map(t[age]).map(_age),
                         "sex": k.map(t[sex]).map(_sex), "source": src})


def oasis1(g):
    rows = []
    for p in g["path"]:
        sess = re.search(r"(OAS1_\d+_MR\d)", p).group(1)
        txt = Path(p).parents[1] / f"{sess}.txt"
        info = dict(re.findall(r"^([A-Z/]+):\s*(.*)$", txt.read_text(), flags=re.M))
        rows.append((_age(info.get("AGE")), _sex(info.get("M/F"))))
    a, s = zip(*rows)
    return pd.DataFrame({"image_id": g["image_id"], "age": a, "sex": s, "source": "OAS1_*.txt"})


def oasis2(g):
    t = pd.read_excel(META / "oasis_longitudinal_demographics.xlsx").set_index("MRI ID")
    sess = g["path"].str.extract(r"(OAS2_\d+_MR\d)")[0]
    return pd.DataFrame({"image_id": g["image_id"], "age": sess.map(t["Age"]).map(_age),
                         "sex": sess.map(t["M/F"]).map(_sex),
                         "source": "oasis_longitudinal_demographics.xlsx"})


def ixi(g):
    a = pd.read_csv(META / "ixi_tustison.csv")
    b = pd.read_csv(META / "ixi_openneurolab.csv")
    b.columns = a.columns
    assert (a["IXI_ID"].values == b["IXI_ID"].values).all()
    assert (a["SEX_ID"].values == b["SEX_ID"].values).all()
    assert np.allclose(a["AGE"], b["AGE"], atol=0.01, equal_nan=True)
    # IXI.xls repeats some IDs; drop IDs whose repeated rows disagree
    t = a.dropna(subset=["AGE"]).assign(AGE=lambda x: x["AGE"].round(1))
    t = t.drop_duplicates(["IXI_ID", "SEX_ID", "AGE"])
    t = t[~t["IXI_ID"].duplicated(keep=False)].set_index("IXI_ID")
    num = g["patient"].str.extract(r"IXI(\d+)")[0].astype(int)
    return pd.DataFrame({"image_id": g["image_id"], "age": num.map(t["AGE"]).map(_age),
                         "sex": num.map(t["SEX_ID"]).map(_sex), "source": "IXI.xls (mirror)"})


def upenn(g):
    t = pd.read_csv(META / "UPENN-GBM_clinical_info_v2.1.csv").set_index("ID")
    sess = g["path"].str.extract(r"(UPENN-GBM-\d+_\d+)")[0]
    return pd.DataFrame({"image_id": g["image_id"], "age": sess.map(t["Age_at_scan_years"]).map(_age),
                         "sex": sess.map(t["Gender"]).map(_sex),
                         "source": "UPENN-GBM_clinical_info_v2.1.csv"})


def mslesseg(g):
    t = pd.read_csv(MSLESSEG, sep=";", encoding="utf-8-sig")
    t["key"] = t["Patient"] + "_" + t["Timepoint"]
    t = t.set_index("key")
    key = g["patient"].where(g["patient"].str.contains("_"), g["patient"] + "_T1")
    return pd.DataFrame({"image_id": g["image_id"], "age": key.map(t["Age"]).map(_age),
                         "sex": key.map(t["Sex"]).map(_sex), "source": "clinical_data.csv"})


def ms60(g):
    t = pd.read_excel(DATASETS_DIR / "MS60/Supplementary Table 1 for patient info .xlsx", header=1)
    t = t.dropna(subset=["ID"])
    t["ID"] = t["ID"].astype(int).astype(str)
    return by_patient(g, t, "ID", "Age", "Gender", "Supplementary Table 1")


def mr3d(g):
    t = pd.read_csv(DATASETS_DIR / "3D-MR-MS/patient_info.csv")
    t["pid"] = "patient" + t["patient_id"].astype(int).map("{:02d}".format)
    return by_patient(g, t, "pid", "age", "sex", "patient_info.csv")


def tsv(path, key, age, sex):
    def f(g):
        t = pd.read_csv(path, sep="\t")
        t[key] = t[key].astype(str)
        return by_patient(g, t, key, age, sex, Path(path).name)
    return f


def umfpd(g):
    root = DATASETS_DIR / "umf_pd"
    a = pd.read_csv(root / "neurocon.tar/neurocon/neurocon_patients.tsv", sep="\t")[["code", "age", "sex"]]
    raw = (root / "taowu.tar/taowu/taowu_patients.tsv").read_text().replace("\r\n", "\n").replace("\r", "\n")
    b = pd.read_csv(io.StringIO(raw), sep="\t").rename(columns={"name": "code"})[["code", "age", "sex"]]
    t = pd.concat([a, b], ignore_index=True)
    t["code"] = "sub-" + t["code"].astype(str).str.strip()
    assert not t["code"].duplicated().any()
    return by_patient(g, t, "code", "age", "sex", "neurocon/taowu_patients.tsv")


LOADERS = {
    "OASIS-1": oasis1,
    "OASIS-2": oasis2,
    "IXI": ixi,
    "UPENN-GBM": upenn,
    "MSLesSeg": mslesseg,
    "MS-60": ms60,
    "3D-MR-MS": mr3d,
    "ISLES22": tsv(DATASETS_DIR / "ISLES-2022/ISLES-2022/participants.tsv", "participant_id", "age", "sex"),
    "NSK-GBM": tsv(DATASETS_DIR / "NSK-GBM/participants.tsv", "participant_id", "age", "gender"),
    "UMF-PD": umfpd,
}


def main():
    idx = pd.read_parquet(OUT / "file_index.parquet")
    parts = []
    for ds, g in idx.groupby("dataset"):
        if ds in LOADERS:
            parts.append(LOADERS[ds](g))
    lab = pd.concat(parts, ignore_index=True)
    out = idx[["image_id", "dataset", "patient", "modality"]].merge(lab, on="image_id", how="left")
    out.to_parquet(OUT / "content_labels.parquet", index=False)

    cov = out.groupby("dataset").agg(
        n=("image_id", "size"), age_n=("age", lambda s: int(s.notna().sum())),
        sex_n=("sex", lambda s: int(s.notna().sum())),
        age_mean=("age", "mean"), age_sd=("age", "std"), age_min=("age", "min"), age_max=("age", "max"),
        female_frac=("sex", lambda s: (s == "F").sum() / max(s.notna().sum(), 1))).reset_index()
    cov.loc[cov["sex_n"] == 0, "female_frac"] = np.nan
    save_table(cov, "content_label_coverage")
    print(cov.to_string())

    with_age = cov[cov["age_n"] > 0]
    with_sex = cov[cov["sex_n"] > 0]
    add_number("Content labels (E6)",
               f"age for {int(with_age['age_n'].sum())} volumes in {len(with_age)} datasets, "
               f"sex for {int(with_sex['sex_n'].sum())} volumes in {len(with_sex)} datasets; "
               f"dataset mean age ranges {with_age['age_mean'].min():.0f}-{with_age['age_mean'].max():.0f} y. "
               "None for BraTS25-MEN/MET/SSA, BrainMetShare, EPISURG, BGSP, NFBS")


if __name__ == "__main__":
    main()

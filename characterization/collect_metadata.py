"""Extract per-image metadata to metadata/metadata_<name>.csv.

Run:  python collect_metadata.py <name> [<name> ...]
      python collect_metadata.py --all
      python collect_metadata.py            (list names)
"""
import os
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from pandas.errors import EmptyDataError
except ImportError:  # older pandas
    EmptyDataError = ValueError

DATASETS_ROOT = "/Volumes/BACH2TB/Datasets"
METADATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata")

SCHEMA = {
    "image_id": str, "dataset": str, "patient": str, "image_path": str, "modality": str,
    "shape": tuple, "voxel_spacing": tuple, "axcodes": (tuple, list, str),
    "min_value": float, "max_value": float, "median_value": float,
    "affine": list, "orientation": list, "is_hwd": bool, "preprocessed_image_path": str,
}


def check_schema(sample: dict) -> bool:
    for key, expected_type in SCHEMA.items():
        if key not in sample:
            print(f"Missing key: {key}")
            return False
        if not isinstance(sample[key], expected_type):
            print(f"Key '{key}': expected {expected_type}, got {type(sample[key])}")
            return False
    return True


def get_paths(directory, file_extension=".nii.gz", file_extension_list=None):
    exts = file_extension_list or [file_extension]
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.endswith(e) for e in exts):
                paths.append(os.path.join(root, f))
    return paths


def extract_image_attributes(image_path):
    try:
        img = nib.load(image_path)
    except Exception:
        print(f"Failed loading {image_path}")
        return {}
    data = np.asanyarray(img.dataobj)  # native dtype, avoids float64 upcast
    affine = img.affine
    ornt = nib.orientations.io_orientation(affine)
    axcodes = "".join(nib.orientations.aff2axcodes(affine))
    is_hwd = (axcodes == "RAS") and np.array_equal(ornt[:, 0], [0, 1, 2])
    return {
        "shape": data.shape,
        "voxel_spacing": tuple(float(v) for v in img.header.get_zooms()),
        "axcodes": axcodes,
        "min_value": float(np.min(data)),
        "max_value": float(np.max(data)),
        "median_value": float(np.median(data)),
        "affine": affine.tolist(),
        "orientation": ornt.tolist(),
        "is_hwd": is_hwd,
    }


def collect_metadata(image_paths, csv_file, extract_nonimage_attributes_fn, flush_every=200):
    n = len(image_paths)
    try:
        done = set(pd.read_csv(csv_file)["image_id"].astype(str))
    except (FileNotFoundError, EmptyDataError):
        done = set()

    buffer = []

    def flush():
        if not buffer:
            return
        header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        pd.DataFrame(buffer).to_csv(csv_file, mode="a", header=header, index=False)
        buffer.clear()

    for i, image_path in enumerate(image_paths):
        non_image = extract_nonimage_attributes_fn(image_path)
        if not non_image:  # parser rejected this file
            continue
        if str(non_image.get("image_id")) in done:
            continue
        metadata = {**extract_image_attributes(image_path), **non_image}
        if not check_schema(metadata):
            raise ValueError(f"Bad schema for {image_path}")
        buffer.append(metadata)
        done.add(str(metadata["image_id"]))
        if len(buffer) >= flush_every:
            flush()
        print(f"{i + 1}/{n}")

    flush()


# --- per-dataset: image_path -> {image_id, patient, modality, timepoint, ...} ---
def _row(dataset, image_id, patient, modality, timepoint, image_path):
    return dict(image_id=image_id, patient=patient, modality=str(modality).lower(),
               timepoint=timepoint, image_path=image_path,
               preprocessed_image_path="", dataset=dataset)


def _mslesseg(p):
    s = os.path.basename(p).replace(".nii.gz", "").split("_")
    patient = "_".join(s[:-1])
    return _row("MSLesSeg", f"MSLesSeg_{patient}_T1_{s[-1].lower()}", patient, s[-1], "T1", p)


def _ms60(p):
    fn = os.path.basename(p).replace(".nii", "")
    patient, modality = fn.split("-")
    return _row("MSBaghdad", f"MS60_{patient}_{modality}", patient, modality, "T1", p)


def _brats(prefix, dataset, tp_from):
    def fn(p):
        name = os.path.basename(p).replace(".nii.gz", "")
        s = name.split("-")
        modality = s[-1]
        if tp_from == "raw":
            timepoint, patient = s[-2], "-".join(s[:-1])
        elif tp_from == "ssa":
            timepoint = "T1" if s[-2] == "000" else s[-2]
            patient = "-".join(s[:-2])
        else:  # men
            timepoint = f"T{int(s[-2][-1]) + 1}"
            patient = "-".join(s[:-2])
        return _row(dataset, f"{prefix}_{name}", patient, modality, timepoint, p)
    return fn


def _isles22(p):
    s = os.path.basename(p).replace(".nii.gz", "").split("_")
    patient, modality, tp = s[0], s[-1], f"T{s[1][-1]}"
    return _row("ISLES22", f"ISLES22_{patient}_{tp}_{modality}", patient, modality, tp, p)


def _episurg(p):
    fn = os.path.basename(p).replace("nii.gz", "")
    s = fn.split("-")
    modality = s[-2].replace("mri", "")
    patient = "-".join(s[:-2]).split("_")[0]
    tp = s[1].split("_")[-1]
    return _row("EPISURG", f"episurge_{patient}_{tp}_{modality.lower()}", patient, modality, tp, p)


def _ixi(p):
    modality = p.split("/")[-2].split("-")[-1].lower()
    s = os.path.basename(p).replace(".nii.gz", "").split("-")
    patient = "-".join(s[:-1])
    return _row("IXI", f"ixi_{patient}_T1_{modality}", patient, modality, "T1", p)


def _oasis(prefix, dataset):
    def fn(p):
        session = p.split("/")[-3]
        patient = "_".join(session.split("_")[:-1])
        return _row(dataset, f"{prefix}_{session}", patient, "t1w", f"T{session[-1]}", p)
    return fn


def _umfpd(p):
    s = os.path.basename(p).replace(".nii.gz", "").split("_")
    return _row("UMFPD", f"UMFPD_{s[0]}_T1_{s[-1].lower()}", s[0], s[-1], "T1", p)


def _nfbs(p):
    s = os.path.basename(p).replace(".nii.gz", "").split("_")
    patient = "_".join(s[:2])
    return _row("NFBS", f"NFBS_{patient}_T1_t1w", patient, "t1w", "T1", p)


def _brainmetshare(p):
    modality = os.path.basename(p).replace(".nii.gz", "").replace("_", "")
    patient = p.split("/")[-2]
    return _row("BrainMetShare", f"BrainMetShare_{patient}_{modality}", patient, modality, "T1", p)


def _mrms(p):
    if os.sep + "raw" + os.sep in p:
        return None
    m = re.match(r"(patient\d+)_(FLAIR|T1W|T1WKS|T2W)$", os.path.basename(p).replace(".nii.gz", ""))
    if not m:
        return None
    return _row("3D-MR-MS", f"3D-MR-MS_{m[1]}_T1_{m[2].lower()}", m[1], m[2], "T1", p)


def _bgsp(p):
    s = os.path.basename(p).replace(".nii.gz", "").split("_")  # sub-XXXX_ses-NN_T1w
    patient = s[0]
    ses = s[1] if len(s) > 2 and s[1].startswith("ses-") else "ses-01"
    return _row("BGSP", f"BGSP_{patient}_{ses}_t1w", patient, "t1w", f"T{ses.split('-')[-1].lstrip('0') or '1'}", p)


def _nskgbm(p):
    m = re.match(r"(sub-\d+)_(flair|T1ce|T1w|T2w)$", os.path.basename(p).replace(".nii.gz", ""))
    if not m:
        return None
    return _row("NSK-GBM", f"NSK-GBM_{m[1]}_T1_{m[2].lower()}", m[1], m[2], "T1", p)


def _upenn(p):
    m = re.match(r"(UPENN-GBM-\d+)_(\d)\d_([A-Za-z0-9]+)_unstripped$",
                 os.path.basename(p).replace(".nii.gz", ""))
    if not m:
        return None
    tp = f"T{m[2]}"  # _11 -> T1, _21 -> T2
    return _row("UPENN-GBM", f"UPENN-GBM_{m[1]}_{tp}_{m[3].lower()}", m[1], m[3], tp, p)


D = os.path.join
DATASETS = {
    "mslesseg":      dict(csv="metadata_mslesseg.csv",      ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "MSLesSeg/27919209/MSLesSeg Dataset")],
                          exclude=["MASK"], parse=_mslesseg),
    "msbaghdad":     dict(csv="metadata_msbaghdad.csv",     ext=[".nii"],
                          roots=[D(DATASETS_ROOT, "MS60/data")], exclude=["Seg"], parse=_ms60),
    "msseg2":        dict(csv="metadata_msseg2.csv",        ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "MSSEG-2")], exclude=[], parse=None),
    "brats25met":    dict(csv="metadata_brats25met.csv",    ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "BraTS25/BraTS25-MET")], exclude=["seg"],
                          parse=_brats("BraTS25", "BraTS25MET", "raw")),
    "brats25ssa":    dict(csv="metadata_brats25ssa.csv",    ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "BraTS25/BraTS25-SSA")], exclude=["seg"],
                          parse=_brats("BraTS25SSA", "BraTS25SSA", "ssa")),
    "brats25men":    dict(csv="metadata_brats25men.csv",    ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "BraTS25/BraTS25-MEN")], exclude=["seg"],
                          parse=_brats("BraTS25MEN", "BraTS25MEN", "men")),
    "isles22":       dict(csv="metadata_isles22.csv",       ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "ISLES-2022/ISLES-2022")], exclude=["msk"], parse=_isles22),
    "episurg":       dict(csv="metadata_episurg.csv",       ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "EPISURG/subjects")], exclude=["-seg-"], parse=_episurg),
    "ixi":           dict(csv="metadata_ixi.csv",           ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "IXI")], exclude=[], parse=_ixi),
    "oasis1":        dict(csv="metadata_oasis1.csv",        ext=[".hdr"],
                          roots=[D(DATASETS_ROOT, "OASIS-1")], keep=["RAW", "_mpr-1_"],
                          exclude=[], parse=_oasis("OASIS1", "OASIS-1")),
    "oasis2":        dict(csv="metadata_oasis2.csv",        ext=[".hdr"],
                          roots=[D(DATASETS_ROOT, "OASIS-2")], keep=["mpr-1.nifti.hdr"],
                          exclude=[], parse=_oasis("OASIS2", "OASIS-2")),
    "nfbs":          dict(csv="metadata_nfbs.csv",          ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "NFBS")], keep=["brain.nii.gz"], exclude=[], parse=_nfbs),
    "brainmetshare": dict(csv="metadata_brainmetshare.csv", ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "BrainMetShare/brainmetshare-3")], exclude=["seg"],
                          parse=_brainmetshare),
    "umfpd":         dict(csv="metadata_umfpd.csv",         ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "umf_pd")], keep=["/anat/"], exclude=[], parse=_umfpd),
    "3dmrms":        dict(csv="metadata_3dmrms.csv",        ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "3D-MR-MS/all")],
                          exclude=["/raw/", "_gt", "brainmask"], parse=_mrms),
    "bgsp":          dict(csv="metadata_bgsp.csv",          ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "BGSP/orig_bids")], keep=["/anat/", "_T1w"],
                          exclude=[], parse=_bgsp),
    "nskgbm":        dict(csv="metadata_nskgbm.csv",        ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "NSK-GBM")], keep=["/anat/"],
                          exclude=["_swi", "_seg"], parse=_nskgbm),
    "upenngbm":      dict(csv="metadata_upenngbm.csv",      ext=[".nii.gz"],
                          roots=[D(DATASETS_ROOT, "PKG - UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/images_structural_unstripped")],
                          exclude=["/old/"], parse=_upenn),
}

# NSK-GBM: keep only the 4 structural contrasts
_NSK_OK = ("_flair.", "_t1ce.", "_t1w.", "_t2w.")


def run(name):
    spec = DATASETS[name]
    if spec["parse"] is None:
        print(f"{name}: no parser (dataset not available), skipped")
        return
    paths = []
    for r in spec["roots"]:
        if os.path.isdir(r):
            paths += get_paths(r, file_extension_list=spec["ext"])
    paths = [p for p in paths if not any(x in p for x in spec.get("exclude", []))]
    for k in spec.get("keep", []):
        paths = [p for p in paths if k in p]
    if name == "nskgbm":
        paths = [p for p in paths if any(t in p.lower() for t in _NSK_OK)]
    if not paths:
        print(f"{name}: no images found under {spec['roots']}, skipped")
        return
    collect_metadata(sorted(paths), D(METADATA_DIR, spec["csv"]), spec["parse"])
    print(f"{name}: wrote {spec['csv']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    names = list(DATASETS) if args == ["--all"] else args
    if not names:
        print(__doc__)
        print("names:", ", ".join(DATASETS))
        sys.exit(0)
    for n in names:
        if n not in DATASETS:
            print(f"unknown dataset: {n}")
            continue
        print(f"== {n} ==")
        run(n)

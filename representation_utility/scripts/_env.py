"""Shared paths and settings. Import first: keeps model caches local, picks the device."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

os.environ.setdefault("HF_HOME", str(ROOT / "weights" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch_cache"))

# optional secrets from .env (KEY=VALUE)
_envfile = ROOT / ".env"
if _envfile.is_file():
    for line in _envfile.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

METADATA_DIR = REPO / "characterization" / "metadata"
DATASETS_DIR = Path("/Volumes/BACH2TB/Datasets")
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
LOG = ROOT / "logs"
WEIGHTS = ROOT / "weights"
for _d in (OUT, FIG, LOG, WEIGHTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42


def get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = get_device()

if __name__ == "__main__":
    print("ROOT     :", ROOT)
    print("METADATA :", METADATA_DIR, "ok" if METADATA_DIR.is_dir() else "MISSING")
    print("DATASETS :", DATASETS_DIR, "ok" if DATASETS_DIR.is_dir() else "NOT MOUNTED")
    print("HF_HOME  :", os.environ["HF_HOME"])
    print("HF_TOKEN :", "set" if os.environ.get("HF_TOKEN") else "not set")
    print("DEVICE   :", DEVICE)

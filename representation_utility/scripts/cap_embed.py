"""CAP E2: embed every pilot variant -> outputs/cap_pairs_{rep}.parquet (raw emb_* + shared z_* space).

The z_* space is debias.fit_feature_pipeline (StandardScaler -> PCA-100 -> StandardScaler)
fit on the original 4167 embeddings, fit split = debias_P_<rep>.npz fit_rows (the same
patient-disjoint 60% split used by removal_eval / INLP / LEACE / CORAL).

Usage: python cap_embed.py [--reps untrained medical] [--batch 4]
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEVICE, OUT  # noqa: E402
from cap_simulate import variants  # noqa: E402
from debias import fit_feature_pipeline  # noqa: E402
from extract_embeddings import build_model  # noqa: E402

META = ["image_id", "dataset", "factor", "level"]


def out_path(rep):
    return OUT / f"cap_pairs_{rep}.parquet"


def simulate(row):
    return row, [(f, lvl, x) for f, lvl, ok, x in variants(row["image_id"], row["path"]) if ok]


def embed(fwd, arrs, batch):
    vecs = []
    with torch.no_grad():
        for s in range(0, len(arrs), batch):
            x = torch.from_numpy(np.stack(arrs[s:s + batch])).unsqueeze(1).to(DEVICE)
            vecs.append(fwd(x).float().cpu().numpy())
    return np.vstack(vecs)


def project(rep):
    """Add z_* columns: the removal_eval feature space, fit on the original embeddings' fit split."""
    idx = pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True)
    emb = pd.read_parquet(OUT / f"emb_{rep}.parquet")
    cols = [c for c in emb.columns if c.startswith("emb_")]
    X_raw = idx[["image_id"]].merge(emb, on="image_id")[cols].to_numpy(float)
    fit_rows = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)["fit_rows"]
    _, (sc1, pca, sc2) = fit_feature_pipeline(X_raw, fit_rows)

    def to_z(X):
        Z = sc1.transform(X)
        return sc2.transform(pca.transform(Z) if pca is not None else Z)

    df = pd.read_parquet(out_path(rep))
    df = df[[c for c in df.columns if not c.startswith("z_")]]
    Z = to_z(df[cols].to_numpy(float))
    df = pd.concat([df, pd.DataFrame(Z, columns=[f"z_{i}" for i in range(Z.shape[1])], index=df.index)], axis=1)
    df.to_parquet(out_path(rep), index=False)

    orig = df[df["factor"] == "original"].merge(emb[["image_id", *cols]], on="image_id", suffixes=("", "_ref"))
    a, b = orig[cols].to_numpy(float), orig[[f"{c}_ref" for c in cols]].to_numpy(float)
    za, zb = to_z(a), to_z(b)
    print(f"[{rep}] {len(df)} rows, NaN={int(df.isna().sum().sum())}, "
          f"fit split = debias_P_{rep}.npz fit_rows ({len(fit_rows)} of {len(X_raw)} volumes); "
          f"re-embedded original vs emb_{rep}.parquet: max |dz| = {np.abs(za - zb).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["untrained", "medical"])
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    pilot = pd.read_csv(OUT / "cap_pilot_ids.csv")
    done = {r: set(pd.read_parquet(out_path(r))["image_id"]) if out_path(r).exists() else set()
            for r in args.reps}
    todo = pilot[[any(i not in done[r] for r in args.reps) for i in pilot["image_id"]]]
    print(f"{len(todo)} of {len(pilot)} pilot volumes to embed, device={DEVICE}")

    models = {r: build_model(r)[1] for r in args.reps}
    new = {r: [] for r in args.reps}
    rows = [r for _, r in todo.iterrows()]
    with ThreadPoolExecutor(2) as pool:
        futs = [pool.submit(simulate, r) for r in rows[:2]]
        for k in range(1, len(rows) + 1):
            row, var = futs.pop(0).result()
            if k + len(futs) < len(rows):
                futs.append(pool.submit(simulate, rows[k + len(futs)]))
            meta = pd.DataFrame([(row["image_id"], row["dataset"], f, lvl) for f, lvl, _ in var], columns=META)
            for r in args.reps:
                if row["image_id"] in done[r]:
                    continue
                v = embed(models[r], [x for _, _, x in var], args.batch)
                new[r].append(pd.concat([meta, pd.DataFrame(v, columns=[f"emb_{i}" for i in range(v.shape[1])])],
                                        axis=1))
            if k % 5 == 0 or k == len(todo):
                for r in args.reps:
                    if new[r]:
                        old = [pd.read_parquet(out_path(r))[lambda d: d.columns[~d.columns.str.startswith("z_")]]] \
                            if out_path(r).exists() else []
                        pd.concat(old + new[r], ignore_index=True).to_parquet(out_path(r), index=False)
                        new[r] = []
                print(f"{k}/{len(todo)} volumes ({row['dataset']})", flush=True)

    for r in args.reps:
        project(r)


if __name__ == "__main__":
    main()

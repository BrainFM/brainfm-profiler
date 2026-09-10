"""Frozen deep embeddings per volume -> outputs/emb_<model>.parquet.

Usage: python extract_embeddings.py --model untrained|medical|selfsup [--batch 4] [--workers 4]
  untrained = random-init 3D DenseNet121
  medical   = MedicalNet ResNet-34, pretrained
  selfsup   = SwinUNETR self-supervised encoder
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, DEVICE, SEED, WEIGHTS  # noqa: E402
from io_utils import prepare  # noqa: E402


def build_model(name):
    from monai.networks.nets import DenseNet121, SwinUNETR, resnet34
    torch.manual_seed(SEED)
    if name == "untrained":
        m = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
        feats = m.features

        def fwd(x):
            return torch.flatten(F.adaptive_avg_pool3d(F.relu(feats(x)), 1), 1)
    elif name == "medical":
        m = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=True,
                     feed_forward=False, shortcut_type="A", bias_downsample=True)
        fwd = m
    elif name == "selfsup":
        m = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_checkpoint=False)
        m.load_from(torch.load(WEIGHTS / "model_swinvit.pt", map_location="cpu", weights_only=False))
        swin = m.swinViT

        def fwd(x):
            return torch.flatten(F.adaptive_avg_pool3d(swin(x, normalize=True)[4], 1), 1)
    else:
        raise NotImplementedError(name)
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, fwd


def sanity_pca(name, emb_df):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from results_lib import save_fig

    cols = [c for c in emb_df.columns if c.startswith("emb_")]
    z = (emb_df[cols] - emb_df[cols].mean()) / (emb_df[cols].std() + 1e-8)
    xy = PCA(2, random_state=SEED).fit_transform(z)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "ds": emb_df["dataset"].values})
    for ds, s in plot_df.groupby("ds"):
        ax.scatter(s["x"], s["y"], s=6, alpha=0.5, label=ds)
    ax.set(title=f"{name} embeddings — PCA(2)", xlabel="PC1", ylabel="PC2")
    ax.legend(fontsize=6, ncol=2, markerscale=2)
    save_fig(fig, f"sanity_pca_{name}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["untrained", "medical", "selfsup"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_path = OUT / f"emb_{args.model}.parquet"
    idx = pd.read_parquet(OUT / "file_index.parquet")
    done = set(pd.read_parquet(out_path)["image_id"]) if out_path.exists() else set()
    todo = idx[~idx["image_id"].isin(done)].reset_index(drop=True)
    print(f"{args.model}: {len(todo)} to embed ({len(done)} done), device={DEVICE}")

    if len(todo) == 0:
        print("nothing to embed; regenerating sanity figure")
        sanity_pca(args.model, pd.read_parquet(out_path))
        return

    model, fwd = build_model(args.model)
    paths, ids = todo["path"].tolist(), todo["image_id"].tolist()
    starts = list(range(0, len(paths), args.batch))

    out_ids, vecs = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, torch.no_grad():
        pending = pool.map(prepare, paths[starts[0]:starts[0] + args.batch]) if starts else iter(())
        for k, s in enumerate(starts):
            arrs = list(pending)
            nxt = starts[k + 1] if k + 1 < len(starts) else None
            if nxt is not None:
                pending = pool.map(prepare, paths[nxt:nxt + args.batch])
            x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).to(DEVICE)
            v = fwd(x).float().cpu().numpy()
            out_ids.extend(ids[s:s + args.batch])
            vecs.append(v)
            if (k + 1) % 50 == 0 or k + 1 == len(starts):
                print(f"{len(out_ids)}/{len(todo)}")

    new = pd.DataFrame(np.vstack(vecs), columns=[f"emb_{i}" for i in range(vecs[0].shape[1])])
    new.insert(0, "image_id", out_ids)
    if out_path.exists():
        new = pd.concat([pd.read_parquet(out_path), new], ignore_index=True)
    new = idx[["image_id", "dataset", "domain", "patient", "modality"]].merge(new, on="image_id")
    new.to_parquet(out_path, index=False)
    dim = new.filter(like="emb_").shape[1]
    print(f"emb_{args.model}.parquet: {len(new)}/{len(idx)} rows, dim={dim}")

    sanity_pca(args.model, new)


if __name__ == "__main__":
    main()

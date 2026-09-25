"""Frozen embeddings from current brain-MRI foundation models -> outputs/emb_<model>.parquet.

Input rule (Option A): the shared Chapter 2 geometry (canonical orientation,
2 mm, 128^3 crop/pad), then only each model's own resize and intensity scaling.
No extra skull stripping or registration.

  brainiac = BrainIAC ViT-B/16, 96^3, z-score on non-zero voxels, CLS token (768)
  3dino    = 3DINO ViT-L/16, 112^3, 0.05/99.95 percentiles -> [-1, 1], CLS token (1024)
  brainfm  = BrainFM 3D U-Net encoder, 128^3 at 2 mm, min-max [0, 1],
             deepest encoder map, mean-pooled (2048)
  sammed3d = SAM-Med3D-turbo ViT-B/16 image encoder, 128^3 at 2 mm, the shared
             z-score (same as its own ZNormalization on x > 0), neck output
             (the image embedding, 384 x 8^3), mean-pooled (384)
Controls for 3DINO:
  3dino_rand = same ViT-L, random weights (seed 42), same input
  3dino_z    = 3DINO weights, input z-scored on non-zero voxels (BrainIAC scaling)
Controls for the Chapter 2 CNNs (same 128^3 geometry, 3DINO percentile scaling
instead of the shared z-score):
  untrained_pct, medical_pct

Usage: python extract_fm_embeddings.py --model brainiac|3dino|brainfm|3dino_rand|3dino_z|untrained_pct|medical_pct|sammed3d [--batch 4] [--limit N]
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEVICE, OUT, ROOT, SEED, WEIGHTS  # noqa: E402
from io_utils import center_crop_pad, load_volume, normalize, resample_iso  # noqa: E402

EXT = ROOT / "external"
CKPT = {
    "brainiac": WEIGHTS / "brainiac" / "BrainIAC.ckpt",
    "3dino": WEIGHTS / "3dino" / "3dino_vit_weights.pth",
    "brainfm": WEIGHTS / "brainfm" / "assets" / "brainfm_pretrained.pth",
}
CKPT["3dino_rand"] = CKPT["3dino_z"] = CKPT["3dino"]
CKPT["sammed3d"] = WEIGHTS / "sammed3d" / "sam_med3d_turbo.pth"
CKPT["untrained_pct"] = CKPT["medical_pct"] = None


def geometry(path):
    data, zooms = load_volume(path)
    return center_crop_pad(resample_iso(data, zooms))


def _resize(x, size):
    return x if x.shape[-1] == size else F.interpolate(x, size=(size,) * 3, mode="trilinear", align_corners=False)


def scale_brainiac(x):
    out = torch.zeros_like(x)
    for i in range(len(x)):
        m = x[i] != 0
        v = x[i][m]
        out[i][m] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def scale_3dino(x):
    out = torch.empty_like(x)
    for i in range(len(x)):
        lo, hi = torch.quantile(x[i].flatten().cpu(), torch.tensor([0.0005, 0.9995])).tolist()
        out[i] = torch.clip((x[i] - lo) / (hi - lo + 1e-8) * 2 - 1, -1, 1)
    return out


def scale_brainfm(x):
    out = torch.empty_like(x)
    for i in range(len(x)):
        v = x[i] - x[i].min()
        out[i] = v / (v.max() + 1e-8)
    return out


def _check_load(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing keys: {missing[:5]} ... ({len(missing)})")
    return unexpected


def build_model(name):
    torch.manual_seed(SEED)
    if name == "brainiac":
        from monai.networks.nets import ViT
        m = ViT(in_channels=1, img_size=(96,) * 3, patch_size=(16,) * 3, hidden_size=768, mlp_dim=3072,
                num_layers=12, num_heads=12, save_attn=True)
        ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
        ck = ck.get("state_dict", ck)
        sd = {k[len("backbone."):]: v for k, v in ck.items() if k.startswith("backbone.")}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # newer MONAI builds an unused cross-attention norm; nothing else may be missing
        if unexpected or any("norm_cross_attn" not in k for k in missing):
            raise RuntimeError(f"brainiac load: missing {missing[:5]}, unexpected {unexpected[:5]}")

        def fwd(x):
            return m(scale_brainiac(_resize(x, 96)))[0][:, 0]
    elif name.startswith("3dino"):
        sys.path.insert(0, str(EXT / "3DINO"))
        from dinov2.models.vision_transformer import vit_large_3d
        m = vit_large_3d(img_size=112, patch_size=16, init_values=1.0e-5, ffn_layer="mlp", block_chunks=4,
                         qkv_bias=True, proj_bias=True, ffn_bias=True)
        if name != "3dino_rand":
            ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
            ck = ck.get("teacher", ck)
            ck = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck.items()}
            extra = _check_load(m, ck)
            print(f"{name}: ignored {len(extra)} non-backbone keys (e.g. {extra[:2]})")
        scale = scale_brainiac if name == "3dino_z" else scale_3dino

        def fwd(x):
            return m(scale(_resize(x, 112)))
    elif name == "brainfm":
        import types
        # load only the U-Net; the package __init__ files pull in the whole training stack
        for pkg in ("Trainer", "Trainer.models"):
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(EXT / "BrainFM" / pkg.replace(".", "/"))]
            sys.modules.setdefault(pkg, mod)
        from Trainer.models.unet3d.model import UNet3D
        m = UNet3D(1, 64, layer_order="gcl", num_groups=8, num_levels=6, is_unit_vector=True)
        ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
        ck = ck.get("model", ck)
        prefix = next(p for p in ("backbone.", "module.backbone.", "") if any(k.startswith(p + "encoders.") for k in ck))
        _check_load(m, {k[len(prefix):]: v for k, v in ck.items() if k.startswith(prefix)})

        def fwd(x):
            h = scale_brainfm(x)
            for enc in m.encoders:
                h = enc(h)
            return h.mean(dim=(2, 3, 4))
    elif name == "sammed3d":
        import importlib.util
        from functools import partial
        f = EXT / "SAM-Med3D" / "segment_anything" / "modeling" / "image_encoder3D.py"
        spec = importlib.util.spec_from_file_location("sammed3d_encoder", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.ImageEncoderViT3D(depth=12, embed_dim=768, img_size=128, mlp_ratio=4,
                                  norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12,
                                  patch_size=16, qkv_bias=True, use_rel_pos=True,
                                  global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=384)
        ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
        ck = ck.get("model_state_dict", ck)
        m.load_state_dict({k[len("image_encoder."):]: v for k, v in ck.items()
                           if k.startswith("image_encoder.")}, strict=True)

        def fwd(x):
            z = torch.stack([torch.from_numpy(normalize(v[0].cpu().numpy())) for v in x]).unsqueeze(1).to(x.device)
            return m(z).mean(dim=(2, 3, 4))
    elif name.endswith("_pct"):
        import extract_embeddings
        m, base_fwd = extract_embeddings.build_model(name.removesuffix("_pct"))

        def fwd(x):
            return base_fwd(scale_3dino(x))
        return m, fwd
    else:
        raise NotImplementedError(name)
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPT))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = OUT / f"emb_{args.model}.parquet"
    if args.limit:
        out_path = OUT / f"emb_{args.model}_test.parquet"
    idx = pd.read_parquet(OUT / "file_index.parquet")
    done = set(pd.read_parquet(out_path)["image_id"]) if out_path.exists() else set()
    todo = idx[~idx["image_id"].isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.groupby("dataset").head(1).head(args.limit).reset_index(drop=True)
    print(f"{args.model}: {len(todo)} to embed ({len(done)} done), device={DEVICE}")
    if len(todo) == 0:
        return

    model, fwd = build_model(args.model)
    paths, ids = todo["path"].tolist(), todo["image_id"].tolist()
    starts = list(range(0, len(paths), args.batch))

    out_ids, vecs, n_done = [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, torch.no_grad():
        pending = pool.map(geometry, paths[0:args.batch])
        for k, s in enumerate(starts):
            arrs = list(pending)
            nxt = starts[k + 1] if k + 1 < len(starts) else None
            if nxt is not None:
                pending = pool.map(geometry, paths[nxt:nxt + args.batch])
            x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).float().to(DEVICE)
            v = fwd(x).float().cpu().numpy()
            out_ids.extend(ids[s:s + args.batch])
            vecs.append(v)
            if (k + 1) % 50 == 0 or k + 1 == len(starts):
                n_done += len(out_ids)
                print(f"{n_done}/{len(todo)}", flush=True)
                _save(out_path, todo, out_ids, vecs)
                out_ids, vecs = [], []


def _save(out_path, todo, out_ids, vecs):
    new = pd.DataFrame(np.vstack(vecs), columns=[f"emb_{i}" for i in range(vecs[0].shape[1])])
    meta = todo.set_index("image_id").loc[out_ids, ["dataset", "domain", "patient", "modality"]].reset_index()
    new = pd.concat([meta, new], axis=1)
    if out_path.exists():
        new = pd.concat([pd.read_parquet(out_path), new], ignore_index=True)
    new.to_parquet(out_path, index=False)
    nan = int(new.filter(like="emb_").isna().any(axis=1).sum())
    print(f"saved {out_path.name}: {len(new)} rows, {nan} with NaN", flush=True)


if __name__ == "__main__":
    main()

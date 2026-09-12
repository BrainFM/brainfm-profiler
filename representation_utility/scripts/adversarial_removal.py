"""Non-linear removal: does the acquisition confound survive even when the
removal operator is allowed to be non-linear? INLP and LEACE only rule out
*linear* operators; this tests whether that's the reason they failed, or
whether the negative result is deeper than "linear methods aren't expressive
enough."

A small domain-adversarial network (DANN: Ganin & Lempitsky, 2015) is trained
on top of the same 100-d PCA feature space INLP/LEACE use:
  - feature head g_theta: 100 -> 128 (GELU) -> 100, the non-linear transform
    being learned (unlike INLP/LEACE's linear-by-construction P).
  - domain head: g(Z) -> 128 (GELU) -> n_datasets, predicting `dataset`
    through a gradient-reversal layer, so g_theta is pushed to make dataset
    identity harder to decode.
  - retention head: g(Z) -> 64 (GELU) -> n_modalities, predicting `modality`
    from a *detached* copy of g(Z) -- it monitors whether content survives,
    but (unlike an earlier draft of this script) never backpropagates into
    g_theta. INLP/LEACE only ever use modality as a stop-check, never as a
    training signal for the operator itself; letting the retention head
    shape g_theta would let it exploit modality labels from *all* 17
    datasets' fit-split patients -- including some patients of whichever
    dataset a later LODO fold holds out -- which is a real leak the linear
    methods don't have. Detaching keeps the comparison fair: g_theta is
    trained only to fool the domain adversary and stay close to Z.
  - a small proximity penalty ||g(Z) - Z||^2 (LEACE's "minimal collateral
    damage" spirit), preventing g_theta from distorting the space arbitrarily.
  - lambda (gradient-reversal weight) ramped per the standard DANN schedule
    lambda_p = 2/(1+exp(-10p)) - 1, p = training progress in [0,1].

Control: an *untrained*, randomly-initialized g_theta of identical
architecture -- a random non-linear transform of matched capacity, the
non-linear analogue of the random-projection control.

Reuses INLP's saved fit_rows/eval_rows (outputs/debias_P_<rep>.npz), fixed
seed everywhere, first use of torch in this pipeline outside embedding
extraction (DEVICE convention from _env.py).

Writes results/tables/adversarial_{decodability,transfer,retrieval,residual}.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEVICE, OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import RETAIN, REMOVE_ORDER, factor_frame, fit_feature_pipeline  # noqa: E402
from removal_eval import decodability_perm, lodo_transfer, nonlinear_residual, retrieval_purity  # noqa: E402
from results_lib import add_number, save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
EPOCHS = 300
LR = 1e-3
PROX_WEIGHT = 0.01
HIDDEN_G, HIDDEN_DOM, HIDDEN_RET = 128, 128, 64


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class FeatureHead(nn.Module):
    def __init__(self, d, hidden=HIDDEN_G):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, d, n_out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_adversarial(Z, ds_fit, mod_mask_fit, mod_fit, d, n_ds, n_mod, seed=SEED):
    torch.manual_seed(seed)
    g = FeatureHead(d).to(DEVICE)
    dom_head = Head(d, n_ds, HIDDEN_DOM).to(DEVICE)
    ret_head = Head(d, n_mod, HIDDEN_RET).to(DEVICE)
    opt = torch.optim.Adam(list(g.parameters()) + list(dom_head.parameters()) + list(ret_head.parameters()), lr=LR)

    Zt = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
    dst = torch.tensor(ds_fit, dtype=torch.long, device=DEVICE)
    modt = torch.tensor(mod_fit, dtype=torch.long, device=DEVICE)
    mmask = torch.tensor(mod_mask_fit, dtype=torch.bool, device=DEVICE)

    for epoch in range(EPOCHS):
        p = epoch / max(EPOCHS - 1, 1)
        lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        opt.zero_grad()
        h = g(Zt)
        dom_logits = dom_head(grad_reverse(h, lambd))
        ret_logits = ret_head(h.detach()[mmask])  # monitor only -- no gradient into g
        loss_dom = F.cross_entropy(dom_logits, dst)
        loss_ret = F.cross_entropy(ret_logits, modt[mmask])
        loss_prox = ((h - Zt) ** 2).mean()
        loss = loss_dom + loss_ret + PROX_WEIGHT * loss_prox
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                dom_acc = (dom_logits.argmax(1) == dst).float().mean().item()
                ret_acc = (ret_logits.argmax(1) == modt[mmask]).float().mean().item()
            print(f"    epoch {epoch:3d}  lambda={lambd:.2f}  dataset_acc={dom_acc:.2f} "
                  f"modality_acc={ret_acc:.2f}  loss={loss.item():.3f}")
    return g


def make_transform(net):
    net.eval()

    def transform(Z, ds):
        with torch.no_grad():
            Zt = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
            return net(Zt).cpu().numpy().astype(Z.dtype)
    return transform


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows, eval_rows = npz["fit_rows"], npz["eval_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    sub_fit = ff.iloc[fit_rows].reset_index(drop=True)
    ds_le = LabelEncoder().fit(ff["dataset"])
    ds_fit = ds_le.transform(sub_fit["dataset"])
    mod_mask_fit = sub_fit["modality"].notna().to_numpy()
    mod_le = LabelEncoder().fit(sub_fit.loc[mod_mask_fit, "modality"])
    mod_fit = np.zeros(len(sub_fit), dtype=int)
    mod_fit[mod_mask_fit] = mod_le.transform(sub_fit.loc[mod_mask_fit, "modality"])

    print(f"\n=== {rep} (adversarial, d={d}, {len(ds_le.classes_)} datasets, "
          f"{len(mod_le.classes_)} modalities) ===")
    print("  training:")
    g = train_adversarial(Z[fit_rows], ds_fit, mod_mask_fit, mod_fit, d,
                          len(ds_le.classes_), len(mod_le.classes_))

    torch.manual_seed(SEED + 1)  # different seed -> a genuinely different random init for the control
    g_control = FeatureHead(d).to(DEVICE)

    transform = make_transform(g)
    transform_rand = make_transform(g_control)

    print("  decodability (before/after/null):")
    dec_rows = decodability_perm(Z, ff, eval_rows, transform, [RETAIN, *REMOVE_ORDER])
    for r in dec_rows:
        r["representation"] = rep

    lodo = lodo_transfer(Z, ff, transform, transform_rand)
    lodo["representation"] = rep
    print(f"  LODO modality transfer: before={lodo[lodo.condition=='before'].accuracy.mean():.3f} "
          f"after={lodo[lodo.condition=='after'].accuracy.mean():.3f} "
          f"random-net-control={lodo[lodo.condition=='random'].accuracy.mean():.3f}")

    ret = retrieval_purity(Z, ff, transform)
    ret["representation"] = rep
    print(f"  retrieval purity: {ret.to_dict('records')}")

    res = nonlinear_residual(Z, ff, eval_rows, transform)
    res["representation"] = rep
    print(f"  non-linear residual (dataset, MLP): "
          f"before={res[res.condition=='before'].balanced_acc.iloc[0]:.3f} "
          f"after={res[res.condition=='after'].balanced_acc.iloc[0]:.3f}")

    torch.save(g.state_dict(), OUT / f"adversarial_{rep}.pt")
    return dec_rows, lodo, ret, res


def main():
    all_dec, all_lodo, all_ret, all_res = [], [], [], []
    for rep in REPS:
        dec, lodo, ret, res = run(rep)
        all_dec += dec
        all_lodo.append(lodo)
        all_ret.append(ret)
        all_res.append(res)

    save_table(pd.DataFrame(all_dec), "adversarial_decodability")
    save_table(pd.concat(all_lodo, ignore_index=True), "adversarial_transfer")
    save_table(pd.concat(all_ret, ignore_index=True), "adversarial_retrieval")
    save_table(pd.concat(all_res, ignore_index=True), "adversarial_residual")

    summary = pd.concat(all_lodo, ignore_index=True)
    for rep in REPS:
        s = summary[summary.representation == rep]
        add_number(f"adversarial LODO ({rep})",
                   f"before={s[s.condition=='before'].accuracy.mean():.3f} "
                   f"after={s[s.condition=='after'].accuracy.mean():.3f} "
                   f"random-net-control={s[s.condition=='random'].accuracy.mean():.3f}")


if __name__ == "__main__":
    reps = REPS
    if "--reps" in sys.argv:
        reps = sys.argv[sys.argv.index("--reps") + 1:]
    if reps == REPS:
        main()
    else:
        for r in reps:
            run(r)

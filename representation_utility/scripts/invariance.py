"""Objective-level invariance instead of representation-level erasure.

Every method in Part B/C (INLP, LEACE, CORAL, adversarial) acts the same way:
leave the classifier's training objective alone (plain ERM), transform the
representation, then check if LODO transfer improves. None did.

This tries the other lever: leave the representation untouched, change how the
probe is *trained* -- Group-DRO (Sagawa et al., "Distributionally Robust Neural
Networks", ICLR 2020), treating each training-side dataset as a "group"/
environment and minimising the worst-group loss instead of the pooled average.
If a more invariant decision boundary exists in the untouched representation,
this is a different, and cheaper, way to find it than deleting directions.

Same LODO protocol as removal_eval.lodo_transfer (leave-one-dataset-out on
modality, same Z, same held-out accuracy metric) so the numbers are directly
comparable to removal_comparison.csv -- this is a 5th arm at the objective
level, not the representation level.

Writes results/tables/invariance_lodo.csv.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT, SEED  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import factor_frame, fit_feature_pipeline  # noqa: E402
from results_lib import save_table  # noqa: E402

REPS = ["handcrafted", "untrained", "medical", "selfsup"]
MODS = ["T1", "T1c", "T2", "FLAIR"]
N_STEPS = 500
LR = 0.05
ETA_Q = 0.01
WEIGHT_DECAY = 3e-3
MIN_GROUP_N = 5  # drop training-side datasets with fewer labeled rows than this


def train_group_dro(Z, y, groups, n_classes, d, seed=SEED):
    """Full-batch Group-DRO: exponentiated-gradient-ascent group weights q,
    gradient descent on sum_g q_g * CE_g(theta). Linear model == same capacity
    as the LogisticRegression used everywhere else in this project."""
    torch.manual_seed(seed)
    X = torch.tensor(Z, dtype=torch.float32)
    classes, y_idx = np.unique(y, return_inverse=True)
    yt = torch.tensor(y_idx, dtype=torch.long)

    names, counts = np.unique(groups, return_counts=True)
    keep = names[counts >= MIN_GROUP_N]
    group_idx = [torch.tensor(np.where(groups == g)[0], dtype=torch.long) for g in keep]
    if not group_idx:  # fall back to plain ERM if nothing survives the group-size filter
        group_idx = [torch.arange(len(y))]

    model = nn.Linear(d, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    q = torch.ones(len(group_idx)) / len(group_idx)

    for _ in range(N_STEPS):
        opt.zero_grad()
        logits = model(X)
        losses = torch.stack([F.cross_entropy(logits[gi], yt[gi]) for gi in group_idx])
        with torch.no_grad():
            q = q * torch.exp(ETA_Q * losses)
            q = q / q.sum()
        loss = (q * losses).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (model(X).argmax(1) == yt).float().mean().item()
    return model, classes, train_acc


def predict_group_dro(model, classes, Z):
    with torch.no_grad():
        idx = model(torch.tensor(Z, dtype=torch.float32)).argmax(1).numpy()
    return classes[idx]


def lodo_invariance(Z, ff, d):
    dff = ff[ff["modality"].isin(MODS)]
    rows_idx = dff.index.to_numpy()
    y_all, ds_all = dff["modality"].to_numpy(), dff["dataset"].to_numpy()
    Zr = Z[rows_idx]
    n_classes = len(MODS)

    out = []
    for held in sorted(set(ds_all)):
        tr, te = ds_all != held, ds_all == held
        if te.sum() == 0 or len(set(y_all[tr])) < 2:
            continue
        k = len(set(y_all[te]))
        sc = StandardScaler().fit(Zr[tr])
        Ztr, Zte = sc.transform(Zr[tr]), sc.transform(Zr[te])

        erm = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Ztr, y_all[tr])
        erm_acc = accuracy_score(y_all[te], erm.predict(Zte))

        model, classes, train_acc = train_group_dro(Ztr, y_all[tr], ds_all[tr], n_classes, d)
        dro_pred = predict_group_dro(model, classes, Zte)
        dro_acc = accuracy_score(y_all[te], dro_pred)

        out.append(dict(dataset=held, n_modalities=k, condition="ERM", accuracy=erm_acc))
        out.append(dict(dataset=held, n_modalities=k, condition="GroupDRO", accuracy=dro_acc))
        print(f"    held={held:16s} n_mod={k}  ERM={erm_acc:.3f}  GroupDRO={dro_acc:.3f}"
              f"  (GroupDRO train_acc={train_acc:.3f})")
    return pd.DataFrame(out)


def run(rep):
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep(rep)
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)

    npz = np.load(OUT / f"debias_P_{rep}.npz", allow_pickle=True)
    fit_rows = npz["fit_rows"]
    Z, _ = fit_feature_pipeline(X_raw, fit_rows)
    d = Z.shape[1]

    print(f"[{rep}] d={d}")
    res = lodo_invariance(Z, ff, d)
    res["representation"] = rep
    return res


def main():
    all_rows = []
    for rep in REPS:
        print(f"=== {rep} ===")
        all_rows.append(run(rep))
    res = pd.concat(all_rows, ignore_index=True)
    save_table(res, "invariance_lodo")

    print("\nmean LODO accuracy, ERM vs GroupDRO:")
    summ = res.groupby(["representation", "condition"])["accuracy"].mean().unstack()
    print(summ)


if __name__ == "__main__":
    main()

"""Is the Group-DRO effect on `medical` real, or one lucky run?

Wilcoxon signed-rank on the 17 paired (ERM, GroupDRO) LODO deltas per
representation, plus a re-run of `medical` with two different seeds to check
the isolated-dataset pattern isn't an artefact of one initialisation.

Writes results/tables/invariance_stability.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import OUT  # noqa: E402
from analyze_representations import add_qc_factors, load_rep  # noqa: E402
from debias import factor_frame, fit_feature_pipeline  # noqa: E402
from invariance import REPS, lodo_invariance  # noqa: E402
from results_lib import save_table  # noqa: E402

ISOLATED_MULTI = ["3D-MR-MS", "MS-60", "BrainMetShare"]  # K3's clearest isolated, multi-modality cases


def main():
    lodo = pd.read_csv(OUT.parent / "results" / "tables" / "invariance_lodo.csv")
    rows = []

    # 1) paired Wilcoxon per representation, all 17 targets
    for rep in REPS:
        sub = lodo[lodo.representation == rep]
        piv = sub.pivot(index="dataset", columns="condition", values="accuracy")
        delta = piv["GroupDRO"] - piv["ERM"]
        stat, p = wilcoxon(delta)
        rows.append(dict(check="wilcoxon_all17", representation=rep,
                          mean_delta=float(delta.mean()), p=float(p), n=len(delta)))
        d3 = delta.loc[[d for d in ISOLATED_MULTI if d in delta.index]]
        rows.append(dict(check="mean_delta_isolated_multi", representation=rep,
                          mean_delta=float(d3.mean()), p=np.nan, n=len(d3)))
        print(f"[{rep}] Wilcoxon all-17 mean_delta={delta.mean():+.3f} p={p:.3f}  "
              f"| isolated-multi mean_delta={d3.mean():+.3f} ({dict(d3)})")

    # 2) stability re-run of `medical` at two more seeds
    idx = add_qc_factors(pd.read_parquet(OUT / "file_index.parquet").reset_index(drop=True))
    ff = factor_frame(idx)
    d_emb, cols = load_rep("medical")
    X_raw = idx[["image_id"]].merge(d_emb, on="image_id")[cols].to_numpy(float)
    npz = np.load(OUT / "debias_P_medical.npz", allow_pickle=True)
    Z, _ = fit_feature_pipeline(X_raw, npz["fit_rows"])
    d = Z.shape[1]

    for seed in [43, 44]:
        import invariance as inv
        inv.SEED = seed  # only affects train_group_dro's torch.manual_seed via default arg binding below
        print(f"\n--- medical, seed={seed} ---")
        res = lodo_invariance_seeded(Z, ff, d, seed)
        piv = res.pivot(index="dataset", columns="condition", values="accuracy")
        delta = piv["GroupDRO"] - piv["ERM"]
        d3 = delta.loc[[dd for dd in ISOLATED_MULTI if dd in delta.index]]
        rows.append(dict(check=f"stability_seed{seed}_all17", representation="medical",
                          mean_delta=float(delta.mean()), p=np.nan, n=len(delta)))
        rows.append(dict(check=f"stability_seed{seed}_isolated_multi", representation="medical",
                          mean_delta=float(d3.mean()), p=np.nan, n=len(d3)))
        print(f"  seed={seed} mean_delta_all17={delta.mean():+.3f}  isolated-multi={d3.mean():+.3f} ({dict(d3)})")

    save_table(pd.DataFrame(rows), "invariance_stability")


def lodo_invariance_seeded(Z, ff, d, seed):
    """lodo_invariance, but train_group_dro seeded differently (monkeypatch SEED default)."""
    import invariance as inv
    orig = inv.train_group_dro
    def seeded(Z_, y_, groups_, n_classes_, d_, seed=seed):
        return orig(Z_, y_, groups_, n_classes_, d_, seed=seed)
    inv.train_group_dro = seeded
    try:
        return lodo_invariance(Z, ff, d)
    finally:
        inv.train_group_dro = orig


if __name__ == "__main__":
    main()

"""Step 8: turn the measured results into guidelines + a hold-out recommendation table."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_lib import RESULTS, save_table  # noqa: E402

HOLDOUT_THRESH = 0.30


def main():
    lodo = pd.read_csv(RESULTS / "tables" / "lodo_transfer.csv")
    rec = lodo[["dataset", "domain", "n_modalities", "transfer_score"]].copy()
    rec = rec.rename(columns={"transfer_score": "lodo_transfer_medical"})
    rec["role"] = rec["lodo_transfer_medical"].apply(
        lambda v: "hold-out (isolated)" if v < HOLDOUT_THRESH else "pool")
    rec = rec.sort_values("lodo_transfer_medical")
    save_table(rec.round(3), "holdout_recommendation")

    isolated = rec[rec["role"].str.startswith("hold-out")]["dataset"].tolist()

    (RESULTS / "guidelines.md").write_text(GUIDELINES.format(
        isolated=", ".join(isolated),
        n_isolated=len(isolated),
        thr=f"{HOLDOUT_THRESH:.2f}",
    ))
    print("wrote results/guidelines.md")
    print(rec.round(3).to_string(index=False))


GUIDELINES = """# Guidelines for building a brain-MRI representation-learning corpus

Based on 4,167 volumes from 17 public adult brain-MRI datasets, embedded with
four frozen representations (hand-crafted statistics, an untrained 3D CNN, a
medically pretrained 3D CNN, a self-supervised 3D transformer).

## What the analysis showed

1. **Datasets are trivially separable, and it is not a learned effect.**
   A linear probe recovers the source dataset from a frozen embedding at
   ~0.92-0.94 balanced accuracy (chance 0.06). An *untrained* network does this
   as well as the pretrained ones, and hand-crafted image statistics are within
   2 points. The separation lives in raw intensity, texture and geometry.

2. **Acquisition, not content, is what is encoded.** Orientation (0.82-0.88),
   anisotropy (0.87-0.95) and scanner identity (0.98-0.99 within one dataset)
   are all near-perfectly decodable from every representation.

3. **The one shared content-ish label, modality, does not transfer.**
   Within a dataset a probe tells T1/T1c/T2/FLAIR apart at ~0.89; trained on
   16 datasets and tested on the 17th it drops to ~0.44. The representation's
   notion of "a T1" is scanner-specific.

4. **Some datasets are isolated.** For {isolated} the leave-one-dataset-out
   modality accuracy is below {thr}: the rest of the corpus cannot reconstruct
   them. The clearest cases are the multi-modality ones - **3D-MR-MS, MS-60,
   BrainMetShare** (a real 3-4 class task, still failing). NFBS and OASIS-1 are
   single-modality, so their score is a degenerate 1-class number, but both are
   also qualitatively out-of-distribution (NFBS is skull-stripped brain-only in
   a mostly non-stripped T1 pool; OASIS-1 is old raw MPRAGE).

5. **Adding datasets barely changes cross-dataset transfer** (in this frozen-
   encoder, modality-probe setting). Removing any one dataset from the training
   pool moves held-out accuracy by <1 point for almost all datasets. This does
   *not* prove more data is useless for pretraining - that needs training
   encoders on varied mixtures and evaluating downstream - but it shows dataset
   count and nominal diversity are poor selection criteria for learning
   representations of this kind.

## Recommendations

**R1. Do not select datasets by size or by how many diseases/modalities they
cover.** Neither predicts contribution to transfer here.

**R2. Reserve the isolated datasets ({n_isolated} of 17: {isolated}) as
external-validation sets.** They are the honest generalisation test precisely
because representations trained on the other datasets do not explain them.

**R3. Publish a corpus datasheet.** For any pretraining corpus, report, per
source dataset: modality mix, number of scanners/sites, orientation-code
distribution, voxel-spacing distribution (and anisotropic fraction), number of
subjects, and whether images are skull-stripped / registered. These are the
axes along which a model can shortcut.

**R4. Run an acquisition-confound audit before pretraining.** Embed the corpus
with a *frozen, untrained* 3D CNN, then linear-probe (patient-level CV,
label-permutation null) for: source dataset, scanner/site, orientation,
voxel-spacing regime. If these are highly decodable - they will be - the corpus
carries a strong acquisition signal that a model will use. Re-run the audit
after any harmonisation to check it was reduced.

**R5. Expect an acquisition shortcut and plan for it.** Options (not evaluated
here, future work): intensity/geometry harmonisation, site-balanced sampling,
domain-adversarial or site-invariance objectives during pretraining, and
reporting downstream results per source dataset rather than pooled.

## Scope and limitations

Frozen off-the-shelf encoders (no encoder trained per dataset or per mixture);
n = 17 datasets; modality is the only shared label and is itself acquisition-
linked; 2 mm / 128^3 geometry; adult structural MRI only. The transfer and
contribution results are specific to modality classification with frozen
encoders and should not be read as statements about foundation-model
pretraining in general.
"""


if __name__ == "__main__":
    main()

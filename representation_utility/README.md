# Dataset-specific directions limit transfer of brain MRI foundation model features

Code and result tables for the preprint **Dataset-Specific Directions Limit Transfer of Brain MRI Foundation Model Features** (Luu and Tuchinov, 2026; arXiv link to be added).

Frozen features of medical foundation models are known to retain where a scan came from and to lose accuracy on new datasets. This work asks why. Eight frozen representations are evaluated on 4,167 volumes from 17 public brain MRI datasets by predicting the MRI sequence (T1, T1c, T2, FLAIR) on a dataset held out from training.

## Main findings

- Every representation identifies the source dataset (balanced accuracy 0.84–0.95) and recognizes the sequence within a dataset (0.84–0.97).
- On an unseen dataset, only the broadly pretrained 3DINO (0.72) and SAM-Med3D (0.69) keep the sequence. All others, including two brain-specific foundation models, fall to 0.44–0.55. Controls attribute the advantage to pretraining.
- Transfer follows **direction consistency**: whether the directions that separate the sequences agree across datasets. It explains which held-out datasets fail (for the sequence and for sex), ranks representations from the training datasets alone, and, computed on an unlabelled new dataset, warns of failure better than confidence-based estimators.
- Removing the dataset signal (INLP, LEACE, CORAL, adversarial), each against a matched control, never improves sequence transfer, because dataset and sequence share feature directions. A centring test shows when alignment can help instead.

## Data

The 17 datasets are public and must be downloaded from their original sources under their own licences (see the paper for the list and references). Neither the images nor the extracted embeddings are redistributed here, because several licences do not permit sharing data derived from the scans.

- `results/tables/sampled_scans.csv`: the 4,167 scans used (dataset, patient, image ID, sequence).
- `configs/datasets.yaml`: the datasets and how their metadata CSVs (in `../characterization/metadata/`) map to local paths. The CSVs store absolute paths from our machines; use `path_rewrite` to point them at your copy.

## Representations

| Group | Representations |
|---|---|
| Baselines | Handcrafted image statistics, untrained 3D DenseNet-121, MedicalNet (ResNet-34), SwinUNETR (self-supervised on CT) |
| Brain-specific foundation models | BrainIAC, BrainFM |
| Broadly pretrained medical foundation models | 3DINO, SAM-Med3D |

Controls: 3DINO with random weights, 3DINO with z-score input, MedicalNet with percentile input.

All volumes are reoriented to RAS, resampled to 2 mm and cropped or padded to 128³. Each encoder then gets only its own input size and intensity scaling (see `scripts/extract_fm_embeddings.py`).

## Installation

```bash
git clone https://github.com/BrainFM/brainfm-profiler.git
cd brainfm-profiler/representation_utility
python -m venv .venv
source configs/env.sh              # activates .venv, keeps model caches in weights/
pip install -r requirements.txt
cp .env.example .env               # optional: add a Hugging Face token
```

The foundation model code is not included. Clone it into `external/` and put the checkpoints in `weights/` (paths are listed in `scripts/extract_fm_embeddings.py`):

- [BrainIAC](https://github.com/AIM-KannLab/BrainIAC)
- [BrainFM](https://github.com/jhuldr/BrainFM)
- [3DINO](https://github.com/AICONSlab/3DINO)
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)

## Reproducing the results

Run from this folder after `source configs/env.sh`:

```bash
# 1. index of all volumes and the per-dataset sample
python scripts/build_index.py

# 2. features and labels
python scripts/features_r1.py                                       # handcrafted
python scripts/extract_embeddings.py --model untrained              # also: medical, selfsup
python scripts/extract_fm_embeddings.py --model 3dino               # also: brainiac, brainfm, sammed3d,
                                                                    #       3dino_rand, 3dino_z, medical_pct
python scripts/qc_factors.py                                        # per-scan acquisition factors
python scripts/content_labels.py                                    # age and sex, where available

# 3. analyses
python scripts/benchmark_fm.py                # dataset decodability, leave-one-dataset-out transfer
python scripts/within_dataset_fm.py           # sequence decodability within each dataset
python scripts/sequence_consistency.py        # direction consistency and centring test, sequence
python scripts/lodo_age_sex.py                # sex and age transfer
python scripts/sex_consistency.py             # direction consistency and centring test, sex
python scripts/failure_prediction.py          # label-free failure prediction
python scripts/removal_lodo.py --reps 3dino   # dataset-signal removal, one run per representation,
python scripts/removal_lodo.py --merge        #   then merge

# 4. figures
python scripts/fig_chapter2.py                # transfer per encoder, identification vs transfer
```

`benchmark_fm.py` reuses `results/tables/decodability.csv` and `pairwise_transfer.csv` from earlier analyses; both are committed.

## Where each result comes from

| Result in the paper | Script | Table in `results/tables/` |
|---|---|---|
| Dataset and acquisition decodability | `benchmark_fm.py` | `decodability_fm.csv` |
| Sequence transfer to an unseen dataset | `benchmark_fm.py` | `lodo_transfer_fm.csv`, `benchmark_fm_summary.csv` |
| Sequence within each dataset | `within_dataset_fm.py` | `within_dataset_fm.csv` |
| Direction consistency and centring, sequence | `sequence_consistency.py` | `sequence_consistency{,_summary}.csv` |
| Sex transfer | `lodo_age_sex.py` | `lodo_age_sex{,_summary}.csv` |
| Direction consistency and centring, sex | `sex_consistency.py` | `sex_consistency{,_summary}.csv` |
| Label-free failure prediction | `failure_prediction.py` | `failure_prediction{,_summary}.csv` |
| Dataset-signal removal | `removal_lodo.py` | `removal_lodo{,_summary}.csv` |

The committed tables are the ones behind the reported numbers. Figures are written to `results/figures/`. Other scripts in `scripts/` belong to earlier analyses and are not needed for the paper.

## Citation

> Luu, M.S.K.; Tuchinov, B.N.
> Dataset-Specific Directions Limit Transfer of Brain MRI Foundation Model Features. Preprint, 2026.

The dataset selection follows our earlier review, which should also be cited if you use the dataset metadata:

> Luu, M.S.K.; Benedichuk, M.V.; Roppert, E.I.; Kenzhin, R.M.; Tuchinov, B.N.
> A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development.
> *Journal of Imaging* **2025**, *11*(12), 454. https://doi.org/10.3390/jimaging11120454

## License

The code is released under the [MIT License](../LICENSE). The datasets and the foundation models keep their own licences and terms of use.

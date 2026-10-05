# brainfm-profiler

Code and results for profiling public brain MRI datasets for foundation model development.

This repository accompanies the paper
**[A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development](https://www.mdpi.com/2313-433X/11/12/454)**
(*Journal of Imaging*, 2025, 11(12), 454).
It also contains new experiments added after the paper was published: a cross-dataset benchmark of frozen representations from eight encoders, including brain-specific and general medical foundation models, on 17 public datasets (`representation_utility/`).

## What is in this repository

| Folder | Contents | Used in |
|---|---|---|
| `characterization/` | Literature search, per-scan metadata extraction (shape, voxel spacing, orientation, intensity), image-level analysis, and the corpus composition figure | Paper and new experiments |
| `characterization/metadata/` | One CSV per dataset with the per-scan descriptors | Paper and new experiments |
| `preprocessing/` | Preprocessing pipeline: histogram-matching intensity normalization, N4 bias field correction (SimpleITK), FSL BET skull stripping, and ANTs SyN registration to MNI152 | Paper |
| `covariate_shift/` | Residual covariate shift between NFBS and IXI after preprocessing, using DenseNet121 features of a central axial slice | Paper |
| `figures/` | Preprocessing example figures from the paper | Paper |
| `representation_utility/` | Frozen-representation benchmark: embedding extraction, dataset and sequence probes, leave-one-dataset-out transfer, and removal of the dataset signal | New experiments |

## Frozen-representation benchmark

The new experiments ask whether frozen features encode the MRI sequence in the same way across datasets, so that it transfers to a dataset not seen in training.

**Data.** 17 public adult brain MRI datasets (tumor, multiple sclerosis, stroke, epilepsy, neurodegenerative, healthy). Up to about 300 volumes are sampled per dataset by patient, 4,167 volumes in total. The dataset list is in `representation_utility/configs/datasets.yaml`.

**Representations.** Eight frozen encoders, none fine-tuned:

| Group | Representations |
|---|---|
| Baselines | Handcrafted image statistics, untrained 3D DenseNet-121, MedicalNet (ResNet-34), SwinUNETR (self-supervised on CT) |
| Brain-specific foundation models | BrainIAC, BrainFM |
| Broadly pretrained medical foundation models | 3DINO, SAM-Med3D |

Three controls are also included: 3DINO with random weights, 3DINO with z-score input, and MedicalNet with percentile input.

**Protocol.** All volumes are reoriented to RAS, resampled to 2 mm and cropped or padded to 128³. Each encoder then gets its own input size and intensity scaling. Linear probes on the features predict the source dataset and the sequence (T1, T1c, T2, FLAIR), with patient-level splits. Transfer is measured by leave-one-dataset-out: train on 16 datasets and test on the held-out one, scored by balanced accuracy.

**Main findings.**

- Every representation identifies the source dataset well (balanced accuracy 0.84–0.95, chance 0.06), including the untrained network and the handcrafted statistics.
- Within a single dataset, every representation recognizes the sequence (0.84–0.97). On an unseen dataset, most fall to 0.44–0.55, including both brain-specific foundation models.
- 3DINO (0.72) and SAM-Med3D (0.69) transfer best. Controls show that this comes from pretraining, not from the architecture, input scaling, skull stripping, or pretraining-data overlap.
- Removing the dataset signal (INLP, LEACE, CORAL, adversarial) inside each training split does not improve transfer for any representation, because dataset and sequence share the same feature directions.

## Installation

```bash
git clone https://github.com/BrainFM/brainfm-profiler.git
cd brainfm-profiler
```

For the paper code (`characterization/`, `preprocessing/`, `covariate_shift/`):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`preprocessing/` also needs [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) for skull stripping.

For the frozen-representation benchmark:

```bash
cd representation_utility
python -m venv .venv
source configs/env.sh              # activates .venv, keeps model caches in weights/
pip install -r requirements.txt
cp .env.example .env               # optional: add a Hugging Face token
```

The foundation model code is not included. Clone it into `representation_utility/external/` and put the checkpoints in `representation_utility/weights/` (paths are listed in `scripts/extract_fm_embeddings.py`):

- [BrainIAC](https://github.com/AIM-KannLab/BrainIAC)
- [BrainFM](https://github.com/jhuldr/BrainFM)
- [3DINO](https://github.com/AICONSlab/3DINO)
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)

## Data

The datasets are not redistributed. Download them from their original sources, listed in the paper. The metadata CSVs in `characterization/metadata/` store absolute image paths from our machines. Set your own root in `characterization/collect_metadata.py`, or use `path_rewrite` in `representation_utility/configs/datasets.yaml`.

## Frozen embeddings

The extracted embeddings for all representations will be available for download here: **link TBA**.
Place the files in `representation_utility/outputs/` to rerun the analyses without extracting features again.

## Reproducing the benchmark

Run from `representation_utility/` after `source configs/env.sh`:

```bash
# 1. index of all volumes and the per-dataset sample
python scripts/build_index.py

# 2. features (skip if you downloaded the embeddings)
python scripts/features_r1.py                                       # handcrafted
python scripts/extract_embeddings.py --model untrained              # also: medical, selfsup
python scripts/extract_fm_embeddings.py --model 3dino               # also: brainiac, brainfm, sammed3d,
                                                                    #       3dino_rand, 3dino_z, medical_pct
python scripts/qc_factors.py                                        # per-scan acquisition factors
python scripts/content_labels.py                                    # age and sex, where available

# 3. analyses
python scripts/benchmark_fm.py                # decodability, leave-one-dataset-out transfer, retrieval
python scripts/within_dataset_fm.py           # sequence decodability within each dataset
python scripts/lodo_age_sex.py                # sex and age transfer
python scripts/removal_lodo.py --reps 3dino   # dataset-signal removal, one run per representation,
python scripts/removal_lodo.py --merge        #   then merge

# 4. figures
python scripts/fig_chapter2.py
```

Result tables (CSV and LaTeX) are written to `representation_utility/results/tables/` and figures to `representation_utility/results/figures/`. The committed tables are the ones behind the reported numbers.

## Citation

If you use this code or its findings, please cite:

> Luu, M.S.K.; Benedichuk, M.V.; Roppert, E.I.; Kenzhin, R.M.; Tuchinov, B.N.
> A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development.
> *Journal of Imaging* **2025**, *11*(12), 454. https://doi.org/10.3390/jimaging11120454

```bibtex
@article{luu2025brainmri,
  title     = {A Structured Review and Quantitative Profiling of Public Brain {MRI} Datasets for Foundation Model Development},
  author    = {Luu, Minh Sao Khue and Benedichuk, Margaret V. and Roppert, Ekaterina I. and Kenzhin, Roman M. and Tuchinov, Bair N.},
  journal   = {Journal of Imaging},
  volume    = {11},
  number    = {12},
  pages     = {454},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/jimaging11120454},
  url       = {https://www.mdpi.com/2313-433X/11/12/454}
}
```

Citation metadata is also in [CITATION.cff](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The public datasets and the foundation models used here keep their own licenses and terms of use.

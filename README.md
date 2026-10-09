# brainfm-profiler

Code and results for profiling public brain MRI datasets for foundation model development.

This repository accompanies the paper
**[A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development](https://www.mdpi.com/2313-433X/11/12/454)**
(*Journal of Imaging*, 2025, 11(12), 454).
It also contains new experiments added after the paper was published: a cross-dataset benchmark of frozen representations from eight encoders, including brain-specific and general medical foundation models, on 17 public datasets (`representation_utility/`). These experiments are described in the preprint **Dataset-Specific Directions Limit Transfer of Brain MRI Foundation Model Features** (arXiv link to be added).

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

The benchmark of frozen representations on 17 public datasets has its own README with findings, installation, reproduction steps and citation: **[representation_utility/README.md](representation_utility/README.md)**.

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

## Data

The datasets are not redistributed. Download them from their original sources, listed in the paper. The metadata CSVs in `characterization/metadata/` store absolute image paths from our machines. Set your own root in `characterization/collect_metadata.py`, or use `path_rewrite` in `representation_utility/configs/datasets.yaml`.

## Citation

If you use the dataset review and profiling code, please cite:

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

Citation metadata is also in [CITATION.cff](CITATION.cff). For the frozen-representation benchmark, cite the preprint listed in [representation_utility/README.md](representation_utility/README.md#citation).

## License

The code is released under the [MIT License](LICENSE). The public datasets and the foundation models used here keep their own licenses and terms of use.

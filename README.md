# A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development

This repository accompanies the paper:
**“A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development”**.

[Link for full paper](https://doi.org/10.3390/jimaging11120454)

It contains all source code used for:

- Image-level analysis (e.g., voxel spacing, orientation, intensity distribution)
- Preprocessing pipelines (bias correction, skull stripping, registration, etc.)
- Visualization and statistics of cohort heterogeneity
- Quantification of residual covariate shift in feature space


## Getting Started

1. Clone the Repository

```bash
git clone https://github.com/BrainFM/brainfm-profiler.git
cd brainfm-profiler
```

2. Install Dependencies


```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Functional Modules

| Module                    | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `preprocessing/`      | Tools for N4 bias correction, FSL BET skull stripping, linear registration  |
| `characterization/`   | Scripts for computing voxel spacing stats, orientation formats, intensity histograms |
| `covariate_shift/`    | Feature extraction via pre-trained 3D DenseNet, shift quantification        |
| `utils/`              | Helper functions for loading NIfTI, computing stats, saving plots           |

## Outputs

- All manuscript figures are saved to `figures/` folder
- Processed CSV summaries and exploratory analyses can be found in `notebooks/` folder

## Citation

If you use this code or its findings in your research, please cite:

> Luu, M.S.K.; Benedichuk, M.V.; Roppert, E.I.; Kenzhin, R.M.; Tuchinov, B.N.
> A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development.
> *Journal of Imaging* **2025**, *11*, 454. https://doi.org/10.3390/jimaging11120454

BibTeX:

```bibtex
@article{luu2025brainmri,
  title   = {A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development},
  author  = {Luu, Minh Sao Khue and Benedichuk, Margaret V. and Roppert, Ekaterina I. and Kenzhin, Roman M. and Tuchinov, Bair N.},
  journal = {Journal of Imaging},
  volume  = {11},
  number  = {12},
  pages   = {454},
  year    = {2025},
  publisher = {MDPI},
  doi     = {10.3390/jimaging11120454}
}
```


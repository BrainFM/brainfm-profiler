# A Multi-Level Characterization of Public Brain Imaging Cohorts for Foundation Model Development

This repository accompanies the paper:
**“A Multi-Level Characterization of Public Brain Imaging Cohorts for Foundation Model Development”**.

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
| `src/preprocessing/`      | Tools for N4 bias correction, FSL BET skull stripping, linear registration  |
| `src/characterization/`   | Scripts for computing voxel spacing stats, orientation formats, intensity histograms |
| `src/covariate_shift/`    | Feature extraction via pre-trained 3D DenseNet, shift quantification        |
| `src/utils/`              | Helper functions for loading NIfTI, computing stats, saving plots           |

## Outputs

- All manuscript figures are saved to `figures/` folder
- Processed CSV summaries and exploratory analyses can be found in `notebooks/` folder

## Citation

TBD


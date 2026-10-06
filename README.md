# Luna

**How much spatial-scale information can be recovered from sparse sensor observations of a physical field?**

This repository hosts the code that accompanies the paper *Multiscale recoverability in sparse-sensor flow reconstruction* . It provides (i) the core evaluation metrics, (ii) the reconstruction methods compared in the paper, and (iii) a one-command pipeline that regenerates the paper's statistics and figures.

`TRACEABILITY.md` maps every figure, table and statistic of the paper to the module that produces it; the docstring of each script states what it computes, which artifacts it reads and how to run it.

---

## What the framework measures

Given a target field and a reconstruction obtained from sparse sensor observations, the framework decomposes both into wavelet scale bands and quantifies, per band, how faithfully the scale content is recovered:

| Quantity | Meaning |
|---|---|
| `GER_u` | streamwise-component relative error `‖u−û‖/‖u‖`; the wavelet identity and the scale diagnostics are defined on it |
| `GER` | relative error of the full two-component state; used for the model-level comparisons |
| per-band errors | relative error of each wavelet band (`A4`, `W4`, …, `W1` for a level-4 DWT) on the streamwise component |
| `S_full` | number of consecutive bands recovered from the coarsest one, each below the tolerance τ |
| `S_coh` | the same count applied to the POD-dominant band error, where every band is projected onto a band-specific POD basis fitted to the same field |

Both counts are evaluated on the streamwise velocity; the two-component state enters the model-level `GER` and the estimators.

The reconstruction methods evaluated in the paper:

- **POD-Ridge / POD-MLP** — sensor observations are mapped (linear / MLP) to rank-`r` POD coefficients;
- **VCNN** — a convolutional network operating directly on the sparse observation grid;
- **Gappy POD** — POD with gappy data reconstruction (rank selected on validation data and capped by the number of scalar observations, i.e. twice the number of sensor locations; the location cap is retained as a sensitivity);
- **rank-`r` POD truncation** — reference baselines.

## Repository layout

```
luna/                            core library (no script dependencies)
  wavelet/                       DWT decomposition, bands, scale-count metrics (GER, S_full, S_coh)
  pod/                           POD decomposition, truncation reference, band-POD
  models/                        POD-Ridge, POD-MLP, VCNN
  data/                          I/O + dataset registry + mask loading
  benchmarks/                    analytical multiscale wake benchmark
features/                        domain library
  sensors/                       sensor masks and the observations built from them
  training/                      training and evaluation of the estimators
  metrics/                       band-wise metric variants
  statistics/                    resampling and temporal-dependence statistics
applications/                    paper pipeline
  pipelines/                     numbered entry points: data -> POD bases -> train -> statistics -> figures
  statistics/                    one module per statistic of the paper
  figures/                       one module per figure of the paper
  configs/, config.py            experiment definitions (YAML)
configs/                         dataset definitions (TOML)
environment/                     conda environment definition
tests/unit/                      pytest unit tests (metric identities, DWT orthogonality, ...)
scripts/                         helper scripts (data fetch, full reproduction)
Makefile                         one-click entry points
```

Every statistic lives in `artifacts/statistics/` and is written by one module in `applications/statistics/`; every figure lives in `artifacts/figures/` and is drawn by one module in `applications/figures/`. `TRACEABILITY.md` maps each figure, table and statistic of the paper to those modules.

The file names of the figure scripts follow the order in which the figures were first drawn and are kept stable, so the number in a file name is not always the number in the paper: `fig09_cross_model_bands.py` draws main figure 5 and `fig05_wavelet_vs_fourier.py` draws supplementary figure S6. The mapping is the table in `TRACEABILITY.md`.

## Datasets

The experiments use three public field datasets. The raw sources are listed below; the exact arrays used by the experiments are derived from them (see `scripts/download_data.sh` for layout requirements and processing notes).

| Short name | Field | Raw public source |
|---|---|---|
| **NC** | 2-D unsteady cylinder wake (Re = 160, von Kármán vortex street) | ETH Zürich CGL *2D Unsteady Cylinder Flow* — Gerris simulation; cite Günther, Gross & Theisel, *Generic Objective Vortices for Flow Visualization*, ACM TOG / SIGGRAPH 2017 ([data page](https://cgl.ethz.ch/research/visualization/data.php)) |
| **RDB** | 2-D shallow-water **radial dam break** (128×128) | PDEBench, *PDEBench Datasets* on DaRUS — DOI [10.18419/DARUS-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2986) (`2D/shallow-water/2D_rdb_NA_NA.h5`) |
| **SST** | weekly sea-surface temperature (NOAA OISST, 180×360) | NOAA OISST field; packaged `.mat` file from *The Senseiver Dataset*, Zenodo — DOI [10.5281/zenodo.8290040](https://zenodo.org/records/8290040) |

The experimental arrays (cropped / normalized npy snapshots, sensor masks, POD bases) and all trained models are **not** distributed in this repository. They are regenerated locally into git-ignored directories (`data/`, `masks*/`, `artifacts/`) — see *Reproducing the paper* below.

## Installation

Requires Python ≥ 3.11 (a conda environment is recommended):

```bash
conda env create -f environment/environment.yml -n luna
conda activate luna
```

Core dependencies: `numpy`, `scipy`, `matplotlib`, `PyWavelets`,
`scikit-learn`, `torch`, `pandas`, `pyyaml`, `h5py`, `pytest`.

## Quick start (no data required)

The **analytical benchmark** validates the metrics on a synthetic, NC-inspired multiscale wake field whose ground-truth scale content is strictly known. It is fully self-contained and is the recommended starting point:

```bash
make demo            # runs applications.statistics.analytical_benchmark
```

This writes `artifacts/statistics/analytical_benchmark.json` and prints the six benchmark cases together with the scale count the construction prescribes.

Run the unit tests (metric identities, DWT orthogonality, S_full / S_coh consistency, three-layer error decomposition, ridge closed form, gappy rank behaviour):

```bash
make test            # pytest tests/unit
```

`make test` needs no data: the tests that exercise the five sampled sensor-mask families skip when those masks have not been generated locally (they are large sampling patterns and are not shipped with the code).

## Reproducing the paper

The full pipeline regenerates the statistics and the figures reported in the paper:

```bash
make reproduce       # = scripts/reproduce_all.sh
```

Prerequisites, in order:

1. **Raw data** — fetch and prepare the three dataset arrays into `data/` (see `scripts/download_data.sh`).
2. **POD bases** — the truncation basis every estimator is expressed in is rebuilt from the raw arrays by `02`.
3. **Trained models** — the learned estimators (POD-Ridge, POD-MLP, VCNN) are produced by the training code under `features/training/` and stored under `artifacts/`. Because training all configurations is compute-intensive, the paper's published numbers were generated from these artifacts; the pipeline re-runs every analysis step on whatever artifacts are present.
4. **Run the pipeline** — the entry points are numbered in the order they have to run:

   ```bash
   python applications/pipelines/01_prepare_data.py              # raw arrays of data/
   python applications/pipelines/02_build_pod_bases.py           # POD bases (rank 128)
   python applications/pipelines/03_train_estimators.py --check  # verify the runs on disk
   python applications/pipelines/04_compute_statistics.py        # every statistic
   python applications/pipelines/05_make_figures.py              # every figure
   ```

   The statistics step runs the producers of `applications/statistics/` in dependency order (`--list` shows them, `--only` runs a subset).

Outputs are written to git-ignored directories; they are never committed.

## Policy of this repository

This is a **source-code-only** repository. Raw data, derived intermediate products, trained models, figures, and the paper sources with the internal working notes are intentionally excluded from version control (via `.gitignore`) and are not pushed. Everything needed to go from code to the paper's quantitative results is either included (code, configs, tests) or regenerable from the public data sources listed above.

## Citation

If you use this code or the accompanying methodology, please cite the paper (citation to be added when published) and, when applicable, the data sources listed under *Datasets*.

## License

[MIT](LICENSE)

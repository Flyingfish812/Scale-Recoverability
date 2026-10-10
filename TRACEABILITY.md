# TRACEABILITY — paper objects ↔ source code

This file maps every object of the paper to the code that produces it: which module writes a statistic, which artifact holds it, and which figure, table or section consumes it.

The paper is *Multiscale recoverability in sparse-sensor flow reconstruction*. Its LaTeX sources and the numerical fact layer (`data/paper_facts.yaml`) live with the paper sources, not in this repository; everything on the code side is here.

## 1. Pipeline

```
raw data           data/*.npy                        ← public datasets (§4)
reconstruction     artifacts/pod_bases/, artifacts/<estimator runs>/
statistics         artifacts/statistics/*.json       ← one producer per statistic
figures            artifacts/figures/*.pdf           ← one script per figure
paper objects      main figures 1–6, table 1, supplementary figures S1–S14,
                   supplementary tables S1–S25
```

| Step | Entry point | Writes |
|---|---|---|
| 00 | `features/sensors/incremental_masks.py` | the sensor masks under `masks2/` and `masks_families/` |
| 01 | `applications/pipelines/01_prepare_data.py` | checks the raw arrays under `data/` |
| 02 | `applications/pipelines/02_build_pod_bases.py` | `artifacts/pod_bases/<dataset>/pod_base_bundle.npz` |
| 03 | `applications/pipelines/03_train_estimators.py` | the estimator runs under `artifacts/` |
| 04 | `applications/pipelines/04_compute_statistics.py` | `artifacts/statistics/` |
| 05 | `applications/pipelines/05_make_figures.py` | `artifacts/figures/` |

Each step can be run on its own; `--list` on step 04 shows the statistics it
runs, `--only` on step 05 runs a subset of the figure scripts. The pipeline never writes into the tracked tree. Step 00 draws one candidate set per family from the seeds listed in `features/sensors/mask_registry.py`, and `--exclude-cylinder-body` drops candidates inside the cylinder section, where the field is identically zero.

## 2. Figures

Each script draws one figure, or one group of figures, from the artifacts of the statistics layer. The file names follow the order in which the figures were first drawn, so the number in the file name is not the number in the paper; the paper number is given below.

| Paper figure | Script | Reads |
|---|---|---|
| main 1 | `fig01_known_scale.py` | `analytical_benchmark.json`; the field is the deterministic construction in `luna.benchmarks.analytical_wake` |
| main 2 | `fig02_global_vs_scale.py` | `band_error_decomposition.json`, `band_error_records.json`, `equal_ger_pairs.json` |
| main 3 | `fig03_sensor_noise.py` | `band_error_decomposition.json`, `sensor_noise_phase.json` |
| main 4 | `fig05_modal_hierarchy.py` | `mode_scale_energy.json`, `modal_coefficient_error.json`, `pod_bases/cylinder2d_q1/pod_base_bundle.npz` |
| main 5 | `fig09_cross_model_bands.py` | `band_error_records.json`, `truncation_reference_audit.json` |
| main 6 | `fig06_robustness.py` | `wavelet_sensitivity.json`, `threshold_sensitivity.json` |
| S1 | `fig01_method_framework.py` | — (schematic, written as SVG then converted to PDF) |
| S2 | `fig02_analytical_benchmark.py` | `analytical_benchmark.json` |
| S3 | `figS01_oracle.py` | `truncation_reference_audit.json` |
| S4 | `figS04_three_layer.py` | `band_error_records.json` |
| S5 | `fig03_counterexample.py` | `band_error_decomposition.json`, one trained MLP run |
| S6 | `fig05_wavelet_vs_fourier.py` | `fourier_band_baseline.json`, `transform_symmetry_check.json` |
| S7, S8 | `figS02_phase.py` | `band_error_records.json`, `sensor_noise_phase.json` |
| S9 | `figS05_mode_scale_energy.py` | `mode_scale_energy.json` |
| S10, S12 | `figS03_diagnostics.py` | `noise_propagation.json`, `level_sensitivity.json` |
| S11 | `figS03c_pod_dominant.py` | one trained VCNN run, band-POD bundle |
| S13 | `figS07_sensor_family_ger.py` | `sensor_family/sensor_count_effect.csv` |
| S14 | `figS08_sensor_family_paired.py` | `paired_model_comparison.json` |

`style.py` holds the shared publication style, `records.py` the record views the figure scripts share, and `publish_figures.py` copies the PDFs into the directory that holds the paper sources.

## 3. Statistics

Every statistic of the paper is one JSON file under `artifacts/statistics/`, written by the module of the same name in `applications/statistics/`. The table lists what each producer computes and the paper objects it feeds; the numbers themselves are registered with their source in the paper's fact layer.

| Producer | Artifact | Computes | Feeds |
|---|---|---|---|
| `analytical_benchmark.py` | `analytical_benchmark.json` | six prescribed-carrier-removal cases: measured scale count and per-band errors | main fig. 1, supplementary tables S1–S2 |
| `band_error_records.py` | `band_error_records.json` | the canonical per-snapshot records of every estimator, sensor count and noise level (48 000 records) | model comparison table, main figs. 3 and 5, supplementary figs. S4, S7–S8, S14 |
| `band_error_decomposition.py` | `band_error_decomposition.json` | per-band error decomposition of the two error terms for the displayed configurations | main figs. 2–3, supplementary table S9 |
| `truncation_reference_audit.py` | `truncation_reference_audit.json` | rank-128 truncation reference and the rank-adequacy audit on NC, RDB and SST | main fig. 5, supplementary figs. S3, tables S3–S6, S19 |
| `sensor_noise_phase.py` | `sensor_noise_phase.json` | pass probabilities and mean scale counts over the sensor-count × noise grid | main fig. 3, supplementary figs. S7–S8, tables S14–S15 |
| `mode_scale_energy.py` | `mode_scale_energy.json` | band energy carried by each POD mode | main fig. 4a, supplementary fig. S9 |
| `modal_coefficient_error.py` | `modal_coefficient_error.json` | per-mode coefficient error against mode energy | main fig. 4b–d, supplementary tables S16–S17 |
| `per_mode_nrmse.py` | `per_mode_nrmse.json` | energy-decile error table and the band recovery rates | supplementary tables S16, S18 |
| `excess_error_recompute.py` | `excess_error_recompute.json` | excess band error above the truncation floor | supplementary table S19 |
| `level_sensitivity.py` | `level_sensitivity.json` | scale count for decomposition levels 3–5 | supplementary fig. S12 |
| `noise_propagation.py` | `noise_propagation.json` | per-band degradation ratio between noise levels | supplementary fig. S10 |
| `threshold_sensitivity.py` | `threshold_sensitivity.json` | scale count at τ = 0.03, 0.05, 0.08 | main fig. 6b, supplementary table S20 |
| `wavelet_sensitivity.py` | `wavelet_sensitivity.json` | scale count in five wavelet bases | main fig. 6a, supplementary table S21 |
| `wavelet_family_sensitivity.py` | `wavelet_family_sensitivity.json` | scale count per wavelet family, derivative errors | supplementary table S21 |
| `fourier_band_baseline.py` | `fourier_band_baseline.json` | Fourier-dyadic count for the same reconstructions | supplementary fig. S6, tables S12–S13 |
| `transform_symmetry_check.py` | `transform_symmetry_check.json` | controlled-field agreement of the two transform-domain counts | supplementary fig. S6, tables S12–S13 |
| `equal_ger_pairs.py` | `equal_ger_pairs.json`, `equal_ger_pairs_strict.json` | one-to-one pairs with matching streamwise global error and different scale counts, with paired and bootstrap statistics | main fig. 2b–d, supplementary tables S8, S10 |
| `within_config_physics_bootstrap.py` | `within_config_physics_bootstrap.json` | within-configuration correlation of scale count with physical error | supplementary table S11 |
| `ger_band_correlation.py` | `ger_band_correlation.json` | within-configuration correlation of the global error with band error | supplementary tables S10–S11 |
| `scoh_vs_sfull.py` | `scoh_vs_sfull.json` | comparison of the full-band and POD-dominant scale counts | main text discussion of the two indices |
| `band_pod_energy_sensitivity.py` | `band_pod_energy_sensitivity.json` | sensitivity of the POD-dominant index to the retained band energy | supplementary table S17 |
| `seed_stability.py` | `seed_stability.json` | spread of the scale indices over training seeds | supplementary tables S22–S24 |
| `seed_audit.py` | `seed_audit.json` | seed-level audit of the per-snapshot fields | supplementary table S24 |
| `paired_model_comparison.py` | `paired_model_comparison.json` | paired MLP–VCNN differences with time-block intervals | supplementary fig. S14 |
| `sensor_family_summary.py` | `sensor_family/sensor_count_effect.csv`, `sensor_family_summary.json` | sensor-count trend over independently sampled sensor families | supplementary figs. S13–S14 |
| `temporal_dependence.py` | `temporal_dependence.json` | temporal correlation of the test snapshots (block length) | supplementary table S23 |
| `blocked_holdout_audit.py` | `blocked_holdout_audit.json` | scale count and error under contiguous temporal holdouts | supplementary table S22 |
| `boundary_sensitivity.py` | `boundary_sensitivity.json` | sensitivity to the wavelet boundary convention | main text robustness discussion |
| `low_ger_analysis.py` | `low_ger_analysis.json` | lower-half global-error records with low scale counts | supplementary table S8 |
| `tau_pairwise_checks.py` | `tau_pairwise_checks.json` | pairwise threshold checks of the ranking | supplementary table S20 |
| `band_denominator_check.py` | `band_denominator_check.json` | stability of the band denominators | supplementary table S25 |
| `rank_sensitivity.py`, `rank_scan_summary.py` | `rank_scan/rank_sensitivity.json` | retained-rank scan of the POD-based estimators (run outside step 04) | supplementary table S24 |

## 4. Data

| Dataset | Local file | Public source | Role |
|---|---|---|---|
| NC | `data/cylinder2d_q1.npy` | ETH Zürich CGL, 2-D unsteady cylinder flow (Gerris, Re = 160); Günther, Gross & Theisel, ACM TOG (2017) | primary: the full reconstruction study |
| RDB | `data/rdb_h5.npy` | PDEBench, 2-D shallow-water radial dam break, doi 10.18419/DARUS-2986 | POD-rank adequacy only (supplementary fig. S3, tables S4–S6) |
| SST | `data/sst_weekly.npy` | NOAA OISST weekly sea-surface temperature, packaged by the Senseiver dataset, doi 10.5281/zenodo.8290040 | POD-rank adequacy only |

You build the sensor masks and the trained runs locally: `features/sensors/incremental_masks.py` draws the masks from their seeds, and `applications/pipelines/03_train_estimators.py` trains the estimators. `scripts/download_data.sh` and `scripts/reproduce_all.sh` list the inputs and the order of the steps.

## 5. Metric conventions

These are the conventions the paper uses; the same definitions are implemented in `luna/` and checked by the unit tests under `tests/`.

- **Band decomposition** — level-4 orthogonal Daubechies-2 wavelet transform with periodisation, giving the bands `A4, W4, W3, W2, W1` from coarse to fine.
- **GER_u** — streamwise-component relative error, `||u - û||₂ / ||u||₂`. The wavelet identity of the paper holds for this quantity, and the analytical benchmark, the matched-pair analysis and all within-configuration correlations use it.
- **GER** — relative error of the full two-component state, `||u_vec - û_vec||₂ / ||u_vec||₂`. The model-level comparisons (estimator table, sensor-count and noise trends, holdout table) use it.
- **E_direct(b)** — relative error of band `b` on the streamwise component.
- **S_full** — number of consecutive bands recovered from the coarsest one whose direct error stays below the tolerance τ. **S_coh** applies the same rule to the POD-dominant band error, where each band is projected onto a band-specific POD basis fitted to the same field the count refers to.
- **Field of the scale diagnostics** — the band error and both counts are evaluated on the streamwise velocity; the two-component state enters the model-level `GER` and the estimators themselves. The POD-dominant basis belongs to the same field as the count.
- **τ** — default tolerance 0.05, with 0.03 and 0.08 as sensitivity values.
- **Model-level and POD-dominant indices** — both are evaluated per reconstruction before averaging; `P_k` is the pass probability `Pr(S_full ≥ k)`.
- **Gappy POD rank** — selected on validation data, capped by the number of scalar observations, `r ≤ m_obs = 2M`.
- **Ridge / MLP** — fixed rank-128 representation, which is underdetermined; the rank rule differs from that of Gappy POD and both are stated in the paper.
- **Splits** — random 70/10/20 for the primary study, contiguous temporal holdout (three placements, 300 held-out snapshots each) for the sensitivity.
- **Noise** — independent Gaussian noise on every scalar observation, with the standard deviation set to `σ` in free-stream-velocity units (`U_∞ = 1`), applied identically to both velocity components; `σ ∈ {0, 0.001, 0.01, 0.1}` spans clean data to a ten-per-cent perturbation of the free-stream speed.

## 6. Generated inputs, and how to rebuild them

The raw arrays, the derived arrays, the sensor masks, the trained models and the artifacts are built locally into git-ignored directories, and the manuscript lives in its own repository. The tracked tree therefore stays code while every reported number remains rebuildable. To rebuild from scratch:

```bash
bash scripts/download_data.sh          # fetch the three public datasets
conda env create -f environment/environment.yml -n luna
bash scripts/reproduce_all.sh          # steps 01-05, writes artifacts/
```

Several producers accept `--verify`, which compares the freshly computed values with the reference values reported in the paper; the reference files appear in the artifacts tree once you have completed a full run.

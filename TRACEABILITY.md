# TRACEABILITY — paper results ↔ source code

> Status: **P1 inventory (2026-09-15)**, produced for the JFM submission refactor.
> This file is both the *input* to the refactor and the *acceptance object* of it:
> every row marked ⚠️/❌ must become ✅ before the repository is published.
>
> Legend — producer location:
> - ✅ producer tracked in this repository
> - ⚠️ producer exists but is **not** tracked here (private paper repo or unshipped application)
> - ❌ producer missing; the intermediate file itself is present under `artifacts/`

---

## 0. How to read this document

The paper is produced by three stages:

```
(1) raw data                data/*.npy                ← public datasets
(2) reconstruction runs     artifacts/**              ← trained estimators + POD bases  (not shipped)
(3) statistics              artifacts/derived/**      ← the numbers that enter the paper
(4) figures & tables        figures/*.pdf, tables/*.tex  ← the paper objects
```

Stage (3) is the contract between code and paper: **a paper object is only
reproducible if its stage-(3) file is produced by a script in this repository.**

The entry points of the repository follow the same order, one module per stage:

| Step | Entry point | Writes |
|---|---|---|
| 01 | `applications/pipelines/01_prepare_data.py` | verifies the raw arrays under `data/` (no output) |
| 02 | `applications/pipelines/02_build_pod_bases.py` | `artifacts/pod_bases/<dataset>/pod_base_bundle.npz` |
| 03 | `applications/pipelines/03_train_estimators.py` | the estimator runs under `artifacts/` |
| 04 | `applications/pipelines/04_compute_statistics.py` | `artifacts/statistics/` |
| 05 | `applications/pipelines/05_make_figures.py` | `artifacts/figures/` |

---

## 1. Stage 1 — raw data

| Dataset | File (local) | Public source | Role in the paper |
|---|---|---|---|
| NC | `data/cylinder2d_q1.npy` | ETH Zürich CGL, *2D unsteady cylinder flow* (Gerris, Re = 160); Günther, Gross & Theisel (2017) | **primary**: full reconstruction study (Figs 2–11, Tables 1–22) |
| RDB | `data/rdb_h5.npy` | PDEBench, 2D shallow-water radial dam break, DOI 10.18419/DARUS-2986 | rank-adequacy check only (Fig S1a, Table S2) |
| SST | `data/sst_weekly.npy` | NOAA OISST weekly SST, packaged by the Senseiver dataset, DOI 10.5281/zenodo.8290040 | rank-adequacy check only (Fig S1b, Table S3) |

Sensor sequences: `masks*/` (NC incremental-random + 5 sensor families for Fig S7/S8).

---

## 2. Stage 2 — reconstruction artifacts (regenerated; **not shipped**)

| Artifact | Producer | Status |
|---|---|---|
| `artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz` | `applications/pipelines/02_build_pod_bases.py` (`luna.pod.decomposition`, rank 128) | ✅ |
| `artifacts/pod_bases/{rdb_h5,sst_weekly}/…` | idem (rank 128 and 1024) | ✅ |
| `artifacts/pod_model_sweep_nc/mlp_n*/seed*/tests/s*/test_raw.npz` | `applications/pipelines/03_train_estimators.py` | ✅ |
| `artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n*_seed*_custom/…` | `applications/pipelines/03_train_estimators.py` | ✅ |
| `artifacts/ridge_closed_form_sweep_nc/ridge_n*/seed000/…` | `applications/pipelines/03_train_estimators.py` | ✅ |

---

## 3. Stage 3 — statistics layer

Every statistic of the submission lives in `artifacts/statistics/` and is written by one
module in `applications/statistics/`. The whole layer is rebuilt with one command:
`python applications/pipelines/04_compute_statistics.py`.

### 3.1 producer inventory

| Statistics file | Producer | Status | Consumed by |
|---|---|---|---|
| `analytical_benchmark.{json,csv}` | `applications/statistics/analytical_benchmark.py` | ✅ | Fig 2, Table 1, Table S7 |
| `band_error_decomposition.json` | `applications/statistics/band_error_decomposition.py` | ✅ | Fig 3, Fig 6, Fig 7; Tables 2, S15 |
| `band_error_records.json` | `applications/statistics/band_error_records.py` | ✅ | Figs 8, 9, S2a, S4; Tables S15–S17; §5.4, §7.2 |
| `sensor_noise_phase.json` | `applications/statistics/sensor_noise_phase.py` | ✅ | Fig 6b,c, Fig 7, Fig S2b; Tables 4, 6 |
| `truncation_reference_audit.{json,csv}` | `applications/statistics/truncation_reference_audit.py` | ✅ | Fig S1, Fig S4; Tables 3, S2–S4, S10 |
| `mode_scale_energy.json` | `applications/statistics/mode_scale_energy.py` | ✅ | Fig S5 |
| `modal_coefficient_error.json` | `applications/statistics/modal_coefficient_error.py` | ✅ | Fig 10 |
| `level_sensitivity.json` | `applications/statistics/level_sensitivity.py` | ✅ | Fig S3b |
| `noise_propagation.json` | `applications/statistics/noise_propagation.py` | ✅ | Fig S3a |
| `band_denominator_check.{json,csv}` | `applications/statistics/band_denominator_check.py` | ✅ | Table S20 |
| `temporal_dependence.json` | `applications/statistics/temporal_dependence.py` | ✅ | §7.4 block bootstrap |
| `paired_model_comparison.{json,csv}` | `applications/statistics/paired_model_comparison.py` | ✅ | Fig S8, §7.4 |
| `sensor_family/…`, `sensor_family_summary.json` | `applications/statistics/sensor_family_summary.py` | ✅ | Fig S7, Table S19, truth layer |
| `gappy_pod_baseline.json` | `applications/statistics/gappy_pod_baseline.py` | ✅ | Fig 8 |
| `equal_ger_pairs.json`, `equal_ger_pairs_strict.json` | `applications/statistics/equal_ger_pairs.py` | ✅ | Fig 4, Table S18 |
| `fourier_band_baseline.{json,csv}` | `applications/statistics/fourier_band_baseline.py` | ✅ | Fig 5, Table S14 |
| `transform_symmetry_check.{json,csv}` | `applications/statistics/transform_symmetry_check.py` | ✅ | Fig 5, Tables S14, S21 |
| `threshold_sensitivity.json` | `applications/statistics/threshold_sensitivity.py` | ✅ | Fig S6, Table S12 |
| `wavelet_sensitivity.{json,csv}` | `applications/statistics/wavelet_sensitivity.py` | ✅ | Fig 11, Table 5, Table S13 |
| `ger_band_correlation.json` | `applications/statistics/ger_band_correlation.py` | ✅ | §4.4, Table S17 |
| `scoh_vs_sfull.json` | `applications/statistics/scoh_vs_sfull.py` | ✅ | §6, Table S18 |
| `seed_stability.json` | `applications/statistics/seed_stability.py` | ✅ | §7.2 |
| `tau_pairwise_checks.json` | `applications/statistics/tau_pairwise_checks.py` | ✅ | App. B, §7.5 |
| `wavelet_family_sensitivity.json` | `applications/statistics/wavelet_family_sensitivity.py` | ✅ | Table S13 |
| `low_ger_analysis.json` | `applications/statistics/low_ger_analysis.py` | ✅ | Table S16 |
| `gappy_band_errors.json` | `applications/statistics/gappy_band_errors.py` | ✅ | Table S15 |
| `per_mode_nrmse.json` | `applications/statistics/per_mode_nrmse.py` | ✅ | Tables S8, S11 |
| `excess_error_recompute.json` | `applications/statistics/excess_error_recompute.py` | ✅ | Tables S8, S15 |
| `within_config_physics_bootstrap.json` | `applications/statistics/within_config_physics_bootstrap.py` | ✅ | Table S17 |
| `seed_audit.json` | `applications/statistics/seed_audit.py` | ✅ | §7.2 |
| `boundary_sensitivity.json` | `applications/statistics/boundary_sensitivity.py` | ✅ | §7.5 |
| `band_pod_energy_sensitivity.json` | `applications/statistics/band_pod_energy_sensitivity.py` | ✅ | §3.3 |

### 3.2 statistics without a producer here

Every value-layer entry now names either an artifact of this repository (108 of 143),
raw data or a definition (34), or a module and a data directory (1). What is left
outside is small and deliberate:

| Statistics file | Status | Consumed by |
|---|---|---|
| `equal_ger_68_pairs_full.json` | retired: the disjoint 68-pair set is superseded by the 1 249 within-configuration pairs | — |
| `ger_fits_v5.json` | retired with the scaling-exponent paragraph (five sensor counts do not support a power law) | — |
| `rossby_wavelet_sensitivity.*` | the experiment is not in the paper | out of scope |

`python tools/check_traceability.py` prints the same accounting from the value layer
itself, so the claim can be re-checked after any edit.

### 3.3 supplementary analyses

All four supplementary analyses now live in `applications/statistics/` and write to
`artifacts/statistics/`. The submitted versions of the paired statistics and of the
convolutional family validation were computed on normalised fields; see plan §5.3.

| Statistics file | Producer | Status | Consumed by |
|---|---|---|---|
| `statistics/paired_model_comparison.json` | `applications/statistics/paired_model_comparison.py` | ✅ | Fig S8, §7.4 |
| `statistics/sensor_family/sensor_count_effect.csv` | `applications/statistics/sensor_family_summary.py` | ✅ | Fig S7, Table S19 |
| `statistics/sensor_family_summary.json` | idem | ✅ | §7.3, Table S19, truth layer (`vcnn_validation.*`, macros currently unreferenced) |
| `statistics/temporal_dependence.json` | `applications/statistics/temporal_dependence.py` | ✅ | §7.4 block bootstrap |
| `statistics/band_denominator_check.json` | `applications/statistics/band_denominator_check.py` | ✅ | Table S20 |
| `statistics/noise_propagation.json` | `applications/statistics/noise_propagation.py` | ✅ | Fig S3a |
| `statistics/level_sensitivity.json` | `applications/statistics/level_sensitivity.py` | ✅ | Fig S3b |
| `statistics/mode_scale_energy.json` | `applications/statistics/mode_scale_energy.py` | ✅ | Fig S5 |
| `derived/supplementary/predictions/{family}/…` | `applications/pipelines/…` (family runs) | ⚠️ runs not shipped | input of the two producers above |

---

## 3.4 Metric conventions

Every number in the paper is produced by one of these definitions. They are listed
because two of them were **not** consistent across the submitted statistics files,
and the repository now uses the unified column only.

| Quantity | Component used | Definition | Used by |
|---|---|---|---|
| global error ratio (GER) | **both components** | ‖u − û‖₂ / ‖u‖₂ over the stacked state | all reported GER (Methods, §3.1, §5, tables) |
| truncation global error | **both components** | same quantity for the rank-r POD truncation of the snapshot | §3.1 POD truncation reference; ratio in §5.1 |
| band errors (total / truncation / prediction) | streamwise u | ‖W_b(·) − W_b(·)‖₂ / ‖W_b(u)‖₂, shared denominator | §3.3, Fig 3, Table 2, Table S5, Table S15 |
| `S_full`, `S_coh` | streamwise u | number of contiguous bands with band error ≤ τ | all scale-recoverability results |
| per-band `E_direct`, `E_coh` | streamwise u | band-POD protocol | Fig S3c, Table S6 |

Unified in this refactor:

- `three_layer_fixed.json` reported GER on the **streamwise component** only; the
  decomposition records now report it on both components, matching every other
  statistics file and the Methods text. The band-wise terms are unchanged and were
  verified to reproduce the frozen values exactly.
- The former `three_layer_fixed.json` also mixed in **AdamW-trained** Ridge records
  (a deprecated estimator); the records now use the closed-form Ridge used
  everywhere else in the paper.

## 4. Figures

All 26 figures are drawn by `applications/figures/` (one script per figure or
figure group) from `artifacts/statistics/`; `make_all_figures.py` redraws the
whole set (the pipeline entry point of this stage is
`applications/pipelines/05_make_figures.py`), `compare_with_manuscript.py` checks
it against the manuscript copies
and `publish_figures.py` updates those copies. The scripts used to live in the
private manuscript tree only, which is why this row of the plan was open.


| Paper figure | File | Data source(s) | Script | Status |
|---|---|---|---|---|
| Fig 1 | `fig01_method_framework.pdf` | — (schematic; `.svg` source) | `applications/figures/fig01_method_framework.py` | ⚠️ |
| Fig 2 | `fig02_analytical_benchmark.pdf` | `analytical_benchmark.json` | `applications/figures/fig02_analytical_benchmark.py` | ⚠️ |
| Fig 3(a) | `fig03_counterexample_a.pdf` | `three_layer_fixed.json` + MLP npz | `applications/figures/fig03_counterexample.py` | ⚠️❌ |
| Fig 3(b) | `fig03_counterexample_b.pdf` | idem + `data/cylinder2d_q1.npy` | idem | ⚠️❌ |
| Fig 4 | `fig04_equal_ger.pdf` | `equal_ger_pairs.json` | `applications/figures/fig04_equal_ger.py` | ⚠️ |
| Fig 5 | `fig05_wavelet_vs_fourier.pdf` | `fourier_band_baseline.*`, `transform_symmetry_check.json` | `applications/figures/fig05_wavelet_vs_fourier.py` | ⚠️ |
| Fig 6(a,b,c) | `fig06_ger_vs_M.pdf`, `fig06_sfull_vs_M.pdf`, `fig06_noise_sfull.pdf` | `three_layer_fixed.json`, `s26_pass_probability.json` | `applications/figures/fig06_results.py` | ⚠️❌ |
| Fig 7 | `fig07_phase_diagram.pdf` | `s26_pass_probability.json` | idem | ⚠️ |
| Fig 8 | `fig08_recoverability_chain.pdf` | `band_error_records.json`, `band_error_records.json` (ridge subset), `gappy_pod_baseline.json` | `applications/figures/fig08_recoverability_chain.py` | ⚠️❌ |
| Fig 9 | `fig09_cross_model_bands.pdf` | `band_error_records.json`, `band_error_records.json` (ridge subset) | `applications/figures/fig09_cross_model_bands.py` | ⚠️ |
| Fig 10 | `fig10_energy_vs_nrmse.pdf` | POD basis npz + MLP/VCNN npz + raw data + masks | `applications/figures/fig10_energy_vs_nrmse.py` | ⚠️ |
| Fig 11 | `fig11_wavelet_sensitivity.pdf` | `wavelet_sensitivity.json` | `applications/figures/fig11_wavelet_sensitivity.py` | ⚠️ |
| Fig S1(a,b) | `figS01_oracle_rdb.pdf`, `figS01_oracle_sst.pdf` | `truncation_reference_audit.json` | `applications/figures/figS01_oracle.py` | ⚠️ |
| Fig S2(a,b) | `figS02_ridge_phase.pdf`, `figS02_vcnn_phase.pdf` | `band_error_records.json` (ridge subset), `s26_pass_probability.json` | `applications/figures/figS02_phase.py` | ⚠️ |
| Fig S3(a) | `figS03_noise_propagation.pdf` | legacy `noise_propagation.json` | `applications/figures/figS03_diagnostics.py` | ⚠️❌ |
| Fig S3(b) | `figS03_level_sensitivity.pdf` | legacy `level_sensitivity.json` | idem | ⚠️❌ |
| Fig S3(c) | `figS03_coherent_only_sample.pdf` | VCNN npz (M=20, σ=0, first failing snapshot) | `applications/figures/figS03c_coherent_only.py` | ⚠️ |
| Fig S4 | `figS04_three_layer.pdf` | `band_error_records.json` | `applications/figures/figS04_three_layer.py` | ⚠️ |
| Fig S5 | `figS05_mode_scale_energy.pdf` | `mode_scale_energy.json` | `applications/figures/figS05_mode_scale_energy.py` | ✅ |
| Fig S6 | `figS06_tau_sensitivity.pdf` | `threshold_sensitivity.json` | `applications/figures/figS06_tau_sensitivity.py` | ⚠️ |
| Fig S7 | `figS07_sensor_family_ger.pdf` | `sensor_family/sensor_count_effect.csv` | `applications/figures/figS07_sensor_family_ger.py` | ⚠️ |
| Fig S8 | `figS08_sensor_family_paired.pdf` | `paired_model_comparison.json` | `applications/figures/figS08_sensor_family_paired.py` | ⚠️ |

Note: the figure scripts live in `applications/figures/` and read only `artifacts/statistics/`.
An earlier application layer (a separate main-paper tree, a supplementary tree and a
retired expansion tree) has been removed from the public tree.

---

## 5. Tables

26 tables, all generated by `thesis_work/.../tools/generate_paper_tables.py` from
`data/paper_facts.yaml` (value layer). Below: table → value-layer section → underlying statistics.

| Table | Value-layer section | Underlying statistics | Status |
|---|---|---|---|
| Table 1 `tab_analytical_benchmark` | `p0_1` | `analytical_benchmark.json` | ⚠️✅ |
| Table S7 `tab_analytical_params` | `p0_1` | idem | ⚠️✅ |
| Table 5 `tab_wavelet_family_sensitivity` | `p0_2` | `wavelet_sensitivity.json` | ⚠️✅ |
| Table 3 `tab_oracle_results` | `oracle` | `truncation_reference_audit.json` (band means) + `ua` (GER/Sfull) | ⚠️✅❌ |
| Table S11 `tab_recovery_rate` | `mechanism.recovery_rates` | `s10_tables_12_13.json` | ⚠️✅ |
| Table S8 `tab_decile` | `mechanism.decile_table` | `s10_tables_12_13.json` ← `s21_nrmse_full.json` | ⚠️✅❌ |
| Table S10 `tab_nrmse_spearman` | `mechanism.nrmse_spearman` | `band_error_records.json` (ridge subset), `s21_nrmse_summary.json` | ⚠️✅❌ |
| Table 2, S15 `tab_three_layer`, `tab_delta` | `results.three_layer_table`, `results.delta_excess_band_error` | `band_error_records.json` + `truncation_reference_audit.json` | ⚠️ |
| Table 4, 6 `tab_phase_mlp`, `tab_phase_vcnn` | `results.phase_diagram_*` | `s26_pass_probability.json` | ⚠️✅ |
| Table S16 `tab_low_ger_stats` | `validation.low_ger_analysis` | `s08b_low_ger_final.json` | ⚠️✅ |
| Table S17 `tab_correlations`, `tab_within_config_corr` | `validation.ger_band_correlation`, `validation.within_config_physics_correlation` | `ua` (❌), `within_config_physics_bootstrap.json` (❌) | ⚠️❌ |
| Table S6 `tab_co_example` | `results.co_example_table` | VCNN npz (M=15, σ=0.01, snapshot 89) | ⚠️ |
| Table S5 `tab_type_a_detail` | `validation.type_a_visualization_pair` | `band_error_records.json` | ⚠️ |
| Table S12 `tab_tau_sensitivity` | `robustness.tau_sensitivity_table` | `threshold_sensitivity.json` | ⚠️✅ |
| Table S13 `tab_wavelet_sensitivity` | `robustness.wavelet_sensitivity_table` | `ua` (early db2/sym2/haar run) | ⚠️❌ |
| Table S21 `tab_controlled_scale_wavelet`/`_fourier` | `robustness.controlled_scale_*` | `transform_symmetry_check.json` | ⚠️✅ |
| Table S1 `tab_app_datasets` | `datasets.*` | `oracle_audit_refined.json` (❌) | ⚠️❌ |
| Table S2/S3/S4 `tab_oracle_audit_nc/rdb/sst` | `appendix.oracle_audit_*` | `truncation_reference_audit.json` | ⚠️✅ |
| Table S19 `tab_supp_multi_mask` | `supplementary.multi_mask_table` | `sensor_family/sensor_count_effect.csv` | ⚠️ |
| Table S20 `band_denominator_audit` | `supplementary.denominator_audit_table` | `band_denominator_check.json` | ⚠️ |
| inline `tab_corr…` | `mechanism.*` | `npz`-derived (unspecified) | ⚠️❌ |

---

## 6. Known gaps

### 6.1 Corrections made during this refactor

Every number below moved in the paper as well; they are corrections, not new free
parameters.

- **Convolutional outputs in physical units.** The family runs stored normalised
  fields, so the paired MLP–VCNN statistics and the family validation were computed
  on the wrong scale. Global error −1.3836 → −0.00099 (interval now contains zero),
  W1 −3.2127 → −0.05707, S_full +2.9832 → +0.43772, S_coh +0.8469 → −0.11539.
- **One definition of the scale count.** The retired ridge file counted recovered
  bands differently from the rest of the paper. The appendix ridge panel now uses
  the common definition: `M=30, σ=0` 2.80 → 2.63, `M=50, σ=0` 4.02 → 3.89.
- **Closed-form ridge throughout.** The threshold sweep used AdamW-trained ridge
  records (global error 0.0300 → 0.0203 for `M=20, σ=0`); those records are retired.
- **Equal-GER pairing unchanged.** Same 1 249 pairs, statistics equal to 1e-8.
- **Five results gained a producer** (`ger_band_correlation`, `scoh_vs_sfull`,
  `seed_stability`, `wavelet_family_sensitivity`, `tau_pairwise_checks`). Their
  values moved where the earlier hand-run used an undocumented definition; each
  entry in the value layer records the definition now used.

### 6.2 Not yet reproducible

| Gap | Affected paper objects | State |
|---|---|---|
| The value layer `paper_facts.yaml` is maintained by hand; its `source:` fields are checked mechanically but its numbers are not generated | all 26 tables and all numeric macros | open by design: `tools/check_traceability.py` and the paper's consistency gate verify the links, the values stay hand-written |
| The table generator and the value layer live in the paper repository, not here | all 26 tables | open by design: the statistics layer is shipped, the manuscript is not |
| Three results were retired rather than reproduced: the disjoint 68-pair equal-GER set, the power-law scaling exponents, and the Rossby-wavelet experiment | none in the submitted figure and table set | closed as retired |

**Acceptance for the statistics layer**: every producer in `applications/statistics/`
recomputes its artifact from `artifacts/`, the value layer points at those artifacts,
and `tools/check_traceability.py` reports no source without a producer.

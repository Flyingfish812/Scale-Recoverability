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
| `blocked_holdout_audit.json` | `applications/statistics/blocked_holdout_audit.py` | ✅ | Table S22, contiguous-holdout audit |
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
| `rank_scan/rank_sensitivity.json` | `applications/statistics/rank_sensitivity.py` | ✅ standalone scan, run outside the `04_compute_statistics.py` orchestrator | retained-rank sensitivity behind §2.2 |
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

## 3.5 The proofreading loop

A number is corrected in one place, `paper_facts.yaml`, and everything downstream
follows from it. Measured on this machine (2026-09-16):

| Step | Command | Time |
|---|---|---|
| macros | `python tools/generate_paper_numbers.py` | 2.0 s |
| tables | `python tools/generate_paper_tables.py` | 2.0 s |
| paper gate | `python tools/check_paper_consistency.py` | 3.9 s |
| manuscript | `tectonic main.tex` | 9.7 s |
| **whole loop** | `bash refresh.sh` in the manuscript repository | **18 s** |

If a *statistic* has to be recomputed rather than re-read, one producer can be run
on its own, without the twenty minutes of the full layer:

| Step | Command | Time |
|---|---|---|
| one aggregate statistic | `python applications/pipelines/04_compute_statistics.py --only tau_pairwise_checks` | 1.9 s |
| one statistic over the runs | `... --only boundary_sensitivity` | 11.8 s |
| all statistics | `python applications/pipelines/04_compute_statistics.py --jobs 3 --workers 10` | ~20 min |
| one figure | `python -m applications.figures.figS04_three_layer` | 3.5 s |
| all figures | `python -m applications.figures.make_all_figures` | 33 s |

Training is the only slow step and is never needed for a proofreading round: the
trained runs and the POD bases stay on disk, and `02_build_pod_bases.py --check`
plus `03_train_estimators.py --check` confirm they are the ones the statistics
layer expects.

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

### 6.3 Second external-review round (2026-09-16)

The reviewer reported "logical breaks where updated numbers had not propagated".
Each item was re-checked against the produced artifacts before editing.

| # | Reported issue | Finding | Fix |
|---|---|---|---|
| 1 | Sec. 5.2 "Ridge reaches `Sfull = 4.02` at `M = 50` but `Scoh = 1.73`" contradicts Sec. 6.2 | stale: the closed-form ridge values are `S_full = 3.887`, `S_coh = 4.427` (`band_error_records.json`, `scoh_vs_sfull.json`), i.e. the coherent count is *not* lower | value layer `results.ridge_m50.sfull/scoh` → 3.89/4.43, sentence rewritten |
| 2 | Appendix D.4 MLP−VCNN differences `+0.23 / +0.30` were stale and hard-coded | `paired_model_comparison.json`: `S_full +0.4377`, `S_coh −0.1154`, `GER −0.00099` | value layer `results.mlp_vs_vcnn.*` now sourced from that artifact; text uses `\FMlpVcnnSfullDiff` / `\FMlpVcnnScohDiff` |
| 3 | scalar `GER_u` mixed with the two-component GER | Eqs. (6)–(9) hold for the scalar streamwise field only | the scalar relation is now `\GER_u`, with an explicit sentence that Secs. 4–7 report the full two-component state; Sec. 4.2 reworded |
| 4 | the band operator in Eq. (9)/Fig. S5 did not name its component | `mode_scale_energy.py` filters the streamwise component of each POD mode | Eq. and Fig. S5(b) axis label now use `\phi_j^{(u)}`, defined in Sec. 6.1 |
| 5 | Sec. 7.2 "across the three main training runs" (wrong: ridge is deterministic) | the three widths also came from an unscripted hand analysis and the `source:` pointed at the wrong artifact | `seed_stability.json` now supplies per-estimator configuration-mean widths 0.06/0.10/0.09 and the range 0.01–0.28 over the 50 non-degenerate configurations |
| 6 | Discussion "significant in 20 configurations and not weaker in the rest" | Table S18: `rho_G >= rho_S` in 20/20, of which 13 have a 95% CI excluding zero | reworded to "stronger in all 20 configurations, significant in 13" |

The same round added the minimal audit requested by the reviewer:

```
python3 -u tools/check_key_results.py      # run from the repository root
```

It cross-checks artifact → value layer → paper macro for six groups: G1 Table 2
(ridge `M = 50` counts), G2 Fig. 8 (MLP error and rank-128 truncation floor),
G3/G6 Fig. 10 + Table S10 + Sec. 7.2 (`S_full` bootstrap CI widths), G4 Fig. S5 +
Table S8 (modal energy tail), G5 Sec. 6.2 (paired differences, `S_coh`/`S_full`
counts).  Current state `PASS=14 FAIL=0 SKIP=0`.  Display-rounded entries use a
half-unit tolerance, subsample conventions (50/150/300 snapshots) use a 5%
relative tolerance, and an unlocatable path reports SKIP instead of passing
silently.

### 6.4 Third external-review round (2026-09-16)

The reviewer re-read all 26 pages and reported eight groups: four to fix before
freezing the numbers, four for completeness.  Findings and actions:

| # | Reported issue | Finding | Fix |
|---|---|---|---|
| 1 | `Sec. ??` on p. 4; Sec. 3.1 says the full-state GER is reported in Secs. 5–7 | the cross-reference pointed at a label (`sec:problem`) that does not exist; the method section defines the two-component state in Sec. 3.1 (`sec:setting`) | reference repaired; range corrected to Secs. 4–7 in Sec. 3.1, matching Sec. 3.3 |
| 2 | scalar vs two-component GER still mixed in three places | Eq. (9)'s implication was written for the two-component symbol and Sec. 4.2/Discussion applied the scalar identity to the full-state metric | the implication now reads `\GER_u`, Sec. 4.2 and the Discussion state that Eq. (9) establishes the masking effect for the streamwise scalar and that the NC results show the analogous effect for the two-component GER |
| 3 | `0.44 / -0.12 / -1e-3` presented as `(M, sigma) = (20,0)` | those are the aggregate values over the 18,000 matched observations of the primary nested sensor sequence; the `(20,0)` difference is `+0.23` (MLP mean `S_full` 4.513 vs VCNN 4.282) | Appendix D.4 relabelled to the 18,000-observation aggregate; Sec. 5.2 uses the new `(20,0)` value from `band_error_records.json`; the value layer now separates `sfull_diff_m20_sigma0` from `supplementary.paired_mlp_vcnn.*` |
| 4 | deprecated "coherent" wording reappeared | `S_coh` is the POD-dominant count | wording replaced by "POD-dominant count / content" |
| 5 | Fig. 4 / pooled test pair scope unclear | the artifact's pooled statistics and bootstrap use the 1,249 within-configuration pairs; the 13 cross-model pairs are a separate set | text now says the figure and pooled test use the 1,249 pairs and that the 13 cross-model pairs are excluded |
| 6 | "50 non-degenerate configurations" undefined | 60 configurations exist; 10 have zero-width intervals | both Sec. 7.2 and Appendix D.2 now say "configurations with non-zero bootstrap width" and quote the count |
| 7 | Fig. 5(a) used 100 of 300 held-out snapshots without justification | it was only the producer's CLI default (`--n-samples 100`), not a documented subset | default raised to 300 and the artifact recomputed: `rho(S_full, S_FFT)` 0.79 → 0.80, spectral-loss/low-RMSE/high-RMSE 0.79/0.80/0.60 → 0.78/0.78/0.55, `GER`–spectral loss 0.998 → 0.996 |
| 8 | Table S1 `Mask type` column contradicts the "RDB/SST are rank-adequacy audits only" narrative | sensor masks are irrelevant for those entries | column removed from the generator; the caption now states that masks enter only the NC reconstruction study |

Verification after the round: paper gate `PASS=118 FAIL=0`, key-result audit
`PASS=15 FAIL=0 SKIP=0`, `pytest tests/unit` 63 passed, repository hygiene 0
failures, traceability clean, tectonic 26 pages with no unresolved reference.

### 6.5 Fourth external-review round (v5-7 → v5-8, 2026-09-16)

Seven groups, all closed; only one touches a number.

| # | Reported issue | Finding | Fix |
|---|---|---|---|
| 1 | Abstract describes the global-band relation without the square | the identity is `GER_u^2 = sum_b omega_b E_direct(b)^2`; the Introduction already stated it correctly | Abstract now reads "the squared GER is an energy-weighted combination of the squared errors at individual scales" |
| 2 | Sec. 3.3 opens with "Global error (GER) is ..." right after the scalar/two-component split | the sentence defined the scalar quantity under the generic name | now "For the scalar field, we define the corresponding relative `l_2` error as", followed by the `GER_u` equation |
| 3 | Statistical populations in Sec. 7.2 / Appendix D.2 | the 42,000 records span all learned-model seeds, whereas the CI-width summary uses one representative seed per learned model plus deterministic Ridge (60 configurations) | both places now state their population explicitly, a new macro gives the 60 configurations, and the "inference uses" typo is fixed; Sec. 3.2 also distinguishes pooled / representative-seed / cross-seed results |
| 4 | Fig. 5 population and the 20%/80% cross-domain accuracy undefined | the correlations belong to the same 300-snapshot VCNN case; cross-domain accuracy averages 10 fields over the 5 truncation levels `k=1..5` (50 cases, untruncated case excluded) | text now says "For the same 300-snapshot VCNN case" and defines the cross-domain metric with its denominator (new `cross_transform_accuracy` entry in the value layer) |
| 5 | Table 4 regime boundaries were descriptive ("about 1-3") | the table contains 3.36 and 3.41, so the interval must be half-open | regimes are now high `S_full >= 4`, intermediate `1 <= S_full < 4`, low `S_full < 1`, in the caption and in Sec. 5.1 |
| 6 | `[18,45]` cited as the source for "reference-free scale assessment" | one entry is an online basis-updating criterion, the other a data-assimilation package; neither is a reference-free diagnostic | citations removed from that sentence (the remaining bibliography entries are unaffected) |
| 7 | `Secs. 4- 6` spacing; Table S21 float placement | the en dash was split across a source line | source joined; float placement intentionally left to the JFM template migration |

### 6.6 Fifth external-review round (v5-8 → v5-9, 2026-09-16)

Four content items; the fifth (float placement) is deferred to the JFM
template migration as the reviewer suggested.

| # | Reported issue | Finding | Fix |
|---|---|---|---|
| 1 | the cross-domain accuracy sentence contradicted itself (`k = 1..5` with "the untruncated case is excluded") and mixed a 10-field population with the 100-field tables | the artifact kept the *empty* case out (`expected_recoverable is not None` covers `k = 1..5`, where `k = 5` is the untruncated all-bands case), and `transform_symmetry_check.py` defaulted to `--n-test-fields 10` while the transform-domain tables use 100 fields | the producer now uses the same 100 fields, the artifact was recomputed, and the text states the range and the 500 judgements explicitly. Numbers moved: `S_full` on Fourier-annulus-truncated fields 80% → 96%; `sfull_on_fourier_targets` third entry `1--3` → `2--3` |
| 2 | 8,879 "samples" read as independent snapshots | 8,879 is a count of pooled evaluation records (21.1% of 42,000) | Sec. 6.2 and Appendix D.5 now say "pooled evaluation records (21.1% of the 42,000 records)" and "those records" |
| 3 | `S_full` and mean `S_full` mixed | the regimes and the ridge `M = 50` counts are configuration means, while a single sample gives an integer | regime definitions and the ridge sentence now say mean `S_full` / mean `S_coh` |
| 4 | Sec. 4.3 quoted `rho` without naming the coefficient | the artifact stores `spearman_rho` | the paragraph now states that all coefficients quoted there are Spearman rank correlations |

### 6.7 Sixth external-review round + knock-on sweep (v5-9 → v5-10, 2026-09-16)

Five content items from the reviewer plus a full end-to-end sweep that turned
up four further inconsistencies created or left behind by earlier rounds.

| # | Item | Fix |
|---|---|---|
| 1 | Sec. 4.3 still concluded "each index is specific to its own transform", which the updated 96 % contradicts | the paragraph now reports the asymmetric transfer (own-domain 100 %; `S_full` correct for 96 % of Fourier-annulus truncations, `SFFT` for only 20 % of wavelet-truncated cases) and defines accuracy as the fraction of the 500 field–truncation pairs whose index equals `k`. Appendix E, the Table S14 caption and the Fig. 5(b) caption were aligned, and the figure script now pools pairs instead of averaging per field (same numbers) |
| 2 | `sample` still used for pooled counts in three places | Sec. 6.2 now reads "at the individual reconstruction-record level"; Table 3 column is `Low-GER records`; Table S11 says "unequal record counts". The paper-wide vocabulary is now: *snapshot* = one held-out field (300), *record* = one model–condition–seed–snapshot evaluation (42,000) |
| 3 | Sec. 3.2 "provide every target field evaluated below" | scoped to "every NC reconstruction target" |
| 4 | Sec. 6.1 carried a dangling "alternative normalization" | removed |
| 5 | Sec. 6.2 "no configuration recovers more bands under the full-band reading" | reworded to "no configuration has mean `S_full` > mean `S_coh`", compatible with the snapshot-dependent ordering of Sec. 3.4 |

Knock-on findings from the sweep:

- **`S_full` means for the Fourier truncation moved** with the 100-field recomputation: `k = 2, 3` are 1.96, 2.86 → **1.97, 2.85** (value layer, macros and Table S14).
- **Sec. 6.1 quoted an unsourced number** (`-0.9843`, "standardized coefficients"): no artifact contains it and the producer computes a single correlation. Replaced by the sourced statement (per-configuration value plus the across-configuration mean and SD).
- **`robustness.amplitude_perturbation` had a wrong `source:`** — the referenced artifact contains no amplitude data and the paper never reports the experiment. The entry is retired and its three unused macros removed.
- **The ridge and gappy W1 excesses were hard-coded** as "more than 0.10 / 0.14"; they are now the sourced values `\FRidgeExcessWOne` = 0.108 and `\FGappyExcessWOne` = 0.148.
- **Residual "coherent" naming** in the POD-component figure (`figS03c_coherent_only.py`, `figS03_coherent_only_sample.pdf`, label `sfig:coherent_only`) renamed to `pod_dominant` throughout; the figure was regenerated and republished.

Verification: paper gate `PASS=118 FAIL=0`, key-result audit `PASS=15 FAIL=0
SKIP=0`, `pytest tests/unit` 63 passed, repository hygiene 0 failures,
traceability clean, 27 figures identical to the manuscript, tectonic 27 pages
with no unresolved reference.

### 6.8 Seventh external-review round (v5-10 → v5-11, 2026-09-16)

| # | Item | Fix |
|---|---|---|
| 1 | the 18,000 MLP-VCNN pairs were still called "observations", a word the paper reserves for sensor measurements ($m_obs$) | all three places (Sec. 7.2, Appendix D.4, Fig. S8 caption) now say "18,000 matched **reconstruction records**" |
| 2 | "three seeds pooled" did not say over how many values the statistics are taken | Fig. 8 caption: "300 held-out snapshots; learned-model bars pool three seeds, i.e. 900 reconstruction records each". Fig. 9 caption: "mean ± one standard deviation over the 900 reconstruction records of that configuration (300 held-out snapshots × three seeds); Ridge is deterministic, so its bar uses the 300 held-out snapshots" — verified against `records.py`/`fig08`/`fig09`, which pool every matching record |
| 3 | Sec. 7.2 quoted "cross-run SD ≤ 0.18" without defining the statistic (the pooled-mean shift is only 0.045) | now "shift the pooled mean by at most ... and leave the SD of the **per-seed means** at ≤ ...", which is what `seed_audit.json` reports (`S_full_sd_across_seeds_3` = 0.175 for the clean case) |
| 4 | Ridge `M = 50` lacked the noise level | now "at $(M,\sigma) = (50, 0)$" |
| 5 | Appendix Section C was titled "VCNN Phase Diagram" although its figure shows Ridge and VCNN | renamed "Additional Model Phase Diagrams"; the section body now cites both the figure and the table. Float ordering is still left to the JFM template |

Two follow-ups from the sweep:

- The figure captions needed the number 900, which the consistency gate correctly
  rejected as a hard-coded literal. It is now the derived entry
  `experiment_config.records_per_learned_config` (300 snapshots × 3 seeds, source
  `definition`) exported as `\FRecordsPerConfig`.
- Sec. 4.3's "20 % of the wavelet-truncated cases" became "20 % of the
  wavelet-domain test cases", since that 20 % is carried by the untruncated
  `k = 5` case.

Verification: paper gate `PASS=118 FAIL=0`, key-result audit `PASS=15 FAIL=0
SKIP=0`, `pytest tests/unit` 63 passed, repository hygiene 0 failures, the
traceability gate resolves every value-layer source (0 unproduced), 27 figures
identical, tectonic 27 pages.

### 6.9 Eighth external-review round (v5-11 → v5-12, 2026-09-16)

Two content items; the float ordering is again left to the JFM migration.

| # | Item | Fix |
|---|---|---|
| 1 | Fig. 5(b) called the counted groups "wavelet-truncated reconstructions", although the cross-domain statistic covers levels `k = 1..5`, and the only correct `S_FFT` wavelet-domain case is the *untruncated* `k = 5` — the 20 % came entirely from it | the figure axis labels and the caption now say "Wavelet-domain test cases" and "Fourier-annulus-domain test cases"; the caption also states the population (100 fields, `k = 1..5`, empty `k = 0` excluded). Sec. 4.3's "Fourier-annulus truncations" became "Fourier-annulus test cases". No numbers were recomputed |
| 2 | Sec. 3.5 pointed at Appendix B for both rank adequacy and error attribution, but Appendix B only holds the rank-adequacy audit | now "Rank adequacy is tested in Appendix B, and error attribution in Sec. 6.2 and Appendix D.7" (`app:three_layer_fig`), i.e. the appendix section holding Fig. S4 / Table S9 |

Verification: paper gate `PASS=118 FAIL=0`, key-result audit `PASS=15 FAIL=0
SKIP=0`, `pytest tests/unit` 63 passed, repository hygiene 0 failures,
traceability 0 unproduced sources, 27 figures identical to the manuscript,
tectonic 27 pages with no unresolved reference.

### 6.10 Ninth external-review round — v5 series frozen (v5-12 → v5-13, 2026-09-16)

One item: Sec. 7.2 said "no band has a near-zero denominator", but the audit in
Appendix F.2 checks the $\\Edirect$ denominators ($\\|\\mathcal{W}_b u\\|_2$ and the
band energy fraction) only, whereas $\\Ecoh$ divides by $\\|P_b\\mathcal{W}_b u\\|_2$.

- Sec. 7.2 now reads "no $\\Edirect(b)$ denominator is near zero".
- Appendix F.2 states that the audited norms are the $\\Edirect$ denominators
  and that the $\\Ecoh$ denominator is the smaller projected norm
  $\\|P_b\\mathcal{W}_bu\\|_2$, whose retained energy is quantified in
  Appendix D.5 (`app:co_energy`).

With this the reviewer closed the review: no further data or logic conflicts
were found, and the v5 series is frozen as the pre-JFM baseline
(paper commit `8ffb2a3` + this round; figures, gates and the value layer are
consistent).  Remaining work is the JFM template/voice migration, which must
re-check appendix heading vs. figure/table ordering after the class change.

### 6.11 Code archive round — JFM v2 series (2026-10-01)

One pass over the published set, so that the repository matches the code that
produced the current manuscript.

| Item | Change |
|---|---|
| New producers/scripts published | `applications/statistics/blocked_holdout_audit.py` (contiguous-holdout table plus the rank-128 representation audit) and `applications/statistics/rank_sensitivity.py` (standalone retained-rank scan, `--out-root`, not part of the orchestrator) |
| Deprecated producers removed | `applications/statistics/gappy_pod_baseline.py` and `gappy_band_errors.py`; Gappy POD now enters only through the canonical per-snapshot records (`band_error_records.py`), and their rows in §3.1 are replaced accordingly |
| Superseded helper removed | `applications/statistics/band_metrics_from_npz.py` — read run `npz` files directly (VCNN fields would have been taken in normalised units) and defaulted to batch directories that no longer exist |
| Figure scripts published | `fig01_known_scale.py`, `fig02_global_vs_scale.py`, `fig03_sensor_noise.py`, `fig05_modal_hierarchy.py`, `fig06_robustness.py`, `graphical_abstract.py` |
| Reported protocol restored as the default | `03_train_estimators.py --gappy-rank-cap` now defaults to `scalars` (r <= m_obs = 2M) with the candidate grid of the paper (multiples of four up to 128), so the documented entry point reproduces the Gappy POD column of the model comparison; `locations` remains available for the cap sensitivity |
| Working-language notes removed | the comments that had slipped into `04_compute_statistics.py` and `fig08_recoverability_chain.py` |
| Layout references refreshed | `.gitignore` no longer names the retired `applications/paper_*` layout; `tests/README.md` points the gappy-rank test at `features.training.pod_sweep` |

Verification of this round: repository hygiene gate 0 failures; clean export of
the archived index (`git write-tree` + `git archive`) runs `pytest tests/unit`
(67 passed) and the self-contained analytical benchmark without any local data,
and contains no reference to a deleted producer.

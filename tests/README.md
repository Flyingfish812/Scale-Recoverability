# Unit tests

The tests pin the mathematical properties the paper relies on, so that a change in
the kernel or in the feature layer cannot silently move a reported number. They
run in about a second and need no data or trained artifacts.

```bash
python -m pytest tests/unit -q
```

| Test file | Property pinned | Implementation under test |
|---|---|---|
| `test_dwt_orthogonality.py` | $\lVert u-\hat u\rVert^2 = \sum_b \lVert W_b(u-\hat u)\rVert^2$ (Parseval plus exact recomposition) | `luna.wavelet.transform.decompose_field_2d`, `recompose_field_2d` |
| `test_ger_band_identity.py` | $\mathrm{GER}^2 = \sum_b \omega_b E_{\text{direct}}(b)^2$ | `luna.wavelet.metrics.rel_l2`, `band_error`, `decompose_field_2d` |
| `test_sfull.py` | contiguity of the scale count: all bands pass, A4 fails, a middle band fails and a later band passes again, the threshold itself counts as a pass, NaN and near-zero denominators | `luna.wavelet.metrics.compute_S_full`, `contiguous_recoverable_index` |
| `test_scoh.py` | the coherent-only count is a projection: idempotence, capture fraction, and no assumption that $S_{\text{coh}} \ge S_{\text{full}}$ | `luna.wavelet.metrics.compute_S_coh` |
| `test_ridge_closed_form.py` | the closed-form estimator matches the normal-equation solution, the bias is not regularised, the fit is deterministic | `features.training.pod_sweep` |
| `test_gappy_rank.py` | the gappy rank is capped at the sensor count and chosen on validation data | `applications.statistics.gappy_pod_baseline` logic, kernel POD projection |
| `test_sample_metrics.py` | the sample metrics are pinned to golden values, including the two-component global error | `features.metrics.sample_metrics` |
| `test_band_error_decomposition.py` | the three-layer split is consistent: the denominators agree and the triangle inequality holds | `features.metrics.band_error` |
| `test_supplementary_statistics.py` | temporal dependence and the block bootstrap reproduce the reported interval | `features.statistics.temporal_dependence` |
| `test_supplementary_denominator.py` | the band denominator audit is Parseval consistent | `features.metrics.band_error.denominator_audit` |
| `test_supplementary_mask_registry.py` | every sensor mask resolves to the documented observation points | `features.sensors.mask_registry` |

"""Band-wise error decomposition and global error ratio.

Checks the identity behind the decomposition,

    u - û = (u - u_ref) + (u_ref - û),

together with the two properties the paper relies on:

1. all three band-wise terms share the denominator ‖W_b(u)‖;
2. the decomposition satisfies the triangle inequality band by band,
   total(b) ≤ truncation(b) + prediction(b).
"""

import numpy as np
import pytest

from luna.core.constants import BANDS_CF, DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET
from luna.wavelet.metrics import (
    band_error,
    band_error_decomposition,
    global_error,
    rel_l2,
)
from luna.wavelet.transform import decompose_field_2d

rng = np.random.default_rng(53)


def _fields(shape=(80, 160)):
    target = rng.standard_normal(shape)
    pred = target + 0.1 * rng.standard_normal(shape)
    reference = target + 0.02 * rng.standard_normal(shape)
    return target, pred, reference


def test_terms_share_the_target_denominator():
    """Every term is normalised by ‖W_b(u)‖."""
    target, pred, reference = _fields()
    result = band_error_decomposition(
        target, pred, reference, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
    )

    target_bands = decompose_field_2d(target, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    pred_bands = decompose_field_2d(pred, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    ref_bands = decompose_field_2d(reference, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)

    for band in BANDS_CF:
        denominator = np.linalg.norm(target_bands[band].ravel())
        assert result[band]["total"] == pytest.approx(
            np.linalg.norm((target_bands[band] - pred_bands[band]).ravel()) / denominator, rel=1e-12
        )
        assert result[band]["truncation"] == pytest.approx(
            np.linalg.norm((target_bands[band] - ref_bands[band]).ravel()) / denominator, rel=1e-12
        )
        assert result[band]["prediction"] == pytest.approx(
            np.linalg.norm((ref_bands[band] - pred_bands[band]).ravel()) / denominator, rel=1e-12
        )


def test_triangle_inequality_holds_band_by_band():
    target, pred, reference = _fields((64, 64))
    result = band_error_decomposition(
        target, pred, reference, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
    )
    for band in BANDS_CF:
        assert result[band]["total"] <= result[band]["truncation"] + result[band]["prediction"] + 1e-12


def test_decomposition_reconstructs_the_exact_error():
    """The two terms reproduce the exact band error when added as vectors."""
    target, pred, reference = _fields((64, 64))
    target_bands = decompose_field_2d(target, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    pred_bands = decompose_field_2d(pred, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    ref_bands = decompose_field_2d(reference, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)

    for band in BANDS_CF:
        residual = (target_bands[band] - ref_bands[band]) + (ref_bands[band] - pred_bands[band])
        assert np.allclose(residual, target_bands[band] - pred_bands[band])


def test_terms_are_non_negative_and_finite():
    target, pred, reference = _fields()
    result = band_error_decomposition(
        target, pred, reference, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
    )
    for band in BANDS_CF:
        for key in ("total", "truncation", "prediction"):
            value = result[band][key]
            assert np.isfinite(value) and value >= 0.0


def test_global_error_covers_the_full_state():
    """The global error ratio uses all components of the state."""
    target = rng.standard_normal((16, 16, 2))
    pred = target.copy()
    pred[:, :, 1] += 0.5 * rng.standard_normal((16, 16))

    both = global_error(target, pred)
    streamwise = rel_l2(pred[:, :, 0], target[:, :, 0])

    assert both == pytest.approx(rel_l2(pred, target), rel=1e-12)
    assert both > streamwise  # the perturbed component contributes


def test_band_error_matches_direct_definition():
    target, pred, _ = _fields((64, 64))
    target_bands = decompose_field_2d(target, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    pred_bands = decompose_field_2d(pred, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    for band in BANDS_CF:
        assert band_error(target_bands[band], pred_bands[band]) == pytest.approx(
            np.linalg.norm((target_bands[band] - pred_bands[band]).ravel())
            / np.linalg.norm(target_bands[band].ravel()),
            rel=1e-12,
        )

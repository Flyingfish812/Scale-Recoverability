"""DWT orthogonality identity tests.

Verifies: ‖u−û‖₂² = Σ_b ‖W_b(u−û)‖₂²
The relative residual is allowed to be < 1e-4 (float32 band components, adjusted to the actual floating-point precision).
"""

import numpy as np
import pytest

from luna.wavelet.transform import decompose_field_2d, recompose_field_2d
from luna.core.constants import BANDS_CF, DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET

rng = np.random.default_rng(20260805)


def _rand_field(shape=(80, 160)):
    u = rng.standard_normal(shape)
    v = rng.standard_normal(shape) * 0.5
    return u, v


@pytest.mark.parametrize("shape", [(64, 64), (80, 160), (96, 128)])
def test_parseval_sum_of_band_energies(shape):
    """Σ_b ‖W_b(u−û)‖² == ‖u−û‖² (Parseval)."""
    u, v = _rand_field(shape)
    d = u - v
    bands = decompose_field_2d(d, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    sum_band_energy = sum(float(np.sum(b.astype(np.float64) ** 2)) for b in bands.values())
    total_energy = float(np.sum(d.astype(np.float64) ** 2))
    rel_residual = abs(sum_band_energy - total_energy) / (total_energy + 1e-12)
    assert rel_residual < 1e-4, f"Parseval residual {rel_residual:.2e}"


def test_band_sum_recomposes_field():
    """Σ_b W_b(u) ≈ u (recomposition)."""
    u, _ = _rand_field()
    bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    recon = recompose_field_2d(bands, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    err = float(np.linalg.norm(recon.ravel() - u.ravel()) / (np.linalg.norm(u.ravel()) + 1e-12))
    assert err < 1e-3


def test_decompose_returns_all_bands():
    """decompose_field_2d returns all 5 bands with correct shapes."""
    u, _ = _rand_field((80, 160))
    bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    assert set(bands.keys()) == set(BANDS_CF)
    for b in BANDS_CF:
        assert bands[b].shape == (80, 160)

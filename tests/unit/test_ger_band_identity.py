"""GER band-identity tests.

Verifies: GER² = Σ_b ω_b E_direct(b)²
    GER         = ‖u−û‖₂ / ‖u‖₂
    ω_b         = ‖W_b u‖₂² / ‖u‖₂²   (band energy fraction)
    E_direct(b) = ‖W_b(u−û)‖₂ / ‖W_b u‖₂
"""

import numpy as np
import pytest

from luna.wavelet.transform import decompose_field_2d
from luna.wavelet.metrics import rel_l2, band_error
from luna.core.constants import BANDS_CF, DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET

rng = np.random.default_rng(7)


@pytest.mark.parametrize("shape", [(64, 64), (80, 160)])
def test_ger_band_identity(shape):
    u = rng.standard_normal(shape)
    v = rng.standard_normal(shape) * 0.3
    ger = rel_l2(v, u)  # ‖u−v‖/‖u‖

    u_bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    d_bands = decompose_field_2d(u - v, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)

    lhs = ger ** 2
    rhs = 0.0
    total_energy = float(np.sum(u.astype(np.float64) ** 2))
    for b in BANDS_CF:
        wb = float(np.sum(u_bands[b].astype(np.float64) ** 2))
        omega = wb / (total_energy + 1e-12)
        ed = band_error(u_bands[b], u_bands[b] - d_bands[b])  # ‖W_b(u−v)‖/‖W_b u‖
        # Note: d_bands[b] = W_b(u−v), hence W_b(u) − W_b(v) = W_b(u) − d_bands[b]
        rhs += omega * ed ** 2

    rel_diff = abs(lhs - rhs) / (lhs + 1e-12)
    assert rel_diff < 1e-4, f"GER identity residual {rel_diff:.2e}"


def test_ger_is_energy_weighted_average():
    """GER is the energy-weighted band error: on smooth fields the coarse A4 dominates."""
    # Smooth field: low-frequency structure, so A4 (approximation) carries most energy
    yy, xx = np.mgrid[0:80, 0:160]
    u = np.sin(2 * np.pi * xx / 80) + 0.5 * np.cos(2 * np.pi * yy / 40)
    u_bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    energies = {b: float(np.sum(u_bands[b].astype(np.float64) ** 2)) for b in BANDS_CF}
    total = sum(energies.values())
    omega = {b: e / total for b, e in energies.items()}
    assert omega["A4"] > omega["W1"]
    assert abs(sum(omega.values()) - 1.0) < 1e-4

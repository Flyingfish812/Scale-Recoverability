"""三层误差测试 (P0-6 §10.7)。

验证: 分母定义一致 (E_total/E_trunc 用 ‖W_b u‖, E_pred 用 ‖W_b u_oracle‖);
三角不等式: ‖W_b(u−û)‖ ≤ ‖W_b(u−u_oracle)‖ + ‖W_b(u_oracle−û)‖。
"""

import numpy as np
import pytest

from luna.wavelet.metrics import compute_three_layer_errors, rel_l2, band_error
from luna.wavelet.transform import decompose_field_2d
from luna.core.constants import BANDS_CF, DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET

rng = np.random.default_rng(53)


def test_three_layer_denominator_definitions():
    """E_total/E_trunc 分母 = ‖W_b u‖; E_pred 分母 = ‖W_b u_oracle‖."""
    u = rng.standard_normal((80, 160))
    v = u + 0.1 * rng.standard_normal((80, 160))
    o = u + 0.02 * rng.standard_normal((80, 160))
    res = compute_three_layer_errors(u, v, o, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)

    u_bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    v_bands = decompose_field_2d(v, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    o_bands = decompose_field_2d(o, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)

    for b in BANDS_CF:
        r = res[b]
        # E_total = band_error(u_band, v_band)
        assert abs(r["E_total"] - band_error(u_bands[b], v_bands[b])) < 1e-6
        # E_trunc = band_error(u_band, o_band)
        assert abs(r["E_trunc"] - band_error(u_bands[b], o_bands[b])) < 1e-6
        # E_pred = band_error(o_band, v_band)
        assert abs(r["E_pred"] - band_error(o_bands[b], v_bands[b])) < 1e-6


def test_triangle_inequality_norms():
    """‖u−û‖ ≤ ‖u−u_oracle‖ + ‖u_oracle−û‖ (逐频带, 范数三角不等式)."""
    u = rng.standard_normal((80, 160))
    v = u + 0.1 * rng.standard_normal((80, 160))
    o = u + 0.02 * rng.standard_normal((80, 160))
    u_bands = decompose_field_2d(u, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    v_bands = decompose_field_2d(v, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    o_bands = decompose_field_2d(o, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    for b in BANDS_CF:
        d_uv = np.linalg.norm((u_bands[b] - v_bands[b]).ravel())
        d_uo = np.linalg.norm((u_bands[b] - o_bands[b]).ravel())
        d_ov = np.linalg.norm((o_bands[b] - v_bands[b]).ravel())
        assert d_uv <= d_uo + d_ov + 1e-9


def test_consistent_global_relation():
    """E_total(A4) ≤ 其他频带? 不假设; 只检查 E_total ≥ 0 且有限."""
    u = rng.standard_normal((64, 64))
    v = u + 0.2 * rng.standard_normal((64, 64))
    o = u + 0.01 * rng.standard_normal((64, 64))
    res = compute_three_layer_errors(u, v, o, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    for b in BANDS_CF:
        for k in ("E_total", "E_trunc", "E_pred"):
            val = res[b][k]
            assert np.isfinite(val) and val >= 0

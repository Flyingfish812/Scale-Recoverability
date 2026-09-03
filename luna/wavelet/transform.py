"""
Wavelet transform utilities for 2D fields.

This is the **single authoritative implementation** of wavelet decomposition.
Previously duplicated in:
    tools/oracle_and_baseline_comparison.py
    tools/run_baselines_efficient.py
    tools/run_controlled_scale_validation.py
    tools/_legacy/wavelet_batch_analysis.py
    tools/plot_per_band_residual.py
    tools/plot_supplementary_figures.py
"""

from __future__ import annotations

import numpy as np
import pywt

from luna.core.constants import DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE, BANDS_CF


def wavelet_coeff_shape(
    shape_2d: tuple[int, int],
    level: int = DEFAULT_LEVEL,
) -> list[tuple[int, ...]]:
    """Return the shapes of wavelet coefficients for a given 2D field shape."""
    coeffs = pywt.wavedec2(
        np.zeros(shape_2d),
        wavelet=DEFAULT_WAVELET,
        level=level,
        mode=DEFAULT_MODE,
    )
    shapes: list[tuple[int, ...]] = [coeffs[0].shape]
    for ch, cv, cd in coeffs[1:]:
        shapes.append((ch.shape, cv.shape, cd.shape))
    return shapes


def zero_coeff_like(coeffs: list) -> list:
    """Create a zero-filled wavelet coefficient list with the same structure.

    Args:
        coeffs: Output of pywt.wavedec2.

    Returns:
        A list where cA is zeros and each detail tuple is (zeros, zeros, zeros).
    """
    out: list = [np.zeros_like(coeffs[0])]
    for c_h, c_v, c_d in coeffs[1:]:
        out.append((np.zeros_like(c_h), np.zeros_like(c_v), np.zeros_like(c_d)))
    return out


def decompose_field_2d(
    field_2d: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, np.ndarray]:
    """Decompose a 2D field into wavelet sub-band components.

    Performs a full DWT, then reconstructs each sub-band independently
    by zeroing out all other coefficients.

    Args:
        field_2d: 2D array (H, W).
        wavelet: Wavelet name (e.g. 'db2').
        level: Decomposition level.
        mode: Boundary extension mode.

    Returns:
        Dict mapping band name → 2D array of that band's spatial component.
        Keys: 'A4', 'W4', 'W3', 'W2', 'W1' (coarse → fine).
    """
    coeffs = pywt.wavedec2(field_2d, wavelet=wavelet, level=level, mode=mode)
    h, w = field_2d.shape
    comp: dict[str, np.ndarray] = {}

    # Approximation band (A4)
    c_a = zero_coeff_like(coeffs)
    c_a[0] = coeffs[0]
    a4 = pywt.waverec2(c_a, wavelet=wavelet, mode=mode)
    comp["A4"] = np.asarray(a4[:h, :w], dtype=np.float32)

    # Detail bands (W1..W4, where W1 is finest)
    for i in range(1, level + 1):
        idx = len(coeffs) - i  # index from the end
        c_w = zero_coeff_like(coeffs)
        c_w[idx] = coeffs[idx]
        wi = pywt.waverec2(c_w, wavelet=wavelet, mode=mode)
        comp[f"W{i}"] = np.asarray(wi[:h, :w], dtype=np.float32)

    return comp


def recompose_field_2d(
    components: dict[str, np.ndarray],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    """Recompose a 2D field from its wavelet sub-band components.

    This is the inverse of decompose_field_2d: sum(A4 + W4 + W3 + W2 + W1).

    Args:
        components: Dict mapping band name → 2D array.
        wavelet: Wavelet name.
        level: Decomposition level.
        mode: Boundary extension mode.

    Returns:
        Reconstructed 2D array.
    """
    # Decompose a zero field to get coefficient shapes
    sample_shape = next(iter(components.values())).shape
    coeffs_template = pywt.wavedec2(
        np.zeros(sample_shape), wavelet=wavelet, level=level, mode=mode
    )

    # Re-decompose each component to get its contribution in coefficient space
    result = np.zeros(sample_shape, dtype=np.float32)
    for band_name, band_field in components.items():
        band_coeffs = pywt.wavedec2(band_field, wavelet=wavelet, level=level, mode=mode)
        recon = pywt.waverec2(band_coeffs, wavelet=wavelet, mode=mode)
        result += np.asarray(recon[: sample_shape[0], : sample_shape[1]], dtype=np.float32)

    return result

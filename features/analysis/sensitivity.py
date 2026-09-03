"""
Hyperparameter sensitivity analysis.

Studies how S_full and band errors vary with:
    - Error threshold τ
    - Band-POD energy threshold η
    - Wavelet type and decomposition level

Replaces: tools/run_sensitivity.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
from luna.wavelet.metrics import band_errors_all, contiguous_recoverable_index


def compute_sensitivity_sweep(
    target: np.ndarray,
    pred: np.ndarray,
    *,
    tau_values: list[float] | None = None,
    wavelet_types: list[str] | None = None,
    levels: list[int] | None = None,
) -> dict[str, Any]:
    """Compute S_full sensitivity to hyperparameter choices.

    Args:
        target: (H, W) ground truth field.
        pred: (H, W) predicted field.
        tau_values: List of τ thresholds to test. Default: [0.01, 0.03, 0.05, 0.07, 0.10].
        wavelet_types: Wavelets to test. Default: ['db2', 'db4', 'sym4'].
        levels: Decomposition levels to test. Default: [3, 4, 5].

    Returns:
        Dict with sensitivity results for each parameter.
    """
    if tau_values is None:
        tau_values = [0.01, 0.03, 0.05, 0.07, 0.10]
    if wavelet_types is None:
        wavelet_types = ["db2", "db4", "sym4"]
    if levels is None:
        levels = [3, 4, 5]

    result: dict[str, Any] = {}

    # τ sensitivity
    tau_results = {}
    base_errors = band_errors_all(target, pred, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE)
    for tau in tau_values:
        err_arr = np.array([base_errors[b] for b in BANDS_CF])
        tau_results[tau] = {
            "S_full": contiguous_recoverable_index(err_arr, tau),
            "bands_below_tau": [b for i, b in enumerate(BANDS_CF) if err_arr[i] <= tau],
        }
    result["tau"] = tau_results

    # Wavelet type sensitivity
    wav_results = {}
    for w in wavelet_types:
        errors = band_errors_all(target, pred, w, DEFAULT_LEVEL, DEFAULT_MODE)
        err_arr = np.array([errors[b] for b in BANDS_CF])
        wav_results[w] = {
            "band_errors": {b: float(errors[b]) for b in BANDS_CF},
            "S_full": contiguous_recoverable_index(err_arr, TAU_DEFAULT),
        }
    result["wavelet_type"] = wav_results

    # Level sensitivity
    level_results = {}
    for lvl in levels:
        errors = band_errors_all(target, pred, DEFAULT_WAVELET, lvl, DEFAULT_MODE)
        err_arr = np.array([errors[b] for b in BANDS_CF])
        level_results[lvl] = {
            "band_errors": {b: float(errors[b]) for b in BANDS_CF},
            "S_full": contiguous_recoverable_index(err_arr, TAU_DEFAULT),
        }
    result["level"] = level_results

    return result

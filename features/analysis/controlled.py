"""
Controlled scale validation — causally validates S_full as a diagnostic tool.

Constructs targets with KNOWN scale content via wavelet band-pass filtering,
then checks whether S_full correctly identifies the known content.

Replaces: tools/run_controlled_scale_validation.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
from luna.wavelet.transform import decompose_field_2d
from luna.wavelet.metrics import compute_S_full, band_errors_all


def build_filtered_target(
    field: np.ndarray,
    keep_bands: list[str],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    """Build a filtered target by keeping only specified wavelet bands.

    Args:
        field: (H, W) source field.
        keep_bands: List of band names to retain (e.g. ['A4', 'W4']).
        wavelet, level, mode: Wavelet parameters.

    Returns:
        (H, W) field containing only the specified bands.
    """
    components = decompose_field_2d(field, wavelet, level, mode)
    result = np.zeros_like(field, dtype=np.float32)
    for b in keep_bands:
        if b in components:
            result += components[b]
    return result


def run_controlled_scale_validation(
    test_fields: np.ndarray,
    test_preds: dict[str, np.ndarray],
    output_dir: str,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Run controlled scale validation experiment.

    For each model's predictions, creates filtered targets with known
    scale content and evaluates S_full diagnostic accuracy.

    Args:
        test_fields: (N, H, W) ground truth fields.
        test_preds: Dict of model_name → (N, H, W) predictions.
        output_dir: Directory for results.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        Validation report with diagnostic accuracy per model and scale configuration.
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Scale configurations to test
    scale_configs = {
        "A4_only": ["A4"],
        "A4_W4": ["A4", "W4"],
        "A4_W4_W3": ["A4", "W4", "W3"],
        "full": BANDS_CF,
    }

    N = test_fields.shape[0]
    results: dict[str, Any] = {"configs": {}, "per_model": {}}

    for config_name, keep_bands in scale_configs.items():
        expected_S = len(keep_bands)
        correct = 0
        total = 0
        per_model_accuracy: dict[str, float] = {}

        for model_name, preds in test_preds.items():
            model_correct = 0
            for i in range(N):
                filtered_target = build_filtered_target(
                    test_fields[i], keep_bands, wavelet, level, mode
                )
                s_val = compute_S_full(
                    filtered_target, preds[i], TAU_DEFAULT, wavelet, level, mode
                )
                if s_val == expected_S:
                    model_correct += 1
                    correct += 1
                total += 1
            per_model_accuracy[model_name] = model_correct / N if N > 0 else 0.0

        results["configs"][config_name] = {
            "expected_S": expected_S,
            "keep_bands": keep_bands,
            "accuracy": correct / total if total > 0 else 0.0,
            "per_model": per_model_accuracy,
        }

    return results

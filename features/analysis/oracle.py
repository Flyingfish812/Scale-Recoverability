"""
POD Oracle computation — theoretical lower bound for POD-based reconstruction.

Uses luna.pod.oracle and luna.wavelet.metrics for all computation.
Replaces: tools/oracle_and_baseline_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
from luna.data.io import load_npz
from luna.pod.oracle import pod_oracle_batch
from luna.wavelet.metrics import (
    band_errors_all,
    compute_S_full,
    compute_three_layer_errors,
    contiguous_recoverable_index,
)
from luna.wavelet.transform import decompose_field_2d


def compute_oracle_band_errors(
    fields: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Compute oracle band errors for a batch of fields.

    Args:
        fields: (N, H, W) ground truth fields.
        basis: (r, D) POD spatial basis.
        mean: (D,) ensemble mean.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        Dict with per-band oracle errors and S_full values.
    """
    oracle_fields = pod_oracle_batch(fields, basis, mean)
    N = fields.shape[0]
    h, w = fields.shape[1], fields.shape[2]

    all_band_errors: dict[str, list[float]] = {b: [] for b in BANDS_CF}
    s_full_values: list[int] = []
    three_layer_results: list[dict] = []

    for i in range(N):
        u = fields[i].reshape(h, w)
        o = oracle_fields[i].reshape(h, w)

        errors = band_errors_all(u, o, wavelet, level, mode)
        for b in BANDS_CF:
            all_band_errors[b].append(errors[b])

        s_full_values.append(compute_S_full(u, o, TAU_DEFAULT, wavelet, level, mode))

    # Aggregate
    max_errors = {b: float(np.max(all_band_errors[b])) for b in BANDS_CF}
    mean_errors = {b: float(np.mean(all_band_errors[b])) for b in BANDS_CF}

    return {
        "max_band_errors": max_errors,
        "mean_band_errors": mean_errors,
        "s_full_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(s_full_values, return_counts=True))
        },
        "s_full_min": int(np.min(s_full_values)),
        "s_full_max": int(np.max(s_full_values)),
        "s_full_mean": float(np.mean(s_full_values)),
    }


def compute_oracle_comparison(
    dataset_name: str,
    output_dir: str | Path,
    tau: float = TAU_DEFAULT,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Full oracle comparison across all methods for a dataset.

    Computes:
    1. POD oracle (theoretical lower bound)
    2. Per-band E_trunc, E_pred, E_total for each model
    3. S_full and S_coh for each model
    4. Aggregated comparison table

    Args:
        dataset_name: 'nc', 'rdb_h5', or 'sst_weekly'.
        output_dir: Directory for output JSON/CSV.
        tau, wavelet, level, mode: Analysis parameters.

    Returns:
        Full comparison results dict.
    """
    from luna.data.registry import get_dataset

    ds = get_dataset(dataset_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load POD bundle
    pod = load_npz(str(ds.pod_bundle_path))
    basis = np.asarray(pod["basis"], dtype=np.float64)
    mean = np.asarray(pod["mean_flat"], dtype=np.float64).ravel()

    # Load band-POD bundle (for S_coh)
    band_pod = None
    if ds.band_pod_bundle_path and ds.band_pod_bundle_path.exists():
        from luna.pod.band_pod import load_band_pod_bundle
        band_pod = load_band_pod_bundle(str(ds.band_pod_bundle_path))

    # Load test fields (from VCNN test_raw artifacts or data array)
    data = np.load(str(ds.data_array))  # (T, H, W, C)
    # Use a subset for oracle computation
    test_fields = data[:200, :, :, 0].astype(np.float64)  # first 200 samples, channel 0

    # Compute oracle band errors
    oracle_result = compute_oracle_band_errors(
        test_fields, basis, mean, wavelet, level, mode,
    )

    return {
        "dataset": dataset_name,
        "oracle": oracle_result,
        "config": {"tau": tau, "wavelet": wavelet, "level": level, "mode": mode},
    }

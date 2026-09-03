"""
Band-wise POD fitting — POD decomposition applied independently to each
wavelet sub-band of the data.

This combines wavelet decomposition with POD to produce a band-specific
low-dimensional subspace, used for S_coh computation.
"""

from __future__ import annotations

import numpy as np

from luna.core.constants import (
    BANDS_CF,
    DEFAULT_WAVELET,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
)
from luna.wavelet.transform import decompose_field_2d
from luna.data.io import load_npz, save_npz


def fit_band_pod(
    fields: np.ndarray,
    pod_max_rank: int = 512,
    pod_energy_threshold: float = 0.99,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit POD on each wavelet sub-band independently.

    Args:
        fields: (N, H, W) array of fields.
        pod_max_rank: Maximum number of POD modes per band.
        pod_energy_threshold: Energy threshold for mode truncation.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        {band: {'mean': (D_b,), 'basis': (r_b, D_b), 'singular_values': (r0_b,)}}
    """
    N = fields.shape[0]
    result: dict[str, dict[str, np.ndarray]] = {}

    for b in BANDS_CF:
        # Extract band for all samples
        band_fields = np.stack([
            decompose_field_2d(fields[i], wavelet, level, mode)[b].ravel()
            for i in range(N)
        ])  # (N, D_b)

        mean_b = band_fields.mean(axis=0, keepdims=True)
        X = band_fields - mean_b
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        S2 = S ** 2
        cumulative = np.cumsum(S2) / S2.sum()
        r = min(
            int(np.searchsorted(cumulative, pod_energy_threshold) + 1),
            pod_max_rank,
            len(S),
        )

        result[b] = {
            "mean": mean_b.ravel().astype(np.float64),
            "basis": Vt[:r, :].astype(np.float64),
            "singular_values": S.astype(np.float64),
        }

    return result


def load_band_pod_bundle(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load a pre-computed band-POD bundle from NPZ."""
    data = load_npz(path)
    bundle: dict[str, dict[str, np.ndarray]] = {}
    for b in BANDS_CF:
        bundle[b] = {
            "mean": np.asarray(data[f"{b}_mean"], dtype=np.float64),
            "basis": np.asarray(data[f"{b}_basis"], dtype=np.float64),
        }
        if f"{b}_S" in data:
            bundle[b]["singular_values"] = np.asarray(data[f"{b}_S"], dtype=np.float64)
    return bundle


def band_pod_project(
    field_band: np.ndarray,
    band_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Project a single band field onto its band-POD subspace.

    Args:
        field_band: (D_b,) or (H_b, W_b) band field.
        band_model: {'mean': (D_b,), 'basis': (r_b, D_b)}.

    Returns:
        Projected band field, same shape as input.
    """
    shape = field_band.shape
    f = np.asarray(field_band, dtype=np.float64).ravel()
    mean = np.asarray(band_model["mean"], dtype=np.float64).ravel()
    basis = np.asarray(band_model["basis"], dtype=np.float64)

    coeff = basis @ (f - mean)
    recon = basis.T @ coeff + mean
    return recon.reshape(shape).astype(np.float32)

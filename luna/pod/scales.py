"""
Physical scale estimation for POD spatial modes.

Uses Fourier energy centroid or peak-finding to estimate the
characteristic spatial scale (ℓ_x, ℓ_y) of each POD mode.

Migrated from l1_pod/scales.py.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _period_factor(scale_definition: str) -> float:
    definition = str(scale_definition).strip().lower()
    if definition == "half_period":
        return 0.5
    if definition == "full_period":
        return 1.0
    raise ValueError("scale_definition must be 'full_period' or 'half_period'")


def _safe_finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _robust_stat(values: np.ndarray) -> dict[str, float]:
    """Compute robust statistics: median, quartiles, mean, std, MAD."""
    v = _safe_finite(values)
    if v.size == 0:
        return {
            "med": float("nan"), "p25": float("nan"), "p75": float("nan"),
            "mean": float("nan"), "std": float("nan"), "mad": float("nan"),
        }
    med = float(np.median(v))
    return {
        "med": med,
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "mad": float(np.median(np.abs(v - med))),
    }


def scale_from_energy_centroid_1d(
    signal: np.ndarray,
    dx: float,
    *,
    demean: bool = True,
    scale_definition: str = "half_period",
) -> float:
    """Estimate scale from the energy-weighted centroid of the 1D FFT spectrum.

    ℓ = 2π / k̄  (× period_factor), where k̄ = Σ(k·E) / Σ(E).
    """
    sig = np.asarray(signal, dtype=np.float64)
    if demean:
        sig = sig - np.mean(sig)

    fft = np.fft.fft(sig)
    k = np.fft.fftfreq(sig.size, d=float(dx)) * 2.0 * math.pi
    energy = np.abs(fft) ** 2

    mask = k > 0
    k_use = np.abs(k[mask])
    e_use = energy[mask]
    denom = float(np.sum(e_use))
    if denom <= 0.0:
        return float("inf")
    k_bar = float(np.sum(k_use * e_use) / denom)
    if k_bar <= 1e-12:
        return float("inf")
    return float(2.0 * math.pi / k_bar) * _period_factor(scale_definition)


def scale_from_peak_1d(
    signal: np.ndarray,
    dx: float,
    *,
    demean: bool = True,
    scale_definition: str = "half_period",
) -> float:
    """Estimate scale from the peak of the 1D FFT amplitude spectrum."""
    sig = np.asarray(signal, dtype=np.float64)
    if demean:
        sig = sig - np.mean(sig)

    fft = np.fft.fft(sig)
    k = np.fft.fftfreq(sig.size, d=float(dx)) * 2.0 * math.pi
    amp = np.abs(fft)
    mask = k > 0
    if not np.any(mask):
        return float("inf")

    k_use = k[mask]
    a_use = amp[mask]
    k_star = float(k_use[int(np.argmax(a_use))])
    if not np.isfinite(k_star) or k_star <= 1e-12:
        return float("inf")
    return float(2.0 * math.pi / k_star) * _period_factor(scale_definition)


def estimate_mode_scales(
    mode_hw: np.ndarray,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    method: str = "energy_centroid",
    scale_definition: str = "half_period",
) -> dict[str, float]:
    """Estimate characteristic spatial scales (ℓ_x, ℓ_y) for a 2D POD mode.

    Args:
        mode_hw: (H, W) POD spatial mode.
        dx, dy: Grid spacing.
        method: 'energy_centroid' or 'peak'.
        scale_definition: 'half_period' or 'full_period'.

    Returns:
        Dict with ell_x_med, ell_y_med, ell_min, ell_geo, etc.
    """
    m = str(method).strip().lower()
    if m in ("peak", "argmax", "a_peak"):
        scale_fn = scale_from_peak_1d
    else:
        scale_fn = scale_from_energy_centroid_1d

    mode = np.asarray(mode_hw, dtype=np.float64)
    h, w = mode.shape

    ell_x = [scale_fn(mode[row, :], dx, scale_definition=scale_definition) for row in range(h)]
    ell_y = [scale_fn(mode[:, col], dy, scale_definition=scale_definition) for col in range(w)]

    sx = _robust_stat(np.asarray(ell_x, dtype=np.float64))
    sy = _robust_stat(np.asarray(ell_y, dtype=np.float64))

    med_x = sx["med"]
    med_y = sy["med"]
    ell_min = float(np.nanmin([med_x, med_y]))
    ell_geo = float(np.sqrt(med_x * med_y)) if np.isfinite(med_x) and np.isfinite(med_y) else float("nan")

    return {
        "ell_x_med": med_x,
        "ell_x_p25": sx["p25"], "ell_x_p75": sx["p75"],
        "ell_x_mean": sx["mean"], "ell_x_std": sx["std"], "ell_x_mad": sx["mad"],
        "ell_y_med": med_y,
        "ell_y_p25": sy["p25"], "ell_y_p75": sy["p75"],
        "ell_y_mean": sy["mean"], "ell_y_std": sy["std"], "ell_y_mad": sy["mad"],
        "ell_min": ell_min,
        "ell_geo": ell_geo,
    }


def reduce_mode_channels(
    q_modes_rhwc: np.ndarray,
    reduce_mode: str = "l2",
) -> np.ndarray:
    """Reduce multi-channel POD modes to a single channel.

    Args:
        q_modes_rhwc: (R, H, W, C) array of spatial modes.
        reduce_mode: 'l2' (L2 norm), 'sum', 'u' (channel 0), 'v' (channel 1).

    Returns:
        (R, H, W) array.
    """
    q = np.asarray(q_modes_rhwc, dtype=np.float64)
    mode = str(reduce_mode).strip().lower()
    if mode == "l2":
        return np.sqrt(np.sum(q ** 2, axis=-1)).astype(np.float32)
    if mode == "sum":
        return np.sum(q, axis=-1).astype(np.float32)
    if mode == "u":
        return q[..., 0].astype(np.float32)
    if mode == "v":
        if q.shape[-1] < 2:
            raise ValueError("reduce_mode='v' requires at least 2 channels")
        return q[..., 1].astype(np.float32)
    raise ValueError(f"Unknown reduce_mode: {reduce_mode}")


def build_scale_table(
    q_modes_rhw: np.ndarray,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    method: str = "energy_centroid",
    scale_definition: str = "half_period",
) -> list[dict[str, Any]]:
    """Build a per-mode physical scale table.

    Args:
        q_modes_rhw: (R, H, W) or (R, H, W, C) spatial modes.
        dx, dy: Grid spacing.
        method: Scale estimation method.
        scale_definition: Period factor.

    Returns:
        List of per-mode dicts with scale statistics.
    """
    q_modes = np.asarray(q_modes_rhw, dtype=np.float32)
    if q_modes.ndim == 4:
        q_modes = reduce_mode_channels(q_modes)

    rows: list[dict[str, Any]] = []
    for idx in range(int(q_modes.shape[0])):
        row = {"mode": idx}
        row.update(
            estimate_mode_scales(
                q_modes[idx],
                dx=dx, dy=dy,
                method=method,
                scale_definition=scale_definition,
            )
        )
        rows.append(row)
    return rows

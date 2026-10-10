"""
Band-error metrics — denominator stability audit.

Background: the per-band error of the paper is computed in the *coefficient* domain of a 2D DWT (db2, level 4, periodization):

    E(A4) = ||a4_p - a4_t|| / (||a4_t|| + EPS)
    E(Wi) = sqrt(sum_hvd ||dp - dt||^2) / (sqrt(sum_hvd ||dt||^2) + EPS)

The denominators ||a4_t|| and sqrt(sum ||dt||^2) are the band-wise L2 norms of the target field u. If a band norm is near zero, the relative band error is unstable / can be amplified. This module audits those denominators:

    q_b(u) = ||W_b u||_2^2 / ||u||_2^2   (energy fraction, Parseval-consistent)
    abs:    ||W_b u||_2                  (absolute band norm)

Near-zero flags use pre-defined thresholds (see configs/metrics.yaml):
    abs norm  < eps_abs   -> near-zero
    energy    < eps_rel   -> near-zero
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pywt

from luna.wavelet.transform import decompose_field_2d

# Canonical paper configuration (must match s01_fix_figure3 / luna defaults)
DEFAULT_WAVELET = "db2"
DEFAULT_LEVEL = 4
DEFAULT_MODE = "periodization"
DEFAULT_BANDS = ["A4", "W4", "W3", "W2", "W1"]


def band_coefficient_norms(
    field_2d: np.ndarray,
    bands: Sequence[str] = DEFAULT_BANDS,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> Dict[str, float]:
    """Return per-band coefficient L2 norms of a 2D field.

    Identical denominator definition as the paper's per-band error:
      - A4 norm  = ||cA||_2
      - Wi norm  = sqrt(||cH||^2 + ||cV||^2 + ||cD||^2)

    Also returns field L2 norm (channel field) for the energy fraction.
    """
    coeffs = pywt.wavedec2(field_2d, wavelet=wavelet, level=level, mode=mode)
    out: Dict[str, float] = {}
    out["A4"] = float(np.linalg.norm(coeffs[0]))
    for i, det in enumerate(coeffs[1:]):  # finest first
        band = f"W{level - i}"
        out[band] = float(np.sqrt(sum(np.sum(d ** 2) for d in det)))
    out["_field"] = float(np.linalg.norm(field_2d))
    return out


def energy_fractions(norms: Dict[str, float], bands: Sequence[str] = DEFAULT_BANDS) -> Dict[str, float]:
    """q_b = ||W_b u||^2 / ||u||^2 for each band (Parseval: sum q_b = 1)."""
    total = float(norms["_field"]) ** 2
    if total <= 0.0:
        return {b: float("nan") for b in bands}
    return {b: float(norms[b] ** 2) / total for b in bands}


def percentile_stats(values: np.ndarray, quantiles: Sequence[float] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> Dict[str, float]:
    """Return percentile statistics for a 1-D array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"min": float("nan"), "1%": float("nan"), "5%": float("nan"),
                "25%": float("nan"), "median": float("nan"), "75%": float("nan"),
                "95%": float("nan"), "max": float("nan")}
    q = np.percentile(arr, [100.0 * x for x in quantiles])
    return {
        "min": float(q[0]), "1%": float(q[1]), "5%": float(q[2]),
        "25%": float(q[3]), "median": float(q[4]), "75%": float(q[5]),
        "95%": float(q[6]), "max": float(q[7]),
    }


def audit_band_denominators(
    fields_2d: np.ndarray,
    bands: Sequence[str] = DEFAULT_BANDS,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-6,
) -> Dict[str, dict]:
    """Audit band denominators over a set of 2D fields.

    Args:
        fields_2d: (N, H, W) array of target fields. eps_abs: absolute band-norm near-zero threshold. eps_rel: energy-fraction near-zero threshold.

    Returns:
        dict per band with:
          abs_norm  : {percentiles..., mean, near_zero_count, n}
          energy    : {percentiles..., mean, near_zero_count, n}
          near_zero: {abs: n, rel: n, any: n}
    """
    fields = np.asarray(fields_2d, dtype=np.float64)
    n = fields.shape[0]

    band_norms: Dict[str, np.ndarray] = {b: np.empty(n, dtype=np.float64) for b in bands}
    field_norms = np.empty(n, dtype=np.float64)

    for i in range(n):
        norms = band_coefficient_norms(fields[i], bands=bands, wavelet=wavelet, level=level, mode=mode)
        for b in bands:
            band_norms[b][i] = norms[b]
        field_norms[i] = norms["_field"]

    report: Dict[str, dict] = {}
    for b in bands:
        abs_arr = band_norms[b]
        total = field_norms ** 2
        safe = total > 0.0
        energy = np.full(n, float("nan"))
        energy[safe] = abs_arr[safe] ** 2 / total[safe]

        abs_stats = percentile_stats(abs_arr)
        abs_stats["mean"] = float(np.mean(abs_arr))
        abs_stats["near_zero_count"] = int(np.sum(abs_arr < eps_abs))
        abs_stats["n"] = int(n)

        en_stats = percentile_stats(energy[~np.isnan(energy)] if np.any(~np.isnan(energy)) else energy)
        en_stats["mean"] = float(np.nanmean(energy))
        en_stats["near_zero_count"] = int(np.sum(energy < eps_rel))
        en_stats["n"] = int(n)

        report[b] = {
            "abs_norm": abs_stats,
            "energy_fraction": en_stats,
            "near_zero": {
                "abs_count": int(np.sum(abs_arr < eps_abs)),
                "rel_count": int(np.sum(energy < eps_rel)),
                "any_count": int(np.sum((abs_arr < eps_abs) | (energy < eps_rel))),
            },
        }

    # Overall field-norm stats (GER denominator context)
    report["_field"] = {
        "abs_norm": percentile_stats(field_norms),
        "abs_norm_mean": float(np.mean(field_norms)),
        "n": int(n),
    }
    return report


def projected_denominators(
    fields_2d: np.ndarray,
    band_pod_models: Dict[str, Dict[str, np.ndarray]],
    bands: Sequence[str] = DEFAULT_BANDS,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-6,
) -> Dict[str, dict]:
    """Stability audit of the projected norms that form the denominators of E_coh.

    The denominator of E_coh is ``||Pi_b(W_b u)||_2`` with the affine projection
    ``Pi_b(w) = mu_b + P_b(w - mu_b)`` onto the band's POD subspace stored in
    ``band_pod_models``. The same near-zero tests as for the band norms are
    applied to the projected norms: an absolute threshold on ``||Pi_b||`` and a
    relative threshold on the projected energy fraction ``||Pi_b||^2 / ||u||^2``.
    The squared-norm ratio ``||Pi_b||^2 / ||W_b u||^2`` (the quantity gamma_b of
    the co-energy section) is also reported; it can lie on either side of one
    because the projection is affine, so it is not compared with a threshold.
    """
    fields = np.asarray(fields_2d, dtype=np.float64)
    n = fields.shape[0]

    proj_norms: Dict[str, np.ndarray] = {b: np.empty(n, dtype=np.float64) for b in bands}
    band_norms: Dict[str, np.ndarray] = {b: np.empty(n, dtype=np.float64) for b in bands}
    field_norms = np.empty(n, dtype=np.float64)

    for i in range(n):
        coefficients = decompose_field_2d(fields[i], wavelet=wavelet, level=level, mode=mode)
        field_norms[i] = float(np.linalg.norm(fields[i]))
        for b in bands:
            model = band_pod_models[b]
            mean = np.asarray(model["mean"], dtype=np.float64).ravel()
            basis = np.asarray(model["basis"], dtype=np.float64)
            w = np.asarray(coefficients[b], dtype=np.float64).ravel()
            projected = basis.T @ (basis @ (w - mean)) + mean
            band_norms[b][i] = float(np.linalg.norm(w))
            proj_norms[b][i] = float(np.linalg.norm(projected))

    report: Dict[str, dict] = {}
    for b in bands:
        ratio = proj_norms[b] ** 2 / np.maximum(band_norms[b] ** 2, 1e-300)
        energy = proj_norms[b] ** 2 / np.maximum(field_norms ** 2, 1e-300)
        report[b] = {
            "proj_norm": percentile_stats(proj_norms[b]),
            "ratio": percentile_stats(ratio),
            "energy_fraction": percentile_stats(energy),
            "near_zero": {
                "abs_count": int(np.sum(proj_norms[b] < eps_abs)),
                "rel_count": int(np.sum(energy < eps_rel)),
                "any_count": int(np.sum((proj_norms[b] < eps_abs) | (energy < eps_rel))),
            },
        }
    return report

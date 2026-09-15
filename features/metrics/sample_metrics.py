"""Sample-level error metrics used throughout the paper.

Definitions follow the conventions of the main text: the global error (GER) is
computed over both channels, while the wavelet-band errors, the recoverability
counts S_full/S_coh and the vorticity/gradient diagnostics use the streamwise
vorticity channel only.

The wavelet decomposition is db2 with periodic boundary handling at level 4, so
the five bands are A4, W4, W3, W2 and W1.
"""

from __future__ import annotations

import numpy as np
import pywt

BANDS = ["A4", "W4", "W3", "W2", "W1"]
WAVELET = "db2"
LEVEL = 4
EPS = 1e-12

_SIGMA_CODES = {0.0: "s0000", 0.001: "s0010", 0.01: "s0100", 0.1: "s1000"}


def sigma_to_code(sigma: float) -> str:
    """Directory name used for a noise level, e.g. ``0.01`` -> ``s0100``."""
    if sigma in _SIGMA_CODES:
        return _SIGMA_CODES[sigma]
    return f"s{int(round(sigma * 10000)):04d}"


def laplacian(field_2d: np.ndarray) -> np.ndarray:
    """Discrete Laplacian of a 2-D field, used as a vorticity proxy."""
    return (
        np.gradient(np.gradient(field_2d, axis=0), axis=0)
        + np.gradient(np.gradient(field_2d, axis=1), axis=1)
    )


def gradient_rmse(target: np.ndarray, pred: np.ndarray) -> float:
    """Combined RMSE of the two first-order spatial derivatives."""
    gy_t = np.gradient(target, axis=0)
    gx_t = np.gradient(target, axis=1)
    gy_p = np.gradient(pred, axis=0)
    gx_p = np.gradient(pred, axis=1)
    err_x = np.sqrt(np.mean((gx_t - gx_p) ** 2))
    err_y = np.sqrt(np.mean((gy_t - gy_p) ** 2))
    return float(np.sqrt(err_x**2 + err_y**2))


def band_errors(target_2d: np.ndarray, pred_2d: np.ndarray, tau: float = 0.05) -> dict:
    """Per-band relative errors and the number of bands below ``tau``."""
    coeffs_pred = pywt.wavedec2(pred_2d, WAVELET, level=LEVEL, mode="periodization")
    coeffs_target = pywt.wavedec2(target_2d, WAVELET, level=LEVEL, mode="periodization")

    errs: dict[str, float] = {}
    n_below = 0

    a4_p, a4_t = coeffs_pred[0], coeffs_target[0]
    e = float(np.linalg.norm(a4_p - a4_t) / (np.linalg.norm(a4_t) + EPS))
    errs["A4"] = e
    n_below += int(e < tau)

    for j, (det_p, det_t) in enumerate(zip(coeffs_pred[1:], coeffs_target[1:])):
        err_sum = sum(float(np.sum((dp - dt) ** 2)) for dp, dt in zip(det_p, det_t))
        norm_sum = sum(float(np.sum(dt**2)) for dt in det_t)
        e = float(np.sqrt(err_sum) / (np.sqrt(norm_sum) + EPS))
        errs[BANDS[j + 1]] = e
        n_below += int(e < tau)

    return {"band_errors": errs, "n_bands_below_tau": n_below}


def compute_sample_metrics(
    output_nchw: np.ndarray,
    target_nchw: np.ndarray,
    sample_idx: int,
    tau: float = 0.05,
) -> dict:
    """All sample-level metrics for one snapshot of a batch.

    Returns ``GER`` over both channels, ``S_full`` (number of bands with a
    relative error below ``tau``), the five per-band errors, and the vorticity
    and gradient RMSE on the streamwise channel.
    """
    o = output_nchw[sample_idx].ravel()
    t = target_nchw[sample_idx].ravel()
    ger = float(np.linalg.norm(o - t) / (np.linalg.norm(t) + EPS))

    out_2d = output_nchw[sample_idx, 0]
    tgt_2d = target_nchw[sample_idx, 0]

    band = band_errors(tgt_2d, out_2d, tau=tau)
    errs = band["band_errors"]

    vort_rmse = float(np.sqrt(np.mean((laplacian(tgt_2d) - laplacian(out_2d)) ** 2)))

    return {
        "GER": ger,
        "S_full": band["n_bands_below_tau"],
        "E_W1": errs.get("W1", float("nan")),
        "E_W2": errs.get("W2", float("nan")),
        "E_W3": errs.get("W3", float("nan")),
        "E_W4": errs.get("W4", float("nan")),
        "E_A4": errs.get("A4", float("nan")),
        "vorticity_RMSE": vort_rmse,
        "gradient_RMSE": gradient_rmse(tgt_2d, out_2d),
    }

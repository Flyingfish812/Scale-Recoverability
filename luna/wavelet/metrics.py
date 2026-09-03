"""
Wavelet-band error metrics for scale-resolved field reconstruction evaluation.

This module is the **single authoritative implementation** of:
    - Relative L2 error (rel_l2)
    - Per-band direct/coherent errors
    - S_full: contiguous recoverable scale level
    - S_coh: POD-coherent subspace recoverability
    - Three-layer error decomposition: E_trunc, E_pred, E_total
    - Oracle audit table (rank sweep for band-wise representation sufficiency)

Key references:
    - 20260710 feedback: "截断误差与预测误差千万不能混淆"
    - S_full defined as contiguous bands with band_error ≤ τ, counting from A4 downward
"""

from __future__ import annotations

import numpy as np

from luna.core.constants import (
    BANDS_CF,
    BANDS_FC,
    N_BANDS,
    TAU_DEFAULT,
    DEFAULT_WAVELET,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    EPS,
)
from luna.wavelet.transform import decompose_field_2d


# ── Basic metrics ──────────────────────────────────────────────────

def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    """Relative L2 error: ‖a − b‖₂ / ‖b‖₂."""
    return float(
        np.linalg.norm(a.ravel() - b.ravel())
        / (np.linalg.norm(b.ravel()) + eps)
    )


def band_error(
    target_band: np.ndarray,
    pred_band: np.ndarray,
    eps: float = EPS,
) -> float:
    """Per-band relative L2 error.

    Args:
        target_band: Ground truth band component (2D).
        pred_band: Predicted/reconstructed band component (2D).

    Returns:
        ‖pred − target‖₂ / ‖target‖₂
    """
    return rel_l2(pred_band, target_band, eps=eps)


def band_errors_all(
    target: np.ndarray,
    pred: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, float]:
    """Compute per-band relative L2 errors for all 5 bands.

    Args:
        target: Ground truth 2D field (H, W).
        pred: Predicted 2D field (H, W).
        wavelet, level, mode: Wavelet parameters.

    Returns:
        Dict: {'A4': 0.012, 'W4': 0.034, 'W3': 0.056, 'W2': 0.089, 'W1': 0.123}
    """
    target_bands = decompose_field_2d(target, wavelet, level, mode)
    pred_bands = decompose_field_2d(pred, wavelet, level, mode)
    return {
        b: band_error(target_bands[b], pred_bands[b])
        for b in BANDS_CF
    }


# ── S_full: contiguous recoverable scale level ─────────────────────

def contiguous_recoverable_index(
    errors_coarse_to_fine: np.ndarray,
    tau: float = TAU_DEFAULT,
) -> int:
    """Count how many bands (from coarsest A4 downward) satisfy error ≤ τ.

    Args:
        errors_coarse_to_fine: Array of 5 band errors [A4, W4, W3, W2, W1].
        tau: Error threshold.

    Returns:
        Integer 0–5: number of contiguous recoverable bands.
        0 = no band recoverable, 5 = all bands recoverable.
    """
    arr = np.asarray(errors_coarse_to_fine, dtype=np.float64)
    out = 0
    for i, v in enumerate(arr, start=1):
        if np.isfinite(v) and v <= float(tau):
            out = i
        else:
            break
    return int(out)


def compute_S_full(
    target: np.ndarray,
    pred: np.ndarray,
    tau: float = TAU_DEFAULT,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> int:
    """Compute S_full: contiguous recoverable scale level.

    S_full ∈ {0, 1, 2, 3, 4, 5}.
    0 = no band recoverable; 5 = all 5 bands (A4..W1) recoverable.

    Args:
        target: Ground truth 2D field.
        pred: Predicted 2D field.
        tau: Error threshold for band recoverability.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        S_full value.
    """
    errors = band_errors_all(target, pred, wavelet, level, mode)
    err_arr = np.array([errors[b] for b in BANDS_CF])
    return contiguous_recoverable_index(err_arr, tau)


# ── S_coh: POD-coherent subspace recoverability ────────────────────

def compute_S_coh(
    target: np.ndarray,
    pred: np.ndarray,
    band_pod_models: dict[str, dict[str, np.ndarray]],
    tau: float = TAU_DEFAULT,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> int:
    """Compute S_coh: recoverable scale level within POD-coherent subspace.

    Projects both target and prediction onto each band's POD subspace,
    then computes band errors in the projected space.

    Args:
        target: Ground truth 2D field.
        pred: Predicted 2D field.
        band_pod_models: Dict from band_pod_fit:
            {band: {'mean': ndarray, 'basis': ndarray}}.
        tau: Error threshold.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        S_coh value ∈ {0..5}.
    """
    target_bands = decompose_field_2d(target, wavelet, level, mode)
    pred_bands = decompose_field_2d(pred, wavelet, level, mode)

    coherent_errors = []
    for b in BANDS_CF:
        if b not in band_pod_models:
            coherent_errors.append(np.inf)
            continue
        model = band_pod_models[b]
        mean = np.asarray(model["mean"], dtype=np.float64).ravel()
        basis = np.asarray(model["basis"], dtype=np.float64)

        t_flat = np.asarray(target_bands[b], dtype=np.float64).ravel() - mean
        p_flat = np.asarray(pred_bands[b], dtype=np.float64).ravel() - mean

        # Project onto POD subspace
        t_coeff = basis @ t_flat
        p_coeff = basis @ p_flat
        t_proj = basis.T @ t_coeff + mean
        p_proj = basis.T @ p_coeff + mean

        err = rel_l2(p_proj, t_proj)
        coherent_errors.append(err)

    return contiguous_recoverable_index(np.array(coherent_errors), tau)


# ── Three-layer error decomposition (导师反馈 2026-07-10) ──────────

def compute_three_layer_errors(
    target: np.ndarray,
    pred: np.ndarray,
    oracle: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, dict[str, float]]:
    """Compute the three-layer error decomposition for each wavelet band.

    Per the advisor's feedback (2026-07-10):

        u - û = (u - u_oracle) + (u_oracle - û)

    Three layers:
        E_total(b) = ‖W_b(u) − W_b(û)‖₂ / ‖W_b(u)‖₂
            Complete system error (representation + prediction).
        E_trunc(b) = ‖W_b(u) − W_b(u_oracle)‖₂ / ‖W_b(u)‖₂
            Truncation/representation error — how well can the POD basis
            represent this band?
        E_pred(b)  = ‖W_b(u_oracle) − W_b(û)‖₂ / ‖W_b(u_oracle)‖₂
            Prediction error — how well does the model predict the
            representable component?

    Args:
        target: Ground truth 2D field u.
        pred: Model prediction û.
        oracle: POD oracle reconstruction u_oracle.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        {band: {'E_total': ..., 'E_trunc': ..., 'E_pred': ...}}
    """
    target_bands = decompose_field_2d(target, wavelet, level, mode)
    pred_bands = decompose_field_2d(pred, wavelet, level, mode)
    oracle_bands = decompose_field_2d(oracle, wavelet, level, mode)

    result: dict[str, dict[str, float]] = {}
    for b in BANDS_CF:
        t = target_bands[b]
        p = pred_bands[b]
        o = oracle_bands[b]

        e_total = band_error(t, p)
        e_trunc = band_error(t, o)
        # E_pred uses oracle as reference (denominator = ‖oracle‖)
        e_pred = band_error(o, p)

        result[b] = {
            "E_total": e_total,
            "E_trunc": e_trunc,
            "E_pred": e_pred,
        }

    return result


# ── Oracle audit table (rank sweep) ────────────────────────────────

def compute_oracle_audit_table(
    target_fields: np.ndarray,
    pod_basis_flat: np.ndarray,
    pod_mean_flat: np.ndarray,
    ranks: list[int],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[int, dict[str, float]]:
    """Compute per-band oracle truncation errors for multiple POD ranks.

    This answers: "What rank is needed for each band's E_trunc < τ/5?"

    Args:
        target_fields: Array of shape (N, H, W) — single-channel, or (N, H, W, C) — multi-channel.
        pod_basis_flat: POD basis, shape (max_rank, D) where D = H*W (*C if multi-channel).
        pod_mean_flat: POD mean, shape (D,).
        ranks: List of truncation ranks to evaluate.
        wavelet, level, mode: Wavelet parameters.

    Returns:
        {rank: {'A4': max_E_trunc, 'W4': ..., 'W3': ..., 'W2': ..., 'W1': ...}}
    """
    mean = np.asarray(pod_mean_flat, dtype=np.float64).ravel()
    basis = np.asarray(pod_basis_flat, dtype=np.float64)
    N = target_fields.shape[0]
    D = basis.shape[1]

    ndim = target_fields.ndim
    if ndim == 4:
        # (N, H, W, C) — multi-channel: keep full for oracle, use ch0 for wavelet
        H, W, C = target_fields.shape[1], target_fields.shape[2], target_fields.shape[3]
        targets_flat = target_fields.reshape(N, -1).astype(np.float64)  # (N, H*W*C)
    elif ndim == 3:
        H, W = target_fields.shape[1], target_fields.shape[2]
        C = 1
        targets_flat = target_fields.reshape(N, -1).astype(np.float64)  # (N, H*W)
    else:
        targets_flat = np.asarray(target_fields, dtype=np.float64)
        h = int(np.sqrt(D))
        if h * h == D:
            H = W = h
            C = 1
        else:
            raise ValueError(f"Cannot infer shape from D={D}.")

    result: dict[int, dict[str, float]] = {}
    for r in ranks:
        rclamped = min(r, basis.shape[0])
        basis_r = basis[:rclamped, :]
        band_errors_all_rank: dict[str, list[float]] = {b: [] for b in BANDS_CF}

        for i in range(N):
            u_flat = targets_flat[i]

            # POD oracle reconstruction
            coeff = basis_r @ (u_flat - mean)
            u_oracle_flat = basis_r.T @ coeff + mean

            # Extract channel 0 for wavelet metrics
            if ndim == 4:
                u = u_flat.reshape(H, W, C)[:, :, 0]
                u_oracle = u_oracle_flat.reshape(H, W, C)[:, :, 0]
            else:
                u = u_flat.reshape(H, W)
                u_oracle = u_oracle_flat.reshape(H, W)

            # Per-band truncation error
            u_bands = decompose_field_2d(u, wavelet, level, mode)
            o_bands = decompose_field_2d(u_oracle, wavelet, level, mode)
            for b in BANDS_CF:
                band_errors_all_rank[b].append(band_error(u_bands[b], o_bands[b]))

        result[r] = {b: float(np.max(band_errors_all_rank[b])) for b in BANDS_CF}

    return result

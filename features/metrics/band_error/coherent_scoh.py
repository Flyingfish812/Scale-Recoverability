"""
Paper-definition coherent recoverability (S_coh) — dual-channel, coefficient-domain.

Background: the thesis S_coh is computed on the FULL dual-channel field
(80x160x2 -> 25600), not the single channel used by luna.compute_S_coh.
Per band b:
    t_b = concat_channel( decompose(target[:,:,ch])[b] )  (25600,)
    y_b = concat_channel( decompose(pred[:,:,ch])[b] )
    yc = y_b - mean_b ; tc = t_b - mean_b
    coh_err_b = || (yc - tc) @ U_b^T || / ( || tc @ U_b^T || + eps )
    S_coh = max k such that coh_err_{A4..band_k} <= tau (contiguous)

This matches _legacy/scripts_main/20260719-2/p15_scoh_full_recomputation.py
(analyze_sample) and the thesis facts (e.g. M=30 sigma=0 sample 49: Ridge=2,
VCNN=5). luna.compute_S_coh (2D single-channel, reconstruction-domain) is a
separate P0 implementation used for unit tests; this module is the paper
definition used by supplementary paired statistics.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from luna.core.constants import BANDS_CF
from luna.wavelet.transform import decompose_field_2d

EPS = 1e-12


def contiguous_recoverable_index(errors_cf: Sequence[float], tau: float) -> int:
    """Max k with errors[A4..band_k] all <= tau (contiguous from coarse)."""
    out = 0
    for i, v in enumerate(errors_cf, start=1):
        if np.isfinite(v) and v <= tau:
            out = i
        else:
            break
    return int(out)


def _concat_channel_bands(field_hwc: np.ndarray, wavelet: str, level: int, mode: str,
                          bands: Sequence[str]) -> Dict[str, np.ndarray]:
    """Decompose each channel and concatenate band components -> {band: (D,)},
    where D = H*W*C (dual-channel field)."""
    h, w, c = field_hwc.shape
    out: Dict[str, np.ndarray] = {}
    for b in bands:
        stack = np.empty((h, w, c), dtype=np.float64)
        for ch in range(c):
            stack[:, :, ch] = decompose_field_2d(field_hwc[:, :, ch], wavelet, level, mode)[b]
        out[b] = stack.ravel()
    return out


def compute_target_bands_bundle(
    target_nhwc: np.ndarray,
    bands: Sequence[str] = BANDS_CF,
    wavelet: str = "db2",
    level: int = 4,
    mode: str = "periodization",
) -> np.ndarray:
    """Precompute dual-channel band components for a batch of targets.

    Args:
        target_nhwc: (B, C, H, W) target fields (single sample: (C, H, W)).
    Returns:
        (B, n_bands, H*W*C) float32 array; band order = `bands`.
    """
    arr = np.asarray(target_nhwc, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None, :, :, :]
    B, C, H, W = arr.shape
    out = np.empty((B, len(bands), H * W * C), dtype=np.float32)
    for i in range(B):
        field = arr[i].transpose(1, 2, 0)  # (H, W, C)
        comps = _concat_channel_bands(field, wavelet, level, mode, bands)
        for j, b in enumerate(bands):
            out[i, j] = comps[b].astype(np.float32)
    return out


def compute_scoh_with_target_bands(
    pred_hwc: np.ndarray,
    target_bands: np.ndarray,
    band_pod: Dict[str, Dict[str, np.ndarray]],
    *,
    tau: float = 0.05,
    wavelet: str = "db2",
    level: int = 4,
    mode: str = "periodization",
    bands: Sequence[str] = BANDS_CF,
) -> int:
    """S_coh using precomputed target band components (see compute_target_bands_bundle).

    Args:
        pred_hwc: (H, W, C) predicted field.
        target_bands: (n_bands, H*W*C) precomputed target band components.
    """
    pred = np.asarray(pred_hwc, dtype=np.float64)
    pred_comps = _concat_channel_bands(pred, wavelet, level, mode, bands)
    coh_errors: list[float] = []
    for j, b in enumerate(bands):
        if b not in band_pod:
            coh_errors.append(float("inf"))
            continue
        t_flat = np.asarray(target_bands[j], dtype=np.float64)
        y_flat = pred_comps[b]
        mean = np.asarray(band_pod[b]["mean"], dtype=np.float64).ravel()
        basis = np.asarray(band_pod[b]["basis"], dtype=np.float64)
        if basis.shape[1] != t_flat.size:
            raise ValueError(
                f"band {b}: basis {basis.shape} incompatible with field {t_flat.size}"
            )
        yc = y_flat - mean
        tc = t_flat - mean
        diff_proj = float(np.linalg.norm((yc - tc) @ basis.T))
        tgt_proj = float(np.linalg.norm(tc @ basis.T))
        coh_errors.append(diff_proj / (tgt_proj + EPS))
    return contiguous_recoverable_index(coh_errors, tau)


def compute_scoh_paper_definition(
    target: np.ndarray,
    pred: np.ndarray,
    band_pod: Dict[str, Dict[str, np.ndarray]],
    *,
    tau: float = 0.05,
    wavelet: str = "db2",
    level: int = 4,
    mode: str = "periodization",
    bands: Sequence[str] = BANDS_CF,
) -> int:
    """Paper-definition S_coh for a dual-channel (H, W, C) field pair."""
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if target.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"Expected (H, W, C) fields, got {target.shape} / {pred.shape}")
    h, w, c = target.shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    coh_errors: list[float] = []
    for b in bands:
        if b not in band_pod:
            coh_errors.append(float("inf"))
            continue
        t_stack = np.empty((h, w, c), dtype=np.float64)
        p_stack = np.empty((h, w, c), dtype=np.float64)
        for ch in range(c):
            t_stack[:, :, ch] = decompose_field_2d(target[:, :, ch], wavelet, level, mode)[b]
            p_stack[:, :, ch] = decompose_field_2d(pred[:, :, ch], wavelet, level, mode)[b]
        t_flat = t_stack.ravel()
        y_flat = p_stack.ravel()

        mean = np.asarray(band_pod[b]["mean"], dtype=np.float64).ravel()
        basis = np.asarray(band_pod[b]["basis"], dtype=np.float64)
        if basis.shape[1] != t_flat.size:
            raise ValueError(
                f"band {b}: basis {basis.shape} incompatible with field {t_flat.size}"
            )

        yc = y_flat - mean
        tc = t_flat - mean
        diff_proj = float(np.linalg.norm((yc - tc) @ basis.T))
        tgt_proj = float(np.linalg.norm(tc @ basis.T))
        coh_errors.append(diff_proj / (tgt_proj + EPS))

    return contiguous_recoverable_index(coh_errors, tau)

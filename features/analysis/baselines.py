"""
Baseline methods: Gappy POD and POD-LS (Least Squares).

Provides traditional linear estimation baselines for comparison
with learned methods (VCNN, Ridge, MLP).

Uses luna.pod and luna.wavelet for all computation.
Replaces: tools/run_baselines_efficient.py, tools/run_baselines_gappy_pod_ls.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from luna.core.constants import BANDS_CF, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE, EPS
from luna.wavelet.transform import decompose_field_2d
from luna.wavelet.metrics import rel_l2, band_error, contiguous_recoverable_index


def run_gappy_pod(
    target_field: np.ndarray,
    mask_hw: np.ndarray,
    pod_basis: np.ndarray,
    pod_mean: np.ndarray,
    rank: int = 32,
) -> np.ndarray:
    """Gappy POD reconstruction using least squares on observed points.

    Args:
        target_field: (H, W) ground truth field.
        mask_hw: (H, W) boolean observation mask.
        pod_basis: (r_max, D) POD spatial basis.
        pod_mean: (D,) ensemble mean.
        rank: Truncation rank for Gappy POD.

    Returns:
        (H, W) reconstructed field.
    """
    h, w = target_field.shape
    D = h * w
    mask_flat = np.asarray(mask_hw, dtype=bool).ravel()
    obs_indices = np.flatnonzero(mask_flat)

    u_flat = np.asarray(target_field, dtype=np.float64).ravel()
    mean = np.asarray(pod_mean, dtype=np.float64).ravel()
    basis = np.asarray(pod_basis, dtype=np.float64)[:rank, :]

    # Gappy POD: solve min_c ‖ M @ (mean + basis.T @ c) - M @ u ‖²
    # → (M @ basis.T) @ c = M @ (u - mean)
    M_basis_T = basis[:, obs_indices].T  # (n_obs, r)
    rhs = u_flat[obs_indices] - mean[obs_indices]  # (n_obs,)

    # Least squares with small regularization for under-determined cases
    if M_basis_T.shape[0] < M_basis_T.shape[1]:
        # Under-determined: use minimum-norm solution
        coeff, _, _, _ = np.linalg.lstsq(
            M_basis_T.T @ M_basis_T + 1e-8 * np.eye(rank),
            M_basis_T.T @ rhs,
            rcond=None,
        )
    else:
        coeff, _, _, _ = np.linalg.lstsq(M_basis_T, rhs, rcond=None)

    recon_flat = mean + basis.T @ coeff
    return recon_flat.reshape(h, w).astype(np.float32)


def run_pod_ls(
    target_field: np.ndarray,
    mask_hw: np.ndarray,
    pod_basis: np.ndarray,
    pod_mean: np.ndarray,
    rank: int = 128,
) -> np.ndarray:
    """POD-LS: project observed values onto POD basis via least squares.

    A variant of Gappy POD that uses regularization suited for
    well-posed systems.

    Args:
        target_field: (H, W) ground truth field.
        mask_hw: (H, W) boolean observation mask.
        pod_basis: (r_max, D) POD spatial basis.
        pod_mean: (D,) ensemble mean.
        rank: Truncation rank.

    Returns:
        (H, W) reconstructed field.
    """
    h, w = target_field.shape
    mask_flat = np.asarray(mask_hw, dtype=bool).ravel()
    obs_indices = np.flatnonzero(mask_flat)

    u_flat = np.asarray(target_field, dtype=np.float64).ravel()
    mean = np.asarray(pod_mean, dtype=np.float64).ravel()
    basis = np.asarray(pod_basis, dtype=np.float64)[:rank, :]

    M_basis_T = basis[:, obs_indices].T  # (n_obs, r)
    rhs = u_flat[obs_indices] - mean[obs_indices]

    # Ridge-regularized least squares
    A = M_basis_T.T @ M_basis_T + 1e-6 * np.eye(rank)
    b = M_basis_T.T @ rhs
    coeff = np.linalg.solve(A, b)

    recon_flat = mean + basis.T @ coeff
    return recon_flat.reshape(h, w).astype(np.float32)


def compute_baseline_comparison(
    dataset_name: str,
    output_dir: str,
    ranks: list[int] | None = None,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Compute full baseline comparison for a dataset.

    Args:
        dataset_name: 'nc', 'rdb_h5', or 'sst_weekly'.
        output_dir: Output directory for results.
        ranks: POD truncation ranks to evaluate. Default: [16, 32, 64, 128].
        wavelet, level, mode: Wavelet parameters.

    Returns:
        Comparison results dict.
    """
    if ranks is None:
        ranks = [16, 32, 64, 128]

    from pathlib import Path
    from luna.data.registry import get_dataset
    from luna.data.io import load_npz, load_npy

    ds = get_dataset(dataset_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pod = load_npz(str(ds.pod_bundle_path))
    basis = np.asarray(pod["basis"], dtype=np.float64)
    mean = np.asarray(pod["mean_flat"], dtype=np.float64).ravel()

    data = load_npy(str(ds.data_array))
    # Use a small subset for baseline evaluation
    test_samples = data[:200, :, :, 0].astype(np.float64)  # (200, H, W)

    results: dict[str, Any] = {"dataset": dataset_name, "ranks": ranks, "methods": {}}

    for method_name, method_fn in [("gappy_pod", run_gappy_pod), ("pod_ls", run_pod_ls)]:
        for r in ranks:
            # Load mask for this dataset
            # ... mask loading logic would go here
            pass

    return results

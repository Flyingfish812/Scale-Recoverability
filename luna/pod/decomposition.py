"""
POD (Proper Orthogonal Decomposition) core operations.

Provides SVD-based POD decomposition, projection, and reconstruction.
"""

from __future__ import annotations

import numpy as np


def compute_pod(
    fields: np.ndarray,
    rank: int | None = None,
    energy_threshold: float | None = None,
    center: bool = True,
) -> dict[str, np.ndarray]:
    """Compute POD decomposition of a field ensemble.

    Args:
        fields: Array of shape (N, H, W) or (N, D).
        rank: Number of modes to retain. If None, determined by energy_threshold.
        energy_threshold: Cumulative energy ratio (e.g. 0.999).
        center: If True, subtract the ensemble mean.

    Returns:
        Dict with keys:
            'mean': (D,) mean field
            'basis': (r, D) spatial modes (U)
            'coefficients': (N, r) temporal coefficients (A)
            'singular_values': (min(N,D),) all singular values
            'energy_ratio': (min(N,D),) per-mode energy fraction
            'cumulative_energy': (min(N,D),) cumulative energy
    """
    # Flatten spatial dims
    if fields.ndim == 3:
        N, H, W = fields.shape
        X = fields.reshape(N, -1).astype(np.float64)
    else:
        X = np.asarray(fields, dtype=np.float64)
        N = X.shape[0]

    # Center
    mean = X.mean(axis=0, keepdims=True) if center else np.zeros((1, X.shape[1]))
    X_centered = X - mean

    # SVD: X_centered = U @ S @ Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Energy
    S2 = S ** 2
    total_energy = S2.sum()
    energy_ratio = S2 / total_energy if total_energy > 0 else np.zeros_like(S2)
    cumulative_energy = np.cumsum(energy_ratio)

    # Determine rank
    if rank is None and energy_threshold is not None:
        rank = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
    if rank is None:
        rank = len(S)
    rank = min(rank, len(S))

    # Truncate
    basis = Vt[:rank, :]  # (r, D)
    # Coefficients from truncated basis: A = X_centered @ Vt[:r].T = U[:,:r] @ diag(S[:r])
    coefficients = (U[:, :rank] * S[None, :rank]).astype(np.float64)

    return {
        "mean": mean.ravel().astype(np.float64),
        "basis": basis.astype(np.float64),
        "coefficients": coefficients.astype(np.float64),
        "singular_values": S.astype(np.float64),
        "energy_ratio": energy_ratio.astype(np.float64),
        "cumulative_energy": cumulative_energy.astype(np.float64),
    }


def project_to_pod(
    field: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Project a field onto the POD basis to get coefficients.

    Args:
        field: (D,) or (H, W) field.
        basis: (r, D) POD basis.
        mean: (D,) ensemble mean.

    Returns:
        (r,) POD coefficients.
    """
    f = np.asarray(field, dtype=np.float64).ravel()
    m = np.asarray(mean, dtype=np.float64).ravel() if mean is not None else np.zeros_like(f)
    return basis @ (f - m)


def reconstruct_from_pod(
    coefficients: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reconstruct a field from POD coefficients.

    Args:
        coefficients: (r,) POD coefficients.
        basis: (r, D) POD basis.
        mean: (D,) ensemble mean.
        spatial_shape: If provided, reshape output to (H, W).

    Returns:
        Reconstructed field (D,) or (H, W).
    """
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    m = np.asarray(mean, dtype=np.float64).ravel() if mean is not None else np.zeros(basis.shape[1])
    recon = basis.T @ c + m
    if spatial_shape is not None:
        return recon.reshape(spatial_shape)
    return recon


def compute_cumulative_energy(singular_values: np.ndarray) -> np.ndarray:
    """Compute cumulative energy ratio from singular values."""
    S2 = np.asarray(singular_values, dtype=np.float64) ** 2
    return np.cumsum(S2) / S2.sum()


def find_rank_for_energy(
    singular_values: np.ndarray,
    energy_threshold: float = 0.999,
) -> int:
    """Find the minimum rank for a given cumulative energy threshold."""
    cum = compute_cumulative_energy(singular_values)
    return int(np.searchsorted(cum, energy_threshold) + 1)

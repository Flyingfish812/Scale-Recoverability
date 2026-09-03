"""
POD Oracle reconstruction — theoretical lower bound for any POD-based method.

Given a POD basis and a target field, the oracle reconstructs the field
by projecting onto the truncated basis. This gives the best possible
reconstruction achievable with the given rank.
"""

from __future__ import annotations

import numpy as np

from luna.pod.decomposition import project_to_pod, reconstruct_from_pod


def pod_oracle_reconstruct(
    field: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Oracle reconstruction: project field onto POD basis, then reconstruct.

    This is the theoretical lower bound — no model can do better than this
    for a given POD basis and rank.

    Args:
        field: (H, W) or (D,) ground truth field.
        basis: (r, D) POD spatial basis.
        mean: (D,) ensemble mean.
        spatial_shape: (H, W) for output reshaping.

    Returns:
        Oracle-reconstructed field.
    """
    coeff = project_to_pod(field, basis, mean)
    return reconstruct_from_pod(coeff, basis, mean, spatial_shape)


def pod_oracle_batch(
    fields: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Batch oracle reconstruction for multiple fields.

    Args:
        fields: (N, H, W) or (N, D) array of fields.
        basis: (r, D) POD basis.
        mean: (D,) mean.
        spatial_shape: (H, W).

    Returns:
        (N, H, W) or (N, D) oracle reconstructions.
    """
    N = fields.shape[0]
    if fields.ndim == 3:
        D = fields.shape[1] * fields.shape[2]
        flat = fields.reshape(N, -1).astype(np.float64)
    else:
        D = fields.shape[1]
        flat = np.asarray(fields, dtype=np.float64)

    m = np.asarray(mean, dtype=np.float64).ravel() if mean is not None else np.zeros(D)
    b = np.asarray(basis, dtype=np.float64)

    # A = (X - mean) @ basis.T  → (N, r)
    coeffs = (flat - m[None, :]) @ b.T
    # X_oracle = mean + A @ basis
    recon_flat = m[None, :] + coeffs @ b

    if spatial_shape is not None:
        return recon_flat.reshape(N, *spatial_shape)
    return recon_flat

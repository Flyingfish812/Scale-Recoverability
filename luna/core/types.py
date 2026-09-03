"""
Shared type definitions (dataclasses) for the Luna project.

These types flow through all layers: public → feature → application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── Wavelet / band error types ─────────────────────────────────────

@dataclass
class BandErrorDict:
    """Per-band error metrics for a single sample.

    Each key is a band name (A4, W4, W3, W2, W1).
    Values are dicts with keys: E_total, E_trunc, E_pred, E_direct.
    """

    errors: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_array(self, band_order: list[str] | None = None) -> np.ndarray:
        """Extract E_total values as a numpy array in the given band order."""
        if band_order is None:
            from luna.core.constants import BANDS_COARSE_TO_FINE
            band_order = BANDS_COARSE_TO_FINE
        return np.array([self.errors[b].get("E_total", np.nan) for b in band_order])


# ── POD types ──────────────────────────────────────────────────────

@dataclass
class PodBundle:
    """Holds a complete POD decomposition bundle."""

    mean_flat: np.ndarray  # (1, D) flattened mean
    basis_flat: np.ndarray  # (rank, D) POD spatial basis
    coefficients: np.ndarray  # (N, rank) temporal coefficients
    singular_values: np.ndarray  # (rank0,) all singular values
    rank: int
    h: int
    w: int
    c: int = 1

    @property
    def hwc(self) -> int:
        return self.h * self.w * self.c


@dataclass
class ScaleResult:
    """Physical scale inference result for a single POD mode."""

    mode_index: int
    energy_ratio: float
    ell_x_med: float
    ell_y_med: float
    ell_x_prefix: float
    ell_y_prefix: float
    ell_x_tail: float
    ell_y_tail: float


# ── Experiment / dataset types ─────────────────────────────────────

@dataclass
class ExperimentCase:
    """Describes a single test case within a sweep."""

    model_name: str
    mask_num: int
    seed: int
    noise_sigma: float
    npz_path: Path
    test_name: str = ""


@dataclass
class DatasetInfo:
    """Metadata and artifact paths for a dataset."""

    name: str
    label: str
    data_array: Path
    vcnn_roots: dict[str, Path] = field(default_factory=dict)
    pod_sweep_root: Path | None = None
    pod_bundle_path: Path | None = None
    band_pod_bundle_path: Path | None = None
    mask_dir: Path | None = None
    mask_pattern: str = ""  # e.g. "cylinder2d_80x160_random_inc_n{num:03d}.csv"
    extra: dict[str, Any] = field(default_factory=dict)

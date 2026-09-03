"""
Global constants used across the Luna project.

All band ordering, wavelet parameters, and numerical tolerances
are defined here as the single source of truth.
"""

from __future__ import annotations

# ── Wavelet band ordering ──────────────────────────────────────────
# Coarse-to-fine: A4 (approximation, coarsest) → W1 (finest detail)
BANDS_COARSE_TO_FINE: list[str] = ["A4", "W4", "W3", "W2", "W1"]
# Fine-to-coarse: W1 (finest) → A4 (coarsest)
BANDS_FINE_TO_COARSE: list[str] = ["W1", "W2", "W3", "W4", "A4"]

# Short aliases for convenience
BANDS_CF = BANDS_COARSE_TO_FINE
BANDS_FC = BANDS_FINE_TO_COARSE
N_BANDS: int = 5

# All wavelet components including "Global"
WAVELET_COMPONENTS: list[str] = ["Global", "W1", "W2", "W3", "W4", "A4"]

# ── Default wavelet parameters ─────────────────────────────────────
DEFAULT_WAVELET: str = "db2"
DEFAULT_LEVEL: int = 4
DEFAULT_MODE: str = "periodization"

# ── Metric defaults ────────────────────────────────────────────────
TAU_DEFAULT: float = 0.05  # error threshold for band recoverability
EPS: float = 1e-12  # numerical stability

# ── Mask encoding ──────────────────────────────────────────────────
# Maps encoded mask IDs (p-values × 10000) to actual sensor counts
MASK_CODE_TO_POINTS: dict[int, int] = {8: 10, 12: 15, 16: 20, 23: 30, 39: 50}
MASK_POINTS_TO_CODE: dict[int, int] = {v: k for k, v in MASK_CODE_TO_POINTS.items()}

# Standard mask sizes and noise levels used across experiments
STANDARD_MASK_NUMS: list[int] = [10, 15, 20, 30, 50]
STANDARD_NOISE_SIGMAS: list[float] = [0.0, 0.001, 0.01, 0.1]

"""
Mask generation and manipulation utilities.

Extracted from training/masks.py — pure functions with no training dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def resolve_num_observations(
    num_points: int,
    mask_rate: float | None,
    mask_num: int | None,
) -> int:
    """Determine the actual number of sensor observations."""
    if mask_num is not None:
        value = int(mask_num)
        if value <= 0:
            raise ValueError(f"mask_num must be positive, got {mask_num}")
        return min(value, int(num_points))
    if mask_rate is None:
        raise ValueError("Either mask_rate or mask_num must be provided")
    if not (0.0 < float(mask_rate) <= 1.0):
        raise ValueError(f"mask_rate must be in (0, 1], got {mask_rate}")
    return max(1, int(round(float(mask_rate) * float(num_points))))


def generate_random_mask_hw(
    height: int,
    width: int,
    *,
    mask_rate: float | None = None,
    mask_num: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a random boolean mask of shape (H, W)."""
    count = resolve_num_observations(height * width, mask_rate, mask_num)
    rng = np.random.RandomState(None if seed is None else int(seed))
    flat = np.zeros(int(height * width), dtype=bool)
    indices = rng.choice(flat.shape[0], size=count, replace=False)
    flat[indices] = True
    return flat.reshape(int(height), int(width))


def generate_grid_mask_hw(
    height: int,
    width: int,
    *,
    mask_rate: float | None = None,
    mask_num: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a regular-grid boolean mask of shape (H, W)."""
    count = resolve_num_observations(height * width, mask_rate, mask_num)
    if count >= int(height * width):
        return np.ones((int(height), int(width)), dtype=bool)

    area_per_point = float(height * width) / float(count)
    step = max(1, int(round(np.sqrt(area_per_point))))

    yy = np.arange(0, int(height), step, dtype=np.int64)
    xx = np.arange(0, int(width), step, dtype=np.int64)

    rng = np.random.RandomState(None if seed is None else int(seed))
    off_y = 0 if step <= 1 else int(rng.randint(0, step))
    off_x = 0 if step <= 1 else int(rng.randint(0, step))
    yy = np.clip(yy + off_y, 0, int(height) - 1)
    xx = np.clip(xx + off_x, 0, int(width) - 1)

    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[np.ix_(yy, xx)] = True

    current = int(mask.sum())
    if current > count:
        keep_flat = np.flatnonzero(mask.ravel())
        choose = rng.choice(keep_flat, size=count, replace=False)
        trimmed = np.zeros(int(height * width), dtype=bool)
        trimmed[choose] = True
        return trimmed.reshape(int(height), int(width))
    if current < count:
        flat = mask.ravel().copy()
        extra = count - current
        zeros = np.flatnonzero(~flat)
        fill = rng.choice(zeros, size=extra, replace=False)
        flat[fill] = True
        return flat.reshape(int(height), int(width))
    return mask


def load_mask_csv(
    mask_path: str | Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Load a row,col CSV mask into a boolean (H,W) array."""
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    rows: list[int] = []
    cols: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Mask file is empty: {path}")
        names = {str(value).strip().lower() for value in reader.fieldnames}
        if not {"row", "col"}.issubset(names):
            raise ValueError(f"Mask file must contain 'row,col' columns: {path}")
        for line_idx, record in enumerate(reader, start=2):
            row = int(record["row"])
            col = int(record["col"])
            if not (0 <= row < int(height) and 0 <= col < int(width)):
                raise ValueError(
                    f"Mask point out of bounds at line {line_idx}: "
                    f"(row={row}, col={col}), shape={(height, width)}"
                )
            rows.append(row)
            cols.append(col)

    mask = np.zeros((int(height), int(width)), dtype=bool)
    if rows:
        mask[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = True
    if int(mask.sum()) <= 0:
        raise ValueError(f"Mask file contains no valid sampled points: {path}")
    return mask


def build_nearest_seed_index(mask_hw: np.ndarray) -> np.ndarray:
    """Build a nearest-neighbor index map for Voronoi interpolation.

    For each pixel, returns the index of the closest observed point.
    """
    mask = np.asarray(mask_hw, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("mask_hw must contain at least one observed point")
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    obs_y = coords[:, 0].astype(np.float32)
    obs_x = coords[:, 1].astype(np.float32)
    dist2 = (
        (yy[:, :, None] - obs_y[None, None, :]) ** 2
        + (xx[:, :, None] - obs_x[None, None, :]) ** 2
    )
    return np.argmin(dist2, axis=2).astype(np.int32)


def add_gaussian_noise(
    values: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add Gaussian noise to an array. Returns a copy."""
    sigma_value = float(sigma)
    array = np.asarray(values, dtype=np.float32)
    if sigma_value <= 0.0:
        return array.astype(np.float32, copy=True)
    rng = np.random.RandomState(None if seed is None else int(seed))
    return (array + rng.normal(loc=0.0, scale=sigma_value, size=array.shape)).astype(np.float32)


def build_voronoi_feature(
    field_hwc: np.ndarray,
    mask_hw: np.ndarray,
    nearest_index_hw: np.ndarray,
) -> np.ndarray:
    """Build a Voronoi-interpolated feature map (C, H, W) from sparse obs."""
    field = np.asarray(field_hwc, dtype=np.float32)
    mask = np.asarray(mask_hw, dtype=bool)
    observed_values = field[mask]
    filled = observed_values[nearest_index_hw]
    return np.transpose(filled, (2, 0, 1)).astype(np.float32, copy=False)

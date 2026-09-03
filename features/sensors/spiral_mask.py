"""
Radial-spiral sensor mask generator.

Ported verbatim from Ena `backend/sampling/masks.py`
(generate_radial_spiral_mask_hw + _resolve_num_obs) so that
`tools/prepare_incremental_masks.py` runs standalone inside Luna.

Only numpy is required; this module is self-contained.
"""

from __future__ import annotations

import numpy as np


def _resolve_num_obs(num_points: int, mask_rate: float | None, mask_num: int | None) -> int:
    if mask_num is not None:
        num_obs = int(mask_num)
        if num_obs <= 0:
            raise ValueError(f"mask_num must be positive, got {mask_num}")
        return min(num_obs, num_points)

    if mask_rate is None:
        raise ValueError("Either mask_rate or mask_num must be provided.")
    if not (0 < mask_rate <= 1.0):
        raise ValueError(f"mask_rate must be in (0,1], got {mask_rate}")
    return max(1, int(round(num_points * mask_rate)))


def generate_radial_spiral_mask_hw(
    H: int,
    W: int,
    mask_rate: float | None = None,
    seed: int | None = None,
    mask_num: int | None = None,
    *,
    max_radius_frac: float = 0.875,
) -> np.ndarray:
    """
    在 H×W 网格上生成“径向螺旋式”观测 mask。

    规则：
    - 采样半径从中心开始均匀增加，最大到 max_radius_frac * min(H, W) / 2
    - 每个半径对应一个独立的随机角度，使采样点不会全部落在同一射线上
    - 若离散化后发生重复点，则在相近半径处回退到最近的未使用网格点
    """
    num_points = H * W
    num_obs = _resolve_num_obs(num_points, mask_rate, mask_num)

    if not (0.0 <= float(max_radius_frac) <= 1.0):
        raise ValueError(f"max_radius_frac must be in [0,1], got {max_radius_frac}")

    rng = np.random.RandomState(seed)
    cy = 0.5 * (H - 1)
    cx = 0.5 * (W - 1)
    radius_limit = float(min(H, W)) * 0.5 * float(max_radius_frac)
    radii = np.linspace(0.0, radius_limit, num_obs, endpoint=True, dtype=np.float64)

    used: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []

    yy, xx = np.indices((H, W), dtype=np.float64)
    radial_dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    for radius in radii:
        point: tuple[int, int] | None = None
        for _ in range(64):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            row = int(round(cy + radius * np.sin(theta)))
            col = int(round(cx + radius * np.cos(theta)))
            row = int(np.clip(row, 0, H - 1))
            col = int(np.clip(col, 0, W - 1))
            candidate = (row, col)
            if candidate not in used:
                point = candidate
                break

        if point is None:
            free_mask = np.ones((H, W), dtype=bool)
            for row, col in used:
                free_mask[row, col] = False

            free_rows, free_cols = np.where(free_mask)
            if free_rows.size == 0:
                break

            free_dist = radial_dist[free_rows, free_cols]
            order = np.argsort(np.abs(free_dist - radius), kind="stable")
            best_idx = int(order[0])
            point = (int(free_rows[best_idx]), int(free_cols[best_idx]))

        used.add(point)
        coords.append(point)

    if len(coords) != num_obs:
        raise RuntimeError(f"Failed to place {num_obs} unique spiral samples on grid {H}x{W}; got {len(coords)}")

    mask = np.zeros((H, W), dtype=bool)
    for row, col in coords:
        mask[row, col] = True
    return mask

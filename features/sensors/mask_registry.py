"""
Mask-family registry (supplementary analysis).

Loads the 5 independent strictly-nested mask families from `masks_families/`.
Supplementary analysis/experiment scripts MUST load masks through this module —
never hard-code `masks2/` (which is the frozen P0 authoritative source).

Canonical seeds:
    family_01 = 20260522  (identical to masks2/, frozen)
    family_02 = 20260806
    family_03 = 20260807
    family_04 = 20260811  (disjoint from 01/02/03)
    family_05 = 20260951  (disjoint from 01/02/03/04)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

MASK_COUNTS: tuple[int, ...] = (10, 15, 20, 30, 50)
NC_PREFIX = "cylinder2d_80x160_random_inc"
NC_GRID = (80, 160)

FAMILY_SEEDS: Dict[str, int] = {
    "family_01": 20260522,
    "family_02": 20260806,
    "family_03": 20260807,
    "family_04": 20260811,
    "family_05": 20260951,
}

DEFAULT_FAMILIES_ROOT = _REPO_ROOT / "masks_families"


def families_root() -> Path:
    return DEFAULT_FAMILIES_ROOT


def list_families() -> List[str]:
    """Return family ids sorted, e.g. ['family_01', ..., 'family_05']."""
    root = families_root()
    fams = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("family_")
    )
    return fams


def iter_families() -> Iterator[str]:
    yield from list_families()


def _family_dir(family_id: str, root: Path | None = None) -> Path:
    base = root if root is not None else families_root()
    d = base / family_id
    if not d.is_dir():
        raise FileNotFoundError(f"Mask family {family_id!r} not found under {base}")
    return d


def _nc_csv_path(family_id: str, M: int, root: Path | None = None) -> Path:
    if M not in MASK_COUNTS:
        raise ValueError(f"M must be one of {MASK_COUNTS}, got {M}")
    return _family_dir(family_id, root) / "masks" / f"{NC_PREFIX}_n{M:03d}.csv"


def load_nc_mask(family_id: str, M: int, root: Path | None = None) -> np.ndarray:
    """Load (M, 2) int array of (row, col) sensor coordinates."""
    path = _nc_csv_path(family_id, M, root)
    coords = np.loadtxt(str(path), delimiter=",", skiprows=1).astype(np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != M:
        raise ValueError(f"Unexpected mask shape for {family_id} M={M}: {coords.shape}")
    return coords


def load_nc_mask_bool(family_id: str, M: int, root: Path | None = None) -> np.ndarray:
    """Return an (H, W) bool mask on the NC grid."""
    H, W = NC_GRID
    coords = load_nc_mask(family_id, M, root)
    m = np.zeros((H, W), dtype=bool)
    m[coords[:, 0], coords[:, 1]] = True
    return m


def load_family(family_id: str, root: Path | None = None) -> Dict[int, np.ndarray]:
    """Load all M masks of a family as {M: (M, 2) coords}."""
    return {M: load_nc_mask(family_id, M, root) for M in MASK_COUNTS}


def family_seed(family_id: str) -> int:
    try:
        return FAMILY_SEEDS[family_id]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"Unknown family {family_id!r}") from exc


def verify_family(family_id: str, root: Path | None = None) -> Dict[str, bool]:
    """Quick strict-nesting + uniqueness check (lighter than tools/validate)."""
    coords = load_family(family_id, root)
    checks: Dict[str, bool] = {}
    flat = {M: set(map(tuple, c.tolist())) for M, c in coords.items()}
    checks["all_unique"] = all(len(s) == M for M, s in flat.items())
    prev: set | None = None
    nested = True
    for M in MASK_COUNTS:
        if prev is not None and not prev.issubset(flat[M]):
            nested = False
        prev = flat[M]
    checks["strict_nested"] = nested
    return checks

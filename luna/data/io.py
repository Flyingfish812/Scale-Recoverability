"""
Unified I/O utilities for NPZ, NPY, JSON, and CSV files.

All paths are resolved relative to the project root.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# Project root: luna/data/io.py → luna/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def load_npy(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a .npy file."""
    mmap_mode = "r" if mmap else None
    return np.load(resolve_project_path(path), mmap_mode=mmap_mode)


def load_npz(path: str | Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    """Load a .npz file, returning a dict of arrays."""
    with np.load(resolve_project_path(path), allow_pickle=allow_pickle) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays to a .npz file."""
    p = resolve_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **arrays)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Save data to a JSON file."""
    p = resolve_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_csv_dict(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dicts (DictReader)."""
    with open(resolve_project_path(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

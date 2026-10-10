"""Strictly nested sensor masks for the three datasets (10/15/20/30/50 points).

One random draw of 50 candidate points fixes a mask family, and the mask for M sensors takes the first M points of that draw, so the five masks of a family nest. The cylinder candidates exclude the cylinder section, where the field is identically zero, and the SST candidates exclude missing values.

Datasets and strategies:
    cylinder2d (80x160)  random sampling outside the cylinder section
    sst_weekly (180x360) random sampling with an avoid-missing-value constraint
    rdb_h5 (128x128)     radial incremental sampling, evenly spread radii among the first ten points

Inputs
    data/sst_weekly.npy      weekly SST field, read only to build the avoid-missing-value candidate set

Output
    ------
    <out-dir>/masks/*.csv    15 mask files (header `row,col`)
    <out-dir>/plots/*.png    one scatter per dataset, with the five increment groups
    <out-dir>/manifest.json  seeds, mask counts and the nesting checks

Usage
    -----
    python -m features.sensors.incremental_masks --out-dir masks_families/family_02 --seed 20260806
    python -m features.sensors.incremental_masks --out-dir masks2 --seed 20260522 --exclude-cylinder-body

The mask CSV files are generated locally and do not belong to the repository; the family seeds the paper uses are listed in `features.sensors.mask_registry`.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .spiral_mask import generate_radial_spiral_mask_hw

REPO_ROOT = Path(__file__).resolve().parents[2]

MASK_COUNTS: Tuple[int, ...] = (10, 15, 20, 30, 50)
TOTAL_POINTS = 50

#: Cylinder section of the NC grid, in grid units: the field is identically zero
#: inside this circle, so it carries no observation. Candidate points exclude it.
CYLINDER_CENTER_RC: Tuple[float, float] = (39.5, 39.7)
CYLINDER_RADIUS: float = 5.0


def _default_out_dir(root: Path) -> Path:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "artifacts" / f"incremental_masks_pack_{tag}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict) -> None:
    import json

    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_mask_csv(path: Path, coords_rc: np.ndarray) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col"])
        for r, c in coords_rc:
            writer.writerow([int(r), int(c)])


def _coords_to_flat(coords_rc: np.ndarray, W: int) -> np.ndarray:
    return coords_rc[:, 0].astype(np.int64) * int(W) + coords_rc[:, 1].astype(np.int64)


def _flat_to_coords(flat_idx: np.ndarray, W: int) -> np.ndarray:
    flat = np.asarray(flat_idx, dtype=np.int64).reshape(-1)
    rows = flat // int(W)
    cols = flat % int(W)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def _sample_incremental_random_coords(*, H: int, W: int, candidates_flat: np.ndarray, seed: int) -> np.ndarray:
    cand = np.unique(np.asarray(candidates_flat, dtype=np.int64).reshape(-1))
    if cand.size < TOTAL_POINTS:
        raise ValueError(f"Not enough candidates for {H}x{W}: need {TOTAL_POINTS}, got {cand.size}")
    rng = np.random.RandomState(int(seed))
    picked = rng.choice(cand, size=TOTAL_POINTS, replace=False)
    return _flat_to_coords(picked, W)


def _pick_even_by_radius(radius: np.ndarray, candidate_idx: np.ndarray, num_pick: int, r_max: float) -> np.ndarray:
    """Pick indices whose radii are close to evenly spaced targets on [0, r_max]."""
    if num_pick <= 0:
        return np.empty((0,), dtype=np.int64)
    if candidate_idx.size < num_pick:
        raise ValueError(f"Not enough candidates to pick {num_pick}, got {candidate_idx.size}")

    targets = np.linspace(0.0, float(r_max), int(num_pick), endpoint=True, dtype=np.float64)
    remaining = set(int(i) for i in candidate_idx.tolist())
    out: List[int] = []
    for t in targets:
        best = min(remaining, key=lambda i: abs(float(radius[i]) - float(t)))
        out.append(int(best))
        remaining.remove(int(best))
    return np.asarray(out, dtype=np.int64)


def _pick_fill_largest_radius_gaps(
    radius: np.ndarray,
    selected_idx: List[int],
    remaining_idx: set[int],
    num_add: int,
    r_max: float,
) -> List[int]:
    """Iteratively add points near the midpoint of the largest current radius gap."""
    if num_add <= 0:
        return []
    if len(remaining_idx) < num_add:
        raise ValueError(f"Not enough remaining points to add {num_add}.")

    added: List[int] = []
    for _ in range(int(num_add)):
        sel_r = sorted(float(radius[i]) for i in selected_idx)
        anchors = [0.0] + sel_r + [float(r_max)]
        gaps = [(anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
        lo, hi = max(gaps, key=lambda p: float(p[1] - p[0]))
        mid = 0.5 * (float(lo) + float(hi))

        best = min(remaining_idx, key=lambda i: abs(float(radius[i]) - mid))
        selected_idx.append(int(best))
        remaining_idx.remove(int(best))
        added.append(int(best))
    return added


def _sample_incremental_radial_coords(*, H: int, W: int, seed: int, max_radius_frac: float = 0.875) -> np.ndarray:
    if not (0.0 < float(max_radius_frac) <= 1.0):
        raise ValueError(f"max_radius_frac must be in (0,1], got {max_radius_frac}")

    cy = 0.5 * (H - 1)
    cx = 0.5 * (W - 1)
    r_max = float(min(H, W)) * 0.5 * float(max_radius_frac)

    # Step-1: build a 50-point radial spiral layout (aligned with project's existing generator behavior).
    mask50 = generate_radial_spiral_mask_hw(
        H,
        W,
        mask_num=TOTAL_POINTS,
        seed=int(seed),
        max_radius_frac=float(max_radius_frac),
    )
    rows, cols = np.where(mask50)
    coords50 = np.stack([rows, cols], axis=1).astype(np.int64)
    if coords50.shape != (TOTAL_POINTS, 2):
        raise RuntimeError(f"Expected {(TOTAL_POINTS, 2)} radial points, got {coords50.shape}")

    # Step-2: reorder to strict incremental sequence.
    # 10-point base: uniformly occupy radius range; then fill gaps for +5/+5/+10/+20.
    d = np.sqrt((coords50[:, 0] - cy) ** 2 + (coords50[:, 1] - cx) ** 2)
    all_idx = np.arange(TOTAL_POINTS, dtype=np.int64)
    base10 = _pick_even_by_radius(d, all_idx, num_pick=10, r_max=r_max)

    selected: List[int] = [int(i) for i in base10.tolist()]
    remaining: set[int] = set(int(i) for i in all_idx.tolist()) - set(selected)

    _pick_fill_largest_radius_gaps(d, selected, remaining, num_add=5, r_max=r_max)
    _pick_fill_largest_radius_gaps(d, selected, remaining, num_add=5, r_max=r_max)
    _pick_fill_largest_radius_gaps(d, selected, remaining, num_add=10, r_max=r_max)
    _pick_fill_largest_radius_gaps(d, selected, remaining, num_add=20, r_max=r_max)

    order = np.asarray(selected, dtype=np.int64)
    out = coords50[order]

    # Check first-10 radial diversity on actual selected points.
    d_out = np.sqrt((out[:, 0] - cy) ** 2 + (out[:, 1] - cx) ** 2)
    unique_first10 = np.unique(np.round(d_out[:10], 2)).size
    if unique_first10 < 8:
        raise RuntimeError(
            f"First-10 radial diversity too low after reordering (unique radii={unique_first10})."
        )

    return out


def _load_sst_finite_candidates(path: Path, key: str = "sst") -> np.ndarray:
    """Load finite SST positions from a 2D (time x space) array.

    Supports both Luna's native .npy and Ena's legacy .mat (h5py) formats.
    """
    if str(path).lower().endswith(".npy"):
        arr = np.asarray(np.load(str(path), mmap_mode="r"), dtype=np.float32)
    else:
        try:
            import h5py  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("h5py is required to read MAT v7.3 datasets") from exc

        with h5py.File(str(path), "r") as f:
            if key not in f:
                raise KeyError(f"Key {key!r} not found in {path}")
            dset = f[key]
            arr = np.asarray(dset[:, :], dtype=np.float32)

    # Normalize to (time, 64800): strip trailing channel dim, flatten spatial.
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0].reshape(arr.shape[0], -1)
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D SST array, got shape={arr.shape}")
    if arr.shape[0] == 180 * 360 and arr.shape[1] != 180 * 360:
        arr = arr.T
    if arr.shape[1] != 180 * 360:
        raise ValueError(f"Unexpected SST shape={arr.shape}; expected one dim=64800")

    finite_all = np.all(np.isfinite(arr), axis=0)
    candidates = np.flatnonzero(finite_all)
    if candidates.size < TOTAL_POINTS:
        raise ValueError(
            f"Not enough finite SST positions for avoid-NaN sampling: {candidates.size} < {TOTAL_POINTS}"
        )
    return candidates.astype(np.int64)


def _check_incremental(coords_rc: np.ndarray, W: int) -> Dict[str, bool]:
    flat = _coords_to_flat(coords_rc, W)
    checks: Dict[str, bool] = {}
    prev = set(flat[: MASK_COUNTS[0]].tolist())
    checks["10_subset"] = True
    for n in MASK_COUNTS[1:]:
        cur = set(flat[:n].tolist())
        checks[f"{n}_extends_prev"] = prev.issubset(cur) and (len(cur) - len(prev) == n - len(prev))
        prev = cur
    checks["all_unique_50"] = (np.unique(flat).size == TOTAL_POINTS)
    return checks


def _exclude_body_candidates(candidates_flat: np.ndarray, *, W: int) -> np.ndarray:
    """Drop candidate grid points that fall inside the cylinder section."""
    candidates_flat = np.asarray(candidates_flat, dtype=np.int64)
    rows, cols = np.divmod(candidates_flat, W)
    distance = np.hypot(rows - CYLINDER_CENTER_RC[0], cols - CYLINDER_CENTER_RC[1])
    return candidates_flat[distance > CYLINDER_RADIUS]


def _save_incremental_csvs(mask_dir: Path, *, prefix: str, coords50: np.ndarray) -> List[str]:
    files: List[str] = []
    for n in MASK_COUNTS:
        out = mask_dir / f"{prefix}_n{n:03d}.csv"
        _write_mask_csv(out, coords50[:n])
        files.append(out.name)
    return files


def _plot_incremental_scatter(
    path: Path,
    *,
    H: int,
    W: int,
    coords50: np.ndarray,
    title: str,
    equal_aspect: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(path.parent)

    groups = [
        (0, 10, "n=10", "#1f77b4"),
        (10, 15, "+5 -> n=15", "#ff7f0e"),
        (15, 20, "+5 -> n=20", "#2ca02c"),
        (20, 30, "+10 -> n=30", "#d62728"),
        (30, 50, "+20 -> n=50", "#9467bd"),
    ]

    fig_size = (6.4, 6.4) if bool(equal_aspect) else (9.0, 4.8)
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)
    for s, e, label, color in groups:
        part = coords50[s:e]
        ax.scatter(part[:, 1], part[:, 0], s=22, c=color, label=label, edgecolors="none", alpha=0.95)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)
    if bool(equal_aspect):
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw strictly nested sensor-mask families from a seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Default: artifacts/incremental_masks_pack_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=20260522, help="Base random seed")
    parser.add_argument(
        "--exclude-cylinder-body",
        action="store_true",
        help="Draw the cylinder2d candidates outside the cylinder section",
    )
    parser.add_argument(
        "--sst-data",
        type=Path,
        default=Path("data") / "sst_weekly.npy",
        help="Path to SST data (npy or h5py MAT; relative to repo root if not absolute)",
    )
    args = parser.parse_args()

    root = REPO_ROOT
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir(root)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    _ensure_dir(out_dir)

    sst_path = args.sst_data if args.sst_data.is_absolute() else (root / args.sst_data)

    mask_dir = out_dir / "masks"
    plot_dir = out_dir / "plots"
    _ensure_dir(mask_dir)
    _ensure_dir(plot_dir)

    # 1) cylinder2d (80x160): random incremental
    cyl_candidates = np.arange(80 * 160, dtype=np.int64)
    if bool(args.exclude_cylinder_body):
        cyl_candidates = _exclude_body_candidates(cyl_candidates, W=160)
    cyl_coords50 = _sample_incremental_random_coords(H=80, W=160, candidates_flat=cyl_candidates, seed=int(args.seed) + 11)
    cyl_files = _save_incremental_csvs(mask_dir, prefix="cylinder2d_80x160_random_inc", coords50=cyl_coords50)
    _plot_incremental_scatter(
        plot_dir / "cylinder2d_80x160_incremental_points.png",
        H=80,
        W=160,
        coords50=cyl_coords50,
        title="cylinder2d (80x160) incremental random mask points",
    )

    # 2) sst_weekly (180x360): avoid-NaN random incremental
    sst_candidates = _load_sst_finite_candidates(sst_path, key="sst")
    sst_coords50 = _sample_incremental_random_coords(H=180, W=360, candidates_flat=sst_candidates, seed=int(args.seed) + 22)
    sst_files = _save_incremental_csvs(mask_dir, prefix="sst_weekly_180x360_random_avoid_nan_inc", coords50=sst_coords50)
    _plot_incremental_scatter(
        plot_dir / "sst_weekly_180x360_incremental_points.png",
        H=180,
        W=360,
        coords50=sst_coords50,
        title="sst_weekly (180x360) incremental random avoid-NaN mask points",
    )

    # 3) rdb_h5 (128x128): radial incremental
    rdb_coords50 = _sample_incremental_radial_coords(H=128, W=128, seed=int(args.seed) + 33, max_radius_frac=1.0)
    rdb_files = _save_incremental_csvs(mask_dir, prefix="rdb_h5_128x128_radial_inc", coords50=rdb_coords50)
    _plot_incremental_scatter(
        plot_dir / "rdb_h5_128x128_incremental_points.png",
        H=128,
        W=128,
        coords50=rdb_coords50,
        title="rdb_h5 (128x128) incremental radial mask points",
        equal_aspect=True,
    )

    cyl_check = _check_incremental(cyl_coords50, W=160)
    sst_check = _check_incremental(sst_coords50, W=360)
    rdb_check = _check_incremental(rdb_coords50, W=128)

    # Extra check: SST avoid-NaN
    sst_flat = _coords_to_flat(sst_coords50, 360)
    sst_avoid_ok = np.isin(sst_flat, sst_candidates).all()

    # Extra check: RDB first-10 radii diversity
    cy, cx = 0.5 * (128 - 1), 0.5 * (128 - 1)
    d = np.sqrt((rdb_coords50[:, 0] - cy) ** 2 + (rdb_coords50[:, 1] - cx) ** 2)
    rdb_first10_unique_r = int(np.unique(np.round(d[:10], 2)).size)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "mask_counts": list(MASK_COUNTS),
        "output": {
            "mask_dir": str(mask_dir),
            "plot_dir": str(plot_dir),
        },
        "datasets": {
            "cylinder2d_80x160": {
                "strategy": "random_incremental",
                "files": cyl_files,
                "checks": cyl_check,
            },
            "sst_weekly_180x360": {
                "strategy": "random_incremental_avoid_nan",
                "files": sst_files,
                "checks": {**sst_check, "avoid_nan_all_points": bool(sst_avoid_ok)},
                "finite_candidate_count": int(sst_candidates.size),
            },
            "rdb_h5_128x128": {
                "strategy": "radial_incremental",
                "files": rdb_files,
                "checks": {
                    **rdb_check,
                    "first10_unique_radii_rounded_2dp": rdb_first10_unique_r,
                    "selection_method": "from_spiral50_then_even10_then_gap_fill",
                },
            },
        },
        "plots": [
            "plots/cylinder2d_80x160_incremental_points.png",
            "plots/sst_weekly_180x360_incremental_points.png",
            "plots/rdb_h5_128x128_incremental_points.png",
        ],
    }

    _write_json(out_dir / "manifest.json", manifest)

    all_files = cyl_files + sst_files + rdb_files
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] masks={len(all_files)} csv files")
    print(f"[OK] plots=3 png files")
    print(f"[OK] incremental checks: cylinder={all(cyl_check.values())}, sst={all(sst_check.values())}, rdb={all(rdb_check.values())}")
    print(f"[OK] sst avoid-nan check={bool(sst_avoid_ok)}")
    print(f"[OK] rdb first10 unique radii (rounded 2dp)={rdb_first10_unique_r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the POD bases that every reconstruction artifact is expressed in.

The estimators of the paper never see the fields directly: the training code reduces the snapshots to POD coefficients (``features.training.pod_sweep``) and the statistics layer reconstructs the truncated reference of the same rank (``applications.statistics.band_error_decomposition``). Both read the bundle written here, so the mean field, the basis, the coefficients and the rank have to be produced once, in one place: this step.

One bundle is written per dataset of the registry, holding the mean field, the rank-r basis, the coefficients of every snapshot, the energy of the retained modes and the prefix-reconstruction error of each snapshot. The keys are the ones the existing consumers read, and the decomposition is ``luna.pod.decomposition.compute_pod``, which is what the published bundles were built with.

Prerequisites
    data/ arrays verified by applications/pipelines/01_prepare_data.py

Rank
    nc 128, rdb_h5 128, sst_weekly 1024, the rank of the published bundles. The main sequence of the paper uses the rank-128 basis of the numerical wake;
    the sea-surface dataset keeps more modes because its energy spectrum is much flatter. ``--rank`` overrides the rank for every selected dataset, which changes the meaning of the artifacts downstream and is meant for inspection.

Outputs
    artifacts/pod_bases/<dataset>/pod_base_bundle.npz with the keys
        dataset_name, source_data_path, interpreted_layout, input_shape, interpreted_shape, mean_field, pod_basis, coefficients, singular_values, mode_energy_ratio, cumulative_energy_ratio, relative_rmse_prefix, relative_rmse_top_r

    A bundle that already exists is compared with the rebuild before anything is written. Identical values leave the file untouched; differing values are reported and the bundle on disk is kept, so a rebuild can never silently replace the basis the paper's numbers were computed from.

Pipeline
    previous  applications/pipelines/01_prepare_data.py this step writes the POD bases under artifacts/pod_bases/
    next      applications/pipelines/03_train_estimators.py

Usage
    python applications/pipelines/02_build_pod_bases.py
    python applications/pipelines/02_build_pod_bases.py --datasets nc
    python applications/pipelines/02_build_pod_bases.py --check
    python applications/pipelines/02_build_pod_bases.py --datasets nc --out-root /tmp/pod_bases

The sea-surface bundle is by far the most expensive one (rank 1024 on 1914 snapshots of 180 x 360 fields); ``--datasets nc`` rebuilds only the basis of the main sequence. ``--out-root`` writes somewhere else and is the way to compare a rebuild against the bundles in place.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.data.io import save_npz  # noqa: E402
from luna.data.registry import get_dataset  # noqa: E402
from luna.pod.decomposition import compute_pod  # noqa: E402

# : Datasets of the paper, in the order of scripts/download_data.sh.
DEFAULT_DATASETS = ("nc", "rdb_h5", "sst_weekly")

# : Rank of the published bundle of each dataset.
RANKS = {"nc": 128, "rdb_h5": 128, "sst_weekly": 1024}

DEFAULT_OUT_ROOT = ROOT / "artifacts" / "pod_bases"

# : Name of the bundle inside the directory of a dataset.
BUNDLE_NAME = "pod_base_bundle.npz"

# : Guard of the relative reconstruction error, as in the published bundles.
EPS = 1e-12


def relative(path: Path) -> str:
    """Path as it is reported to the user: relative to the repository when it is
    inside it, absolute otherwise."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sample_layout(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Snapshots as (N, H, W, C) plus the layout note stored in the bundle.

    The published bundles record the layout they were given, and that field is part of the bundle contract, so a rebuild applies the same convention: a four-dimensional array whose axis 1 is a small channel axis is read as NCHW and transposed, everything else is taken as it is and noted NHWC (four dimensions) or BNC (three). The three datasets of the paper are already in that layout, so they are used unchanged.
    """
    if array.ndim == 4:
        if array.shape[1] <= 4 and array.shape[-1] > 4:
            return np.transpose(array, (0, 2, 3, 1)), "NCHW->NHWC"
        return array, "NHWC"
    if array.ndim == 3:
        if array.shape[1] <= 4 and array.shape[2] > 4:
            return np.transpose(array, (0, 2, 1)), "BCN->BNC"
        return array, "BNC"
    raise ValueError(f"expected 3D or 4D snapshots (N, H, W[, C]), got {array.shape}")


def relative_rmse_prefix(coefficients: np.ndarray, target_norm_sq: np.ndarray) -> np.ndarray:
    """Relative reconstruction error of every snapshot as a function of the
    number of retained modes.

    With an orthonormal basis the error of the first r modes is the part of the snapshot energy the first r coefficients do not carry. The energy of a snapshot is reduced with ``np.sum``, the reduction the published bundles were written with: the difference between two summation orders is at the last bit of the total, but it survives the subtraction of two nearly equal energies, so keeping the reduction makes a rebuild bit-identical.
    """
    cumulated = np.cumsum(coefficients * coefficients, axis=1)
    residual = np.maximum(target_norm_sq[:, None] - cumulated, 0.0)
    return np.sqrt(residual / np.maximum(target_norm_sq[:, None], EPS))


def build_bundle(dataset: str, rank: int) -> dict[str, np.ndarray]:
    """Every array of the bundle of one dataset, laid out as in the published
    bundles; the caller writes them."""
    info = get_dataset(dataset)
    data_path = Path(info.data_array)
    raw = np.load(data_path)
    fields, layout = sample_layout(np.asarray(raw, dtype=np.float32))
    sample_shape = tuple(int(v) for v in fields.shape[1:])
    snapshots, dim = int(fields.shape[0]), int(np.prod(sample_shape, dtype=np.int64))

    print(f"   [{dataset}] SVD of ({snapshots}, {dim}) centered matrix, rank {rank}")
    pod = compute_pod(fields.reshape(snapshots, dim).astype(np.float64), rank=rank)
    used_rank = int(pod["basis"].shape[0])
    basis = pod["basis"].astype(np.float32)
    coefficients = pod["coefficients"].astype(np.float32)

    # The prefix-error diagnostics are read off the coefficients as they are stored, so that a rebuild reproduces the bundle field by field.
    centered = fields.reshape(snapshots, dim).astype(np.float64) - pod["mean"]
    target_norm_sq = np.sum(centered * centered, axis=1)
    prefix = relative_rmse_prefix(np.asarray(coefficients, dtype=np.float64), target_norm_sq)
    top_r = prefix[:, used_rank - 1] if used_rank > 0 else np.full(snapshots, np.nan)

    return {
        "dataset_name": np.asarray(data_path.name),
        "source_data_path": np.asarray(str(data_path)),
        "interpreted_layout": np.asarray(layout),
        "input_shape": np.asarray(raw.shape, dtype=np.int32),
        "interpreted_shape": np.asarray(fields.shape, dtype=np.int32),
        "mean_field": pod["mean"].reshape(sample_shape).astype(np.float32),
        "pod_basis": basis.reshape((used_rank, *sample_shape)),
        "coefficients": coefficients,
        "singular_values": pod["singular_values"][:used_rank].astype(np.float32),
        "mode_energy_ratio": pod["energy_ratio"][:used_rank].astype(np.float32),
        "cumulative_energy_ratio": pod["cumulative_energy"][:used_rank].astype(np.float32),
        "relative_rmse_prefix": prefix.astype(np.float32),
        "relative_rmse_top_r": top_r.astype(np.float32),
    }


def compare_bundles(path: Path, arrays: dict[str, np.ndarray]) -> list[str]:
    """Field-by-field differences between a bundle on disk and a rebuild.

    The comparison is exact. A bundle is regenerated from the same raw array with the same decomposition, so a faithful rebuild is bit-identical; a different LAPACK flips the sign of a singular vector, which this reports as a difference of the basis instead of hiding it behind a tolerance.
    """
    existing = np.load(path)
    present = set(existing.files)
    differences: list[str] = []
    for key in sorted(set(arrays) - present):
        differences.append(f"{key}: missing from the bundle")
    for key in sorted(present - set(arrays)):
        differences.append(f"{key}: present in the bundle, not in the rebuild")
    for key in sorted(present & set(arrays)):
        reference, value = existing[key], arrays[key]
        if tuple(reference.shape) != tuple(value.shape):
            differences.append(f"{key}: shape {reference.shape} on disk, {value.shape} in the rebuild")
        elif not np.array_equal(np.asarray(reference), np.asarray(value)):
            if reference.dtype.kind in "US" or value.dtype.kind in "US":
                differences.append(f"{key}: {reference.item()!r} on disk, {value.item()!r} in the rebuild")
            else:
                deviation = np.max(np.abs(np.asarray(reference, dtype=np.float64)
                                          - np.asarray(value, dtype=np.float64)))
                differences.append(f"{key}: values differ (largest deviation {deviation:.3e})")
        elif reference.dtype != value.dtype:
            differences.append(f"{key}: dtype {reference.dtype} on disk, {value.dtype} in the rebuild")
    existing.close()
    return differences


def check(datasets: list[str], out_root: Path) -> int:
    """Report the bundles that are in place and the rank of each."""
    failed: list[str] = []
    for dataset in datasets:
        path = out_root / Path(get_dataset(dataset).data_array).stem / BUNDLE_NAME
        if not path.exists():
            print(f"   [!!] {dataset:11s} {relative(path)} is missing")
            failed.append(dataset)
            continue
        with np.load(path) as bundle:
            coefficients = bundle["coefficients"]
            print(f"   [ok] {dataset:11s} {relative(path)}  "
                  f"rank {int(coefficients.shape[1])}  "
                  f"{int(coefficients.shape[0])} snapshots")
    if failed:
        print(f"\n[failed] {len(failed)}/{len(datasets)} bundles are missing; "
              "run this step without --check to build them")
        return 1
    print(f"\n[ok] {len(datasets)}/{len(datasets)} bundles in place")
    return 0


def build(datasets: list[str], out_root: Path, rank: int | None) -> int:
    """Rebuild the bundles and write the ones that are not in place."""
    failed: list[str] = []
    for dataset in datasets:
        target = out_root / Path(get_dataset(dataset).data_array).stem / BUNDLE_NAME
        started = time.time()
        arrays = build_bundle(dataset, rank or RANKS[dataset])
        if target.exists():
            differences = compare_bundles(target, arrays)
            if differences:
                print(f"   [!] {relative(target)} differs from the rebuild "
                      f"and was NOT overwritten")
                for line in differences[:10]:
                    print(f"       {line}")
                failed.append(dataset)
                continue
            print(f"   [=] {relative(target)} already matches the rebuild "
                  f"({time.time() - started:.1f}s, nothing written)")
            continue
        save_npz(target, **arrays)
        print(f"   [+] wrote {relative(target)} "
              f"(rank {int(arrays['coefficients'].shape[1])}, "
              f"{int(arrays['coefficients'].shape[0])} snapshots, "
              f"{time.time() - started:.1f}s)")
    if failed:
        print(f"\n[failed] {len(failed)}/{len(datasets)} bundles disagree with the files "
              "on disk and were kept as they are")
        return 1
    print(f"\n[ok] {len(datasets)}/{len(datasets)} bundles are up to date")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"datasets to build (default: {' '.join(DEFAULT_DATASETS)})")
    parser.add_argument("--check", action="store_true",
                        help="verify that the bundles exist and report their rank")
    parser.add_argument("--rank", type=int, default=None,
                        help="number of modes to retain for every selected dataset")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="directory the bundles are written to")
    args = parser.parse_args()

    datasets = args.datasets or list(DEFAULT_DATASETS)
    unknown = [name for name in datasets if name not in RANKS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; "
                         f"known: {', '.join(DEFAULT_DATASETS)}")

    if args.check:
        print("== checking the POD bases under "
              f"{relative(args.out_root)}/")
        return check(datasets, args.out_root)

    print(f"== building the POD bases under {relative(args.out_root)}/")
    return build(datasets, args.out_root, args.rank)


if __name__ == "__main__":
    raise SystemExit(main())

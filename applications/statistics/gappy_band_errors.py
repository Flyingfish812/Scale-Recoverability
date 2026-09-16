"""Band-wise error of the gappy-POD baseline.

Gappy POD reconstructs the field from the sensor observations alone,

    a_hat = (C_M Phi_r)^+ (y - C_M u_bar),

with the rank capped at the sensor count and selected on the validation split.
Comparing that reconstruction, band by band, with the rank-r POD truncation of
the same target field isolates the error the sparse sampling adds on top of the
representation floor of the basis:

    total(b)      = ‖W_b(u) − W_b(u_hat)‖₂ / ‖W_b(u)‖₂
    truncation(b) = ‖W_b(u) − W_b(u_ref)‖₂ / ‖W_b(u)‖₂
    delta(b)      = total(b) − truncation(b)

Both terms are evaluated on the streamwise component, as everywhere else in the
paper; the global error that identifies the reconstruction is evaluated on the
full two-component state. Fields are in physical units.

The rank is selected on the validation split, and its error ``val_ger`` pairs the
measurement-noise realisation with the validation snapshots in the order the
split returns them. The public split returns sorted snapshot indices, which is
one of the possible pairings; the selected rank itself is what the artifact and
the paper carry forward, and the test-side quantities do not depend on the
pairing at all.

Inputs
    data/cylinder2d_q1.npy                     raw snapshots
    artifacts/pod_bases/...                    rank-r POD basis and mean
    masks_families/family_01/masks/...         sensor sequence of the main runs
    artifacts/pod_model_sweep_nc/...           run defining the test snapshots
Output
    artifacts/statistics/gappy_band_errors.json   one row per (sensor count,
                                                  noise level), plus the shared
                                                  truncation reference and the
                                                  configuration the paper quotes

Usage
    python -m applications.statistics.gappy_band_errors
    python -m applications.statistics.gappy_band_errors --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.sensors.mask_registry import load_nc_mask_bool  # noqa: E402
from features.training.estimator_runs import run_path  # noqa: E402
from features.training.pod_sweep import split_like_p0_seed0  # noqa: E402
from luna.core.constants import EPS  # noqa: E402
from luna.data.io import load_npz  # noqa: E402
from luna.pod.truncation_reference import truncation_reference_reconstruct  # noqa: E402
from luna.wavelet.metrics import band_errors_all, global_error  # noqa: E402

DATA_ARRAY = ROOT / "data" / "cylinder2d_q1.npy"
POD_BUNDLE = ROOT / "artifacts" / "pod_bases" / "cylinder2d_q1" / "pod_base_bundle.npz"
OUTPUT = ROOT / "artifacts" / "statistics" / "gappy_band_errors.json"
#: Frozen artifact of the earlier analysis of the same quantity, read by --verify.
REFERENCE = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "s021_gappy_pod_wavelet.json")

#: Sensor sequence of the main experiments.
MASK_FAMILY = "family_01"
#: Ranks the validation split may choose from, capped at the sensor count.
CANDIDATE_RANKS = (4, 8, 12, 16, 20, 24, 32)
#: Seed of the measurement noise, which makes the closed-form estimator repeatable.
NOISE_SEED = 42
#: Configuration the paper quotes band by band.
REPRESENTATIVE_SENSORS, REPRESENTATIVE_SIGMA = 20, 0.0
#: Fields compared numerically by --verify.
BAND_FIELDS = ("oracle_errors", "total_errors", "delta")
SCALAR_FIELDS = ("selected_rank", "val_ger", "test_ger_mean", "n_test")


def observation_matrix(fields: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sensor readings of every snapshot, flattened exactly like the basis below."""
    selected = np.asarray(fields, dtype=np.float64)[:, rows, cols, :]
    return selected.reshape(len(fields), -1)


def truncation_errors(
    test_fields: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
    shape: tuple[int, ...],
    bands: list[str],
) -> dict[str, float]:
    """Mean band error of the rank-r POD truncation of every test snapshot.

    The truncation is the best a rank-r basis can represent, so it is the
    reference the reconstruction error is measured against. It depends on the
    snapshot and the rank only, not on the sensor count or the noise level, so
    it is computed once and shared by every row.
    """
    sums = {band: 0.0 for band in bands}
    for field in test_fields:
        reference = truncation_reference_reconstruct(field, basis, mean, spatial_shape=shape)
        errors = band_errors_all(field[:, :, 0], reference[:, :, 0])
        for band in bands:
            sums[band] += errors[band]
    return {band: sums[band] / len(test_fields) for band in bands}


def gappy_row(
    sensor_count: int,
    basis: np.ndarray,
    basis_flat: np.ndarray,
    mean: np.ndarray,
    shape: tuple[int, ...],
    rows: np.ndarray,
    cols: np.ndarray,
    mean_obs: np.ndarray,
    val_fields: np.ndarray,
    test_fields: np.ndarray,
    val_obs: np.ndarray,
    test_obs: np.ndarray,
    sigma: float,
    reference: dict[str, float],
    bands: list[str],
    candidates: list[int],
) -> dict:
    """One (sensor count, noise level) cell: rank selection, then the test errors.

    The rank is chosen on the validation split, which is the protocol of the
    reference runs; the same noise realisation is used for every candidate rank
    so that the comparison between ranks is not confounded by the noise.
    """
    rng = np.random.RandomState(NOISE_SEED)
    val_noisy = val_obs + rng.normal(0.0, sigma, val_obs.shape)
    test_noisy = test_obs + rng.normal(0.0, sigma, test_obs.shape)

    val_flat = val_fields.reshape(len(val_fields), -1)
    val_norm = np.linalg.norm(val_flat, axis=1) + EPS

    best_rank, best_error = candidates[0], float("inf")
    for rank in candidates:
        phi = basis[:rank][:, rows, cols, :].reshape(rank, -1).T
        coeffs = (val_noisy - mean_obs) @ np.linalg.pinv(phi).T
        recon = mean[None, :] + coeffs @ basis_flat[:rank]
        error = float(np.mean(np.linalg.norm(recon - val_flat, axis=1) / val_norm))
        if error < best_error:
            best_rank, best_error = rank, error

    phi = basis[:best_rank][:, rows, cols, :].reshape(best_rank, -1).T
    coeffs = (test_noisy - mean_obs) @ np.linalg.pinv(phi).T
    recon = (mean[None, :] + coeffs @ basis_flat[:best_rank]).reshape(-1, *shape)

    band_sums = {band: 0.0 for band in bands}
    ger_sum = 0.0
    for field, prediction in zip(test_fields, recon):
        errors = band_errors_all(field[:, :, 0], prediction[:, :, 0])
        for band in bands:
            band_sums[band] += errors[band]
        ger_sum += global_error(field, prediction)

    n_test = len(test_fields)
    total = {band: band_sums[band] / n_test for band in bands}
    return {
        "mask_num": int(sensor_count),
        "sigma": float(sigma),
        "selected_rank": int(best_rank),
        "val_ger": best_error,
        "test_ger_mean": ger_sum / n_test,
        "oracle_errors": reference,
        "total_errors": total,
        "delta": {band: total[band] - reference[band] for band in bands},
        "n_test": n_test,
    }


def representative_row(rows: list[dict]) -> dict | None:
    """The configuration the paper quotes, in the shape the table expects."""
    for row in rows:
        if row["mask_num"] == REPRESENTATIVE_SENSORS and (
            abs(row["sigma"] - REPRESENTATIVE_SIGMA) < 1e-12
        ):
            return {
                "config": f"M={REPRESENTATIVE_SENSORS}, σ={REPRESENTATIVE_SIGMA:g}",
                "selected_rank": row["selected_rank"],
                "GER": row["test_ger_mean"],
                "per_band_total_error": row["total_errors"],
                "delta_excess_error": row["delta"],
            }
    return None


def verify(report: dict, bands: list[str], tolerance: float) -> int:
    """Compare every number with the frozen artifact of the earlier analysis."""
    if not REFERENCE.exists():
        print(f"  [skip] {REFERENCE.relative_to(ROOT).as_posix()} is not available")
        return 0

    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
    new_rows = {(row["mask_num"], round(row["sigma"], 6)): row for row in report["results"]}
    old_rows = {(row["mask_num"], round(row["sigma"], 6)): row for row in frozen["results"]}

    compared = differences = 0
    worst = 0.0
    print(f"\ncomparison with {REFERENCE.relative_to(ROOT).as_posix()}"
          f" (tolerance {tolerance:g})")
    for key in sorted(old_rows):
        old, new = old_rows[key], new_rows.get(key)
        if new is None:
            raise SystemExit(f"configuration {key} is missing from the new artifact")
        for field in SCALAR_FIELDS:
            compared += 1
            gap = abs(float(new[field]) - float(old[field]))
            worst = max(worst, gap)
            if gap > tolerance:
                differences += 1
                print(f"  DIFF M={key[0]} sigma={key[1]} {field}: {new[field]} vs {old[field]}")
        for group in BAND_FIELDS:
            for band in bands:
                compared += 1
                gap = abs(float(new[group][band]) - float(old[group][band]))
                worst = max(worst, gap)
                if gap > tolerance:
                    differences += 1
                    print(f"  DIFF M={key[0]} sigma={key[1]} {group}.{band}: "
                          f"{new[group][band]:.12g} vs {old[group][band]:.12g}")
    print(f"  {compared} numbers compared, {differences} above {tolerance:g}, "
          f"largest difference {worst:.3e}")
    return 1 if differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="largest accepted difference in the --verify comparison")
    args = parser.parse_args()

    cfg = get_config()
    bands = list(cfg.bands)
    started = time.time()
    print("== gappy-POD band errors")

    bundle = load_npz(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)[: cfg.pod_rank]
    mean_field = np.asarray(bundle["mean_field"], dtype=np.float64)
    shape = mean_field.shape
    basis_flat = basis.reshape(basis.shape[0], -1)
    mean = mean_field.ravel()

    fields = np.load(str(DATA_ARRAY), mmap_mode="r")
    # The test snapshots are the ones the reference runs use; the validation
    # split is the one the closed-form estimators of the paper share with them.
    reference_run = run_path("mlp", cfg.M_values[0], cfg.sigma_values[0], cfg.seeds[0])
    if reference_run is None:
        raise SystemExit(f"the reference run that fixes the test split is missing: {reference_run}")
    split = split_like_p0_seed0(fields.shape[0], reference_run)
    val_fields = np.asarray(fields[split["val"]], dtype=np.float64)
    test_fields = np.asarray(fields[split["test"]], dtype=np.float64)
    print(f"   {fields.shape[0]} snapshots, rank-{cfg.pod_rank} basis, "
          f"split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    reference = truncation_errors(test_fields, basis_flat, mean, shape, bands)
    print("   rank-{} truncation, mean band error: {}".format(
        cfg.pod_rank, ", ".join(f"{band}={reference[band]:.6f}" for band in bands)))

    results = []
    for sensor_count in cfg.M_values:
        mask = load_nc_mask_bool(MASK_FAMILY, sensor_count)
        indices = np.argwhere(mask)
        rows, cols = indices[:, 0], indices[:, 1]
        candidates = [rank for rank in CANDIDATE_RANKS if rank <= len(indices)]
        if not candidates:
            candidates = [len(indices)]
        val_obs = observation_matrix(val_fields, rows, cols)
        test_obs = observation_matrix(test_fields, rows, cols)
        mean_obs = mean_field[rows, cols, :].ravel()
        for sigma in cfg.sigma_values:
            row = gappy_row(
                sensor_count, basis, basis_flat, mean, shape, rows, cols, mean_obs,
                val_fields, test_fields, val_obs, test_obs, float(sigma),
                reference, bands, candidates,
            )
            results.append(row)
            excess = ", ".join(f"{band}={row['delta'][band]:+.4f}" for band in bands)
            print(f"   M={sensor_count:>2} sigma={sigma:<5} rank={row['selected_rank']:>2} "
                  f"GER={row['test_ger_mean']:.6f}  delta {excess}")
        print(f"   M={sensor_count}: done ({time.time() - started:.0f} s)")

    report = {
        "task": "gappy_band_errors",
        "description": ("Band-wise error of the gappy-POD baseline against the rank-r "
                        "POD truncation of the same snapshots"),
        "method": ("a_hat = pinv(C_M Phi_r) (y - C_M u_bar), rank in "
                   f"{CANDIDATE_RANKS} capped at the sensor count and selected on "
                   "the validation split"),
        "wavelet": {"family": cfg.wavelet_family, "level": cfg.wavelet_level,
                    "mode": cfg.wavelet_mode},
        "data_split": {"train": int(len(split["train"])), "val": int(len(split["val"])),
                       "test": int(len(split["test"]))},
        "oracle_errors": reference,
        "results": results,
        "representative": representative_row(results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote {args.output.relative_to(ROOT).as_posix()} "
          f"({time.time() - started:.0f} s)")

    if args.verify:
        return verify(report, bands, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

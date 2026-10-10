"""Per-mode error of the recovered POD coefficients.

For every configuration the reconstruction is projected onto the rank-128 POD basis and the error of each modal coefficient j is reported as

    e_j = sqrt( Σ_i (â_ij − a_ij)² / Σ_i a_ij² )

summed over the test snapshots i. Since the modal energy λ_j² spans several orders of magnitude, the interesting quantity is the correlation between e_j and λ_j² — the mechanism discussed in the paper.

Both the coefficients and the projection use the **physical** fields: the convolutional estimator stores its outputs in normalised units, so those are de-normalised first.

Inputs
    artifacts/pod_bases/...            rank-128 POD basis (both components)
    artifacts/<estimator runs>/        one test_raw.npz per configuration
Output
    artifacts/statistics/modal_coefficient_error.json
        per configuration: 128 per-mode errors, the Spearman correlation with
        the modal energy, and the energy-decile summary of the reported example (MLP at M=20, sigma=0)

Usage
    python -m applications.statistics.modal_coefficient_error
    python -m applications.statistics.modal_coefficient_error --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.data.io import load_npz  # noqa: E402

from applications.statistics.band_error_decomposition import (  # noqa: E402
    NOISE_SIGMAS,
    POD_BUNDLE,
    POD_RANK,
    SENSOR_COUNTS,
)
from features.training.estimator_runs import (  # noqa: E402
    load_run,
    noise_code,
    run_path,
)

OUTPUT = ROOT / "artifacts" / "statistics" / "modal_coefficient_error.json"

# The analysis is reported for one representative training run per estimator.
REPRESENTATIVE_SEED = 0
# Energy-decile grouping of the 128 modes used in the paper's table:
# 8 groups of 13 modes plus 2 groups of 12, counted from the most energetic mode.
DECILE_SIZES = [13] * 8 + [12] * 2
EXAMPLE_CONFIG = ("mlp", 20, 0.0)


def modal_errors(
    target: np.ndarray,
    reconstruction: np.ndarray,
    projection: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Per-mode NRMSE of the projected coefficients, one value per POD mode."""
    n = target.shape[0]
    truth = target.transpose(0, 2, 3, 1).reshape(n, -1) - mean
    estimate = reconstruction.transpose(0, 2, 3, 1).reshape(n, -1) - mean
    a_true = truth @ projection
    a_pred = estimate @ projection
    numerator = np.sum((a_pred - a_true) ** 2, axis=0)
    denominator = np.sum(a_true ** 2, axis=0) + 1e-12
    return np.sqrt(numerator / denominator)


def bootstrap_ci(
    mode_energy: np.ndarray,
    errors: np.ndarray,
    n_draws: int = 2000,
    seed: int = 42,
) -> list[float]:
    """95% bootstrap interval of the correlation, resampling the 128 modes."""
    rng = np.random.RandomState(seed)
    n_modes = errors.size
    draws = np.empty(n_draws)
    for draw in range(n_draws):
        selection = rng.randint(0, n_modes, size=n_modes)
        draws[draw] = spearmanr(mode_energy[selection], errors[selection])[0]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def decile_summary(errors: np.ndarray, mode_energy: np.ndarray) -> list[dict]:
    """Error statistics grouped by modal-energy decile (most energetic first)."""
    order = np.argsort(mode_energy)[::-1]
    rows = []
    start = 0
    for index, size in enumerate(DECILE_SIZES, start=1):
        selected = order[start:start + size]
        start += size
        rows.append({
            "decile": index,
            "n_modes": int(size),
            "energy_range_pct": [float(mode_energy[selected].min() * 100.0),
                                 float(mode_energy[selected].max() * 100.0)],
            "energy_share_pct": float(mode_energy[selected].sum() * 100.0),
            "nrmse_mean": float(np.mean(errors[selected])),
            "nrmse_median": float(np.median(errors[selected])),
            "nrmse_q25": float(np.percentile(errors[selected], 25)),
            "nrmse_q75": float(np.percentile(errors[selected], 75)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return verify(json.loads(args.output.read_text(encoding="utf-8"))["records"])

    bundle = load_npz(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)
    mean = np.asarray(bundle["mean_field"], dtype=np.float64)
    singular_values = np.asarray(bundle["singular_values"], dtype=np.float64)
    projection = basis.reshape(basis.shape[0], -1)[:POD_RANK].T          # (D, r)
    mean_flat = mean.ravel()
    mode_energy = (singular_values[:POD_RANK] ** 2) / (singular_values[0] ** 2)

    started = time.time()
    records, missing = [], []
    for model in ("mlp", "vcnn", "ridge"):
        for sensor_count in SENSOR_COUNTS:
            for sigma in NOISE_SIGMAS:
                run = run_path(model, sensor_count, sigma, REPRESENTATIVE_SEED)
                if run is None:
                    missing.append(f"{model} M={sensor_count} sigma={sigma}")
                    continue
                target, reconstruction = load_run(run)
                # load_run already returns physical fields for every estimator
                errors = modal_errors(target, reconstruction, projection, mean_flat)
                correlation, p_value = spearmanr(mode_energy, errors)
                records.append({
                    "model": model,
                    "sensor_count": sensor_count,
                    "noise_sigma": sigma,
                    "training_seed": REPRESENTATIVE_SEED,
                    "n_snapshots": int(target.shape[0]),
                    "nrmse_per_mode": errors.tolist(),
                    "spearman_r": float(correlation),
                    "spearman_p": float(p_value),
                    "spearman_ci_95": bootstrap_ci(mode_energy, errors),
                    "deciles": (decile_summary(errors, mode_energy)
                                if (model, sensor_count, sigma) == EXAMPLE_CONFIG else None),
                })
                print(f"  {model} M={sensor_count:<3} sigma={sigma:<6} "
                      f"rho = {correlation:+.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "per-mode error of the recovered POD coefficients",
        "config": {
            "pod_rank": POD_RANK,
            "representative_seed": REPRESENTATIVE_SEED,
            "n_snapshots_per_configuration": records[0]["n_snapshots"] if records else None,
            "mode_energy": "lambda_j^2 / lambda_1^2",
            "fields": "physical units",
        },
        "records": records,
    }), encoding="utf-8")
    print(f"wrote {len(records)} records ({time.time() - started:.0f} s)")
    if missing:
        print(f"WARNING: missing configurations: {missing[:3]}")
    if args.verify:
        return verify(records)
    return 0


def verify(records: list[dict]) -> int:
    """Compare the reported correlations with the values reported in the paper."""
    expected = {("mlp", 20, 0.0): -0.9772, ("vcnn", 20, 0.0): -0.9152,
                ("ridge", 20, 0.0): -0.8260}
    print("\nverification against the reference values")
    ok = True
    for record in records:
        key = (record["model"], record["sensor_count"], record["noise_sigma"])
        if key not in expected:
            continue
        want = expected[key]
        diff = abs(record["spearman_r"] - want)
        flag = "OK" if diff < 5e-3 else "DIFFERS"
        print(f"  {record['model']} M={record['sensor_count']} sigma={record['noise_sigma']}: "
              f"rho = {record['spearman_r']:+.4f} (paper {want:+.4f})  {flag}")
        ok &= diff < 5e-3
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

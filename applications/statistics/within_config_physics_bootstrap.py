"""Within-configuration correlation of the scale index and the global error with
the Laplacian error.

For every sensor count and noise level of one representative convolutional run,
the module correlates two per-snapshot diagnostics with the derivative error of
the same reconstruction:

    S_full  the contiguous scale count the reconstruction recovers
    GER     the global relative error of the reconstruction

The reported quantity is the difference of the two absolute Spearman
correlations,

    delta_rho = |rho(S_full, E)| - |rho(GER, E)|,

together with a 95% bootstrap interval and the fraction of resamples on the
other side of zero. A negative value means that the global error is the stronger
correlate of the physical error, which is what the manuscript reports. The same
statistics are computed a second time against the gradient RMSE.

A snapshot is the cluster of the analysis: ``block_bootstrap`` is called with
block length one, so each resample draws the run's snapshots with replacement.
The analysis keeps the fixed parameters of the manuscript -- one representative
convolutional run, 10,000 resamples, derivatives taken on the streamwise
component, bands and threshold taken from ``applications.config``.

Inputs
    the trained runs under artifacts/ (see applications/pipelines/03_train_estimators.py)
Output
    artifacts/statistics/within_config_physics_bootstrap.json

Usage
    python -m applications.statistics.within_config_physics_bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.statistics.block_bootstrap import block_bootstrap  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402
from luna.core.constants import EPS  # noqa: E402
from luna.wavelet.metrics import compute_S_full  # noqa: E402

OUTPUT = ROOT / "artifacts" / "statistics" / "within_config_physics_bootstrap.json"

#: Estimator and training seed of the representative run the manuscript reports.
MODEL = "vcnn"
TRAINING_SEED = 0
#: Channel carrying the streamwise component, on which the indices are defined.
COMPONENT = 0
#: Bootstrap settings of the manuscript.
N_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
CI_LEVEL = 95.0
#: A correlation needs variation in both arguments; a configuration below this
#: many snapshots or with a constant argument is not reported.
MIN_SNAPSHOTS = 10

#: The diagnostics correlated with the physical error, in the order of the
#: feature matrix handed to the bootstrap.
PHYSICS_FIELDS = ("laplacian_rmse", "gradient_rmse")


def _laplacian_rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """RMSE between the discrete Laplacians of two fields."""

    def laplacian(field: np.ndarray) -> np.ndarray:
        return (np.gradient(np.gradient(field, axis=0), axis=0)
                + np.gradient(np.gradient(field, axis=1), axis=1))

    return float(np.sqrt(np.mean((laplacian(truth) - laplacian(prediction)) ** 2)))


def _gradient_rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """RMS of the gradient difference, with the two directions combined."""
    dy_truth, dx_truth = np.gradient(truth)
    dy_prediction, dx_prediction = np.gradient(prediction)
    return float(np.sqrt(np.mean((dx_truth - dx_prediction) ** 2)
                         + np.mean((dy_truth - dy_prediction) ** 2)))


def snapshot_diagnostics(target: np.ndarray, reconstruction: np.ndarray, cfg) -> dict:
    """Per-snapshot scale count, global error and derivative errors.

    ``target`` and ``reconstruction`` are the physical-unit fields of one run,
    shaped (snapshots, channels, rows, columns). The band errors behind
    ``S_full`` use the configuration's wavelet, level, boundary mode and
    threshold.
    """
    n_snapshots = target.shape[0]
    diagnostics = {
        "s_full": np.empty(n_snapshots, dtype=np.float64),
        "ger": np.empty(n_snapshots, dtype=np.float64),
        "laplacian_rmse": np.empty(n_snapshots, dtype=np.float64),
        "gradient_rmse": np.empty(n_snapshots, dtype=np.float64),
    }
    for index in range(n_snapshots):
        truth = target[index, COMPONENT]
        prediction = reconstruction[index, COMPONENT]
        diagnostics["ger"][index] = (
            np.linalg.norm((truth - prediction).ravel())
            / (np.linalg.norm(truth.ravel()) + EPS)
        )
        diagnostics["s_full"][index] = compute_S_full(
            truth, prediction, cfg.tau, cfg.wavelet_family,
            cfg.wavelet_level, cfg.wavelet_mode,
        )
        diagnostics["laplacian_rmse"][index] = _laplacian_rmse(truth, prediction)
        diagnostics["gradient_rmse"][index] = _gradient_rmse(truth, prediction)
    return diagnostics


def bootstrap_delta_rho(features: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Snapshot-cluster bootstrap distribution of the correlation difference.

    ``features`` holds one row per snapshot with the columns ``S_full``,
    ``GER`` and the physical error. A block length of one resamples single
    snapshots with replacement, which is the cluster bootstrap of this analysis.
    """
    def statistic(rows: np.ndarray) -> float:
        with warnings.catch_warnings():
            # A resample can collapse a diagnostic to a constant, for which the
            # correlation is undefined; the legacy analysis propagated that NaN.
            warnings.simplefilter("ignore")
            rho_scale, _ = sp_stats.spearmanr(rows[:, 0], rows[:, 2])
            rho_ger, _ = sp_stats.spearmanr(rows[:, 1], rows[:, 2])
        return abs(float(rho_scale)) - abs(float(rho_ger))

    return block_bootstrap(
        features, statistic, block_len=1, n_resamples=n_resamples,
        seed=seed, method="moving_block",
    )


def correlation_row(sensor_count, sigma, diagnostics, physics_field,
                    n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """One (sensor count, noise level) row, or ``None`` when it is degenerate."""
    scale = diagnostics["s_full"].astype(np.float64)
    ger = diagnostics["ger"].astype(np.float64)
    physics = diagnostics[physics_field].astype(np.float64)
    if scale.size < MIN_SNAPSHOTS or np.std(scale) < EPS or np.std(physics) < EPS:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_scale, _ = sp_stats.spearmanr(scale, physics)
        rho_ger, _ = sp_stats.spearmanr(ger, physics)
    delta_rho = abs(float(rho_scale)) - abs(float(rho_ger))

    features = np.column_stack([scale, ger, physics])
    distribution = bootstrap_delta_rho(features, n_resamples, seed)
    alpha = (100.0 - CI_LEVEL) / 2.0
    ci_low = float(np.percentile(distribution, alpha))
    ci_high = float(np.percentile(distribution, 100.0 - alpha))
    # The legacy analysis used the same definition: resamples beyond zero count
    # against the sign of the observed difference.
    p_value = float(np.mean(distribution <= 0) if delta_rho > 0
                    else np.mean(distribution >= 0))
    significant = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))

    return {
        "M": int(sensor_count),
        "sigma": float(sigma),
        "n_samples": int(scale.size),
        "rho_S_full": float(rho_scale),
        "rho_GER": float(rho_ger),
        "abs_rho_S_full": abs(float(rho_scale)),
        "abs_rho_GER": abs(float(rho_ger)),
        "delta_rho": delta_rho,
        "delta_rho_CI": [ci_low, ci_high],
        "p_value": p_value,
        "significant": significant,
        "better_predictor": "S_full" if delta_rho > 0 else "GER",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    cfg = get_config()
    print("=" * 60)
    print(f"within-configuration physics correlation: {MODEL} seed {TRAINING_SEED}")
    print(f"  tau={cfg.tau}, wavelet={cfg.wavelet_family}, level={cfg.wavelet_level}, "
          f"{N_RESAMPLES} resamples")
    print("=" * 60 + "\n")

    rows: dict[str, list[dict]] = {field: [] for field in PHYSICS_FIELDS}
    n_total = 0
    for sensor_count in cfg.M_values:
        for sigma in cfg.sigma_values:
            path = run_path(MODEL, sensor_count, sigma, TRAINING_SEED)
            if path is None:
                print(f"  [SKIP] {MODEL} M={sensor_count} sigma={sigma}")
                continue
            target, reconstruction = load_run(path)
            diagnostics = snapshot_diagnostics(target, reconstruction, cfg)
            if n_total == 0:
                n_total = int(target.shape[0])
            for field in PHYSICS_FIELDS:
                row = correlation_row(sensor_count, sigma, diagnostics, field)
                if row is None:
                    print(f"  [SKIP] {MODEL} M={sensor_count} sigma={sigma} ({field})")
                    continue
                rows[field].append(row)
                if field == PHYSICS_FIELDS[0]:
                    print(f"  M={sensor_count:>2} sigma={sigma:<6}: "
                          f"delta_rho={row['delta_rho']:+.4f} "
                          f"[{row['delta_rho_CI'][0]:+.4f}, "
                          f"{row['delta_rho_CI'][1]:+.4f}] "
                          f"-> {row['better_predictor']}")

    payload = {
        "method": f"full_bootstrap_{MODEL}_seed{TRAINING_SEED}",
        "description": (
            f"{MODEL} seed={TRAINING_SEED}, full {n_total} samples per configuration, "
            f"{N_RESAMPLES} bootstrap resamples; delta_rho = |rho(S_full, E)| - |rho(GER, E)| "
            "with E the Laplacian RMSE (bootstrap_results) or the gradient RMSE "
            "(gradient_results)"
        ),
        "n_total": n_total,
        "bootstrap_results": rows["laplacian_rmse"],
        "gradient_results": rows["gradient_rmse"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reported = len(rows["laplacian_rmse"])
    winners = sum(1 for row in rows["laplacian_rmse"] if row["better_predictor"] == "GER")
    significant = sum(1 for row in rows["laplacian_rmse"] if row["significant"])
    print(f"\n  {reported} configurations reported, GER the stronger correlate in "
          f"{winners}, interval excluding zero in {significant}")
    print(f"[OK] wrote {args.output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

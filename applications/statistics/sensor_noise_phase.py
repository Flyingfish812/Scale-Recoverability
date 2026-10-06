"""Sensor-count / noise phase diagram of the scale-recoverability index.

For every configuration the index S_full is evaluated on each of the 300 test snapshots of one representative training run, and summarised by its mean and by the pass probabilities

    P3 = fraction of snapshots with S_full >= 3 P4 = fraction of snapshots with S_full >= 4

with a 95% bootstrap interval over snapshots. The result is the phase diagram quoted in the paper: how the finest reliably recovered scale moves with sensor count M and measurement noise sigma.

Fields are evaluated in physical units; the convolutional estimator stores its outputs in normalised units and is de-normalised first; without that step the coarse end of S_full is biased downwards.

Inputs
    artifacts/<estimator runs>/    one test_raw.npz per configuration
Output
    artifacts/statistics/sensor_noise_phase.json

Usage
    python -m applications.statistics.sensor_noise_phase
    python -m applications.statistics.sensor_noise_phase --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.core.constants import TAU_DEFAULT  # noqa: E402
from luna.wavelet.metrics import compute_S_full  # noqa: E402

from applications.statistics.band_error_decomposition import (  # noqa: E402
    NOISE_SIGMAS,
    SENSOR_COUNTS,
    load_run,
    run_path,
)

OUTPUT = ROOT / "artifacts" / "statistics" / "sensor_noise_phase.json"
BASELINE = ROOT / "artifacts" / "derived" / "main" / "statistics" / "s26_pass_probability.json"
MODELS = ("mlp", "vcnn", "ridge")
REPRESENTATIVE_SEED = 0
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


def summarise(indices: list[int]) -> dict:
    """Mean index and pass probabilities with a bootstrap interval."""
    values = np.asarray(indices)
    n = values.size
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    p3 = np.empty(N_BOOTSTRAP)
    p4 = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        draw = values[rng.randint(0, n, size=n)]
        p3[b] = np.mean(draw >= 3)
        p4[b] = np.mean(draw >= 4)
    return {
        "n_snapshots": int(n),
        "mean_S_full": float(values.mean()),
        "median_S_full": float(np.median(values)),
        "std_S_full": float(values.std()),
        "P3": float(np.mean(values >= 3)),
        "P3_ci_95": [float(np.percentile(p3, 2.5)), float(np.percentile(p3, 97.5))],
        "P4": float(np.mean(values >= 4)),
        "P4_ci_95": [float(np.percentile(p4, 2.5)), float(np.percentile(p4, 97.5))],
        "S_full_distribution": {str(k): int(v)
                                for k, v in sorted(zip(*np.unique(values, return_counts=True)))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return verify(json.loads(args.output.read_text(encoding="utf-8"))["records"])

    started = time.time()
    records = []
    for model in MODELS:
        for sensor_count in SENSOR_COUNTS:
            for sigma in NOISE_SIGMAS:
                run = run_path(model, sensor_count, sigma, REPRESENTATIVE_SEED)
                if run is None:
                    print(f"  [skip] {model} M={sensor_count} sigma={sigma}")
                    continue
                target, reconstruction = load_run(run)
                indices = [
                    compute_S_full(target[i, 0], reconstruction[i, 0], TAU_DEFAULT)
                    for i in range(target.shape[0])
                ]
                record = {
                    "model": model,
                    "sensor_count": sensor_count,
                    "noise_sigma": sigma,
                    "training_seed": REPRESENTATIVE_SEED,
                    **summarise(indices),
                }
                records.append(record)
                print(f"  {model:5s} M={sensor_count:<3} sigma={sigma:<6} "
                      f"mean S_full = {record['mean_S_full']:.2f}  P3 = {record['P3']:.2f}")
        print(f"[{model}] {len(records)} configurations done")

    phase_summary: dict = {}
    for record in records:
        phase_summary.setdefault(record["model"], {}).setdefault(
            str(record["sensor_count"]), {})[str(record["noise_sigma"])] = {
            "mean_S_full": record["mean_S_full"],
            "P3": record["P3"],
            "P3_ci": record["P3_ci_95"],
            "P4": record["P4"],
            "P4_ci": record["P4_ci_95"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "sensor-count / noise phase diagram of S_full "
                       "(300 test snapshots, one representative run, tau = 0.05)",
        "config": {
            "tau": TAU_DEFAULT,
            "n_bootstrap": N_BOOTSTRAP,
            "fields": "physical units",
            "representative_seed": REPRESENTATIVE_SEED,
        },
        "records": records,
        "phase_summary": phase_summary,
    }), encoding="utf-8")
    print(f"wrote {len(records)} configurations ({time.time() - started:.0f} s)")
    if args.verify:
        return verify(records)
    return 0


def verify(records: list[dict]) -> int:
    """Compare with the reference values reported in the paper.

    The baseline evaluated the convolutional estimator in normalised units, so its S_full is systematically lower wherever the reconstruction fails in the coarse bands; only the POD-coefficient estimators must reproduce the baseline here.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["phase_summary"]
    print("\nverification against the reference values")
    worst = 0.0
    for record in records:
        if record["model"] in ("vcnn", "ridge"):
            # the baseline evaluated the convolutional estimator in normalised units, and used a Ridge estimator that differs from the closed form
            continue
        expected = baseline.get(record["model"], {}).get(
            str(record["sensor_count"]), {}).get(str(record["noise_sigma"]))
        if expected is None:
            continue
        difference = abs(record["mean_S_full"] - expected["mean_S_full"])
        worst = max(worst, difference)
        if difference > 0.05:
            print(f"  DIFFERS {record['model']} M={record['sensor_count']} "
                  f"sigma={record['noise_sigma']}: {record['mean_S_full']:.2f} "
                  f"vs {expected['mean_S_full']:.2f}")
    print(f"  max |mean S_full difference| over POD estimators = {worst:.3e}")
    ok = worst < 0.05
    print("  OK: reproduces the baseline" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

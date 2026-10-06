"""Across-family summary of the sensor-placement study.

The main experiments use one nested sensor sequence. To show that the conclusions do not depend on that particular sampling, the same protocol is repeated on five independently drawn nested sequences, and the convolutional estimator is additionally retrained on two of them as an auxiliary check. This module reduces those runs to the per-family error levels, the spread across families, and the sensor-count trend that the paper reports.

Every quantity is evaluated on physical fields: the convolutional runs store normalised fields and are converted back with the constants recovered from the raw sequence, so the numbers are comparable with the POD-coefficient and least-squares estimators.

Inputs
    artifacts/derived/supplementary/predictions/{family}/...   test_raw.npz
    data/cylinder2d_q1.npy                                     raw snapshots
Output
    artifacts/statistics/sensor_family_summary.json
    artifacts/statistics/sensor_family/mask_level_summary.csv
    artifacts/statistics/sensor_family/mask_variance.csv
    artifacts/statistics/sensor_family/sensor_count_effect.csv

Usage
    python -m applications.statistics.sensor_family_summary
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.training.estimator_runs import (  # noqa: E402
    FAMILY_ROOT,
    load_run,
    noise_code,
)

NOISE_CODES = {noise_code(sigma): sigma for sigma in (0.0, 0.001, 0.01, 0.1)}
# : Models whose across-family spread the paper reports.
FAMILY_MODELS = ("mlp", "ridge", "gappy")
# : Sensor counts of the auxiliary convolutional validation.
VCNN_SENSOR_COUNTS = (10, 30, 50)
# : Noise levels of the auxiliary convolutional validation.
VCNN_SIGMAS = (0.0, 0.1)
# : Training epochs of every learned estimator in the family study.
TRAINING_EPOCHS = 2000
OUT_DIR = ROOT / "artifacts" / "statistics" / "sensor_family"
OUTPUT = ROOT / "artifacts" / "statistics" / "sensor_family_summary.json"


def discover_runs() -> list[dict]:
    """Every family run on disk, with its configuration read from its path."""
    runs = []
    for path in sorted(FAMILY_ROOT.rglob("test_raw.npz")):
        parts = path.relative_to(FAMILY_ROOT).parts
        if len(parts) < 6:
            continue
        family, model_dir, seed_dir, _, code, _ = parts
        model, _, sensor_count = model_dir.partition("_n")
        if not sensor_count.isdigit() or code not in NOISE_CODES:
            continue
        runs.append({
            "mask_family": family,
            "model": model,
            "sensor_count": int(sensor_count),
            "training_seed": int(seed_dir.removeprefix("seed")),
            "sigma": NOISE_CODES[code],
            "path": path,
        })
    return runs


def errors_of(path: Path) -> tuple[np.ndarray, float]:
    """Per-snapshot relative L2 error over the full state, in physical units."""
    target, reconstruction = load_run(path)
    flat_target = target.reshape(target.shape[0], -1)
    flat_recon = reconstruction.reshape(reconstruction.shape[0], -1)
    errors = np.linalg.norm(flat_recon - flat_target, axis=1) / np.linalg.norm(
        flat_target, axis=1
    )
    return errors, float(errors.mean())


def aggregate(records: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Group the per-run error levels and summarise each group."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for record in records:
        groups[tuple(record[f] for f in key_fields)].append(record["ger_mean"])

    rows = []
    for key, values in sorted(groups.items()):
        array = np.asarray(values)
        row = dict(zip(key_fields, key))
        row.update({
            "n_runs": int(array.size),
            "ger_mean_min": float(array.min()),
            "ger_mean_median": float(np.median(array)),
            "ger_mean_max": float(array.max()),
            "ger_mean_span": float(array.max() - array.min()),
        })
        rows.append(row)
    return rows


def sensor_count_trend(
    variance_rows: list[dict], model: str, sigma: float, sensor_counts: tuple[int, ...]
) -> dict:
    """Median error level of a model at the requested sensor counts."""
    trend = {}
    for row in variance_rows:
        if row["model"] == model and row["sigma"] == sigma and row["sensor_count"] in sensor_counts:
            trend[row["sensor_count"]] = row["ger_mean_median"]
    return trend


def main() -> int:
    cfg = get_config()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("== sensor-family summary")
    runs = discover_runs()
    if not runs:
        raise SystemExit(f"no family runs under {FAMILY_ROOT}")
    print(f"   {len(runs)} runs found")

    records = []
    for run in runs:
        _, ger_mean = errors_of(run["path"])
        records.append({k: v for k, v in run.items() if k != "path"} | {"ger_mean": ger_mean})

    with (out_dir / "mask_level_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "mask_family", "sensor_count", "training_seed", "sigma", "ger_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda r: (r["model"], r["sensor_count"], r["sigma"], r["mask_family"], r["training_seed"]),
        ):
            writer.writerow({f: record[f] for f in fields})

    variance_rows = aggregate(records, ("model", "sensor_count", "sigma"))
    with (out_dir / "mask_variance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(variance_rows)

    with (out_dir / "sensor_count_effect.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(variance_rows)

    families = sorted({record["mask_family"] for record in records})
    models = sorted({record["model"] for record in records})
    n_families = {
        model: len({r["mask_family"] for r in records if r["model"] == model})
        for model in models
    }

    clean = {
        model: sensor_count_trend(variance_rows, model, 0.0, cfg.M_values)
        for model in FAMILY_MODELS
    }
    convolutional = {
        "n_families": n_families.get("vcnn", 0),
        "sensor_counts": list(VCNN_SENSOR_COUNTS),
        "sigmas": list(VCNN_SIGMAS),
        "epochs": TRAINING_EPOCHS,
        "clean": {
            str(M): sensor_count_trend(variance_rows, "vcnn", 0.0, (M,)).get(M)
            for M in VCNN_SENSOR_COUNTS
        },
        "noisy": {
            str(M): sensor_count_trend(variance_rows, "vcnn", 0.1, (M,)).get(M)
            for M in VCNN_SENSOR_COUNTS
        },
    }
    m30 = convolutional["clean"].get("30")
    m50 = convolutional["clean"].get("50")
    convolutional["m30_to_m50_improvement_percent"] = (
        round(100.0 * (m30 - m50) / m30, 1) if m30 and m50 else None
    )

    mlp_30, mlp_50 = clean["mlp"].get(30), clean["mlp"].get(50)
    summary = {
        "quantity": "error level across independently sampled nested sensor sequences",
        "convention": {
            "fields": "physical units for every estimator",
            "error": "relative L2 over the full two-component state, 300 test snapshots",
            "aggregation": "median over the runs of each (model, sensor count, noise level)",
        },
        "families": families,
        "n_runs": len(records),
        "n_runs_per_model": {
            model: sum(1 for r in records if r["model"] == model) for model in models
        },
        "n_families_per_model": n_families,
        "clean_ger": clean,
        "noisy_ger": {
            model: sensor_count_trend(variance_rows, model, 0.1, cfg.M_values)
            for model in FAMILY_MODELS
        },
        "mlp_saturated_at_m30": bool(
            mlp_30 and mlp_50 and (mlp_30 - mlp_50) / mlp_30 < 0.02
        ),
        "convolutional_validation": convolutional,
        "across_family_spread": [
            row for row in variance_rows if row["sigma"] == 0.0
        ],
        "runtime_s": round(time.time() - start, 2),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] {OUTPUT.name} and 3 CSV files ({time.time() - start:.1f}s)")
    for model in FAMILY_MODELS:
        trend = clean[model]
        print(f"   {model:6s} (n_families={n_families[model]}): " + "  ".join(
            f"M={M}: {value:.5f}" for M, value in sorted(trend.items())
        ))
    print(f"   vcnn   (n_families={n_families.get('vcnn', 0)}): " + "  ".join(
        f"M={M}: {value:.5f}" for M, value in sorted(convolutional["clean"].items())
    ))
    print(f"   vcnn noisy: " + "  ".join(
        f"M={M}: {value:.5f}" for M, value in sorted(convolutional["noisy"].items())
    ))
    print(f"   vcnn M30->M50 improvement: "
          f"{convolutional['m30_to_m50_improvement_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

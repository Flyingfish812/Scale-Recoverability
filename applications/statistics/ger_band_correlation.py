"""Correlation between the global error and the per-band errors.

Within one configuration (sensor count, noise level, training seed) the global
error and the individual band errors are strongly correlated, which is what makes
the global ratio a usable proxy for the recursive scale count. This module
measures that correlation with Spearman's rank coefficient for every
configuration of the POD-coefficient sweep and reports the median over
configurations, band by band.

Output
------
artifacts/statistics/ger_band_correlation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.metrics.sample_metrics import BANDS  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
OUT_DIR = ROOT / "artifacts" / "statistics"
MODEL = "mlp"


def configuration_keys(records: list[dict]) -> list[tuple[int, float, int]]:
    """Every (sensor count, noise level, training seed) of the sweep."""
    keys = {(r["sensor_count"], r["noise_sigma"], r["training_seed"])
            for r in records if r["model"] == MODEL}
    return sorted(keys)


def main() -> int:
    print("== global vs band error correlation")
    records = json.loads(RECORDS.read_text(encoding="utf-8"))["records"]

    per_config: dict[str, dict[str, float]] = {}
    for sensors, sigma, seed in configuration_keys(records):
        rows = [r for r in records
                if r["model"] == MODEL and r["sensor_count"] == sensors
                and abs(r["noise_sigma"] - sigma) < 1e-12
                and r["training_seed"] == seed]
        if len(rows) < 10:
            continue
        global_error = [r["global_error"] for r in rows]
        key = f"M{sensors}_s{sigma:g}_seed{seed}"
        per_config[key] = {
            band: float(spearmanr(global_error,
                                  [r["band_errors"][band]["total"] for r in rows]).statistic)
            for band in BANDS
        }

    median = {band: round(float(np.median([v[band] for v in per_config.values()])), 3)
              for band in BANDS}
    spread = {band: [round(float(np.min([v[band] for v in per_config.values()])), 3),
                     round(float(np.max([v[band] for v in per_config.values()])), 3)]
              for band in BANDS}

    result = {
        "description": ("Spearman correlation between the global error ratio and the "
                        "per-band direct error, within each configuration of the "
                        f"{MODEL.upper()} sweep; medians and extremes over "
                        f"{len(per_config)} configurations"),
        "model": MODEL,
        "n_configurations": len(per_config),
        "values": median,
        "range": [min(median.values()), max(median.values())],
        "median": round(float(np.median(list(median.values()))), 3),
        "per_band_min_max": spread,
        "per_configuration": per_config,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "ger_band_correlation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   {len(per_config)} configurations; medians {median}")
    print(f"   range over bands {result['range']}, median {result['median']}")
    print(f"[OK] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

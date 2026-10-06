"""How much of the scale count is training-seed variation?

Each configuration of the sweep is trained with three seeds. This module pools the test snapshots of those seeds and gives the uncertainty of the mean scale count of the configuration from a percentile bootstrap, so that the reported differences between configurations can be read against the noise floor of the training.

Output
------
artifacts/statistics/seed_stability.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402

RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
OUT_DIR = ROOT / "artifacts" / "statistics"
N_RESAMPLES = 10_000
CI_LEVEL = 0.95


def bootstrap_width(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Mean and the width of the percentile interval of the mean."""
    n = values.size
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = values[draws].mean(axis=1)
    low = float(np.quantile(means, (1.0 - CI_LEVEL) / 2))
    high = float(np.quantile(means, 1.0 - (1.0 - CI_LEVEL) / 2))
    return float(values.mean()), round(high - low, 3)


def main() -> int:
    print("== training-seed stability of the scale count")
    config = get_config()
    records = json.loads(RECORDS.read_text(encoding="utf-8"))["records"]
    rng = np.random.default_rng(config.block_bootstrap.get("seed", 0))

    per_config = {}
    for model in ("mlp", "vcnn", "ridge"):
        for sensors in config.M_values:
            for sigma in config.sigma_values:
                rows = [r for r in records
                        if r["model"] == model and r["sensor_count"] == sensors
                        and abs(r["noise_sigma"] - sigma) < 1e-12]
                if not rows:
                    continue
                seeds = sorted({r["training_seed"] for r in rows})
                scale_counts = np.asarray([r["s_full"] for r in rows], dtype=float)
                mean, width = bootstrap_width(scale_counts, rng)
                per_config_key = f"{model}_M{sensors}_s{sigma:g}"
                per_config[per_config_key] = {
                    "model": model,
                    "sensor_count": sensors,
                    "noise_sigma": sigma,
                    "n_seeds": len(seeds),
                    "n_samples": int(scale_counts.size),
                    "mean_S_full": round(mean, 3),
                    "ci_width": width,
                }

    widths = [v["ci_width"] for v in per_config.values()]
    variable = [w for w in widths if w > 1e-9]
    result = {
        "description": (f"Percentile bootstrap ({N_RESAMPLES} resamples) of the mean "
                        f"scale count of each configuration, test snapshots of the "
                        f"training seeds pooled; {int(CI_LEVEL * 100)}% interval width"),
        "ci_level": CI_LEVEL,
        "n_resamples": N_RESAMPLES,
        "n_configurations": len(per_config),
        "ci_width_range": [min(widths), max(widths)],
        "ci_width_range_variable": [min(variable), max(variable)],
        "per_configuration": per_config,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "seed_stability.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   {len(per_config)} configurations, CI width in "
          f"[{min(widths)}, {max(widths)}] "
          f"({len(variable)} with a varying scale count, "
          f"[{min(variable)}, {max(variable)}])")
    print(f"[OK] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

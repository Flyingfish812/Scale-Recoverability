"""How measurement noise amplifies the error of each wavelet band.

For every estimator the mean band error at the highest tested noise level is compared with the clean case,

    ratio_b = mean over snapshots of E_total(b) at sigma = 0.1
              -----------------------------------------------------  ,
              mean over snapshots of E_total(b) at sigma = 0

and the same ratio is formed for the global error. Ratios above one mean that noise degrades that band; the comparison across bands shows which scales the noise reaches first.

Inputs
    artifacts/statistics/band_error_records.json   (produced by
    applications.statistics.band_error_records)
Output
    artifacts/statistics/noise_propagation.json

Usage
    python -m applications.statistics.noise_propagation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.core.constants import BANDS_CF  # noqa: E402

RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
OUTPUT = ROOT / "artifacts" / "statistics" / "noise_propagation.json"
CLEAN, NOISY = 0.0, 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    records = json.loads(RECORDS.read_text(encoding="utf-8"))["records"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["sensor_count"], record["noise_sigma"])].append(record)

    per_configuration = {}
    for (model, sensor_count, _sigma) in list(grouped):
        if _sigma != CLEAN:
            continue
        clean = grouped.get((model, sensor_count, CLEAN))
        noisy = grouped.get((model, sensor_count, NOISY))
        if not clean or not noisy:
            continue
        ratios = {}
        clean_means, noisy_means = {}, {}
        for band in BANDS_CF:
            clean_means[band] = float(np.mean([r["band_errors"][band]["total"] for r in clean]))
            noisy_means[band] = float(np.mean([r["band_errors"][band]["total"] for r in noisy]))
            ratios[band] = noisy_means[band] / max(clean_means[band], 1e-300)
        clean_global = float(np.mean([r["global_error"] for r in clean]))
        noisy_global = float(np.mean([r["global_error"] for r in noisy]))
        per_configuration[f"{model}_M{sensor_count:02d}"] = {
            "model": model,
            "sensor_count": sensor_count,
            "n_clean": len(clean),
            "n_noisy": len(noisy),
            "clean_band_means": clean_means,
            "noisy_band_means": noisy_means,
            "clean_global_error": clean_global,
            "noisy_global_error": noisy_global,
            "degradation_ratio": {**ratios, "global": noisy_global / max(clean_global, 1e-300)},
        }

    summary = {}
    for model in ("mlp", "vcnn", "ridge"):
        entries = [v for v in per_configuration.values() if v["model"] == model]
        if not entries:
            continue
        summary[model] = {
            band: float(np.mean([e["degradation_ratio"][band] for e in entries]))
            for band in BANDS_CF
        }
        summary[model]["global"] = float(np.mean([e["degradation_ratio"]["global"] for e in entries]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "band-wise amplification of the error by measurement noise, "
                       "comparing sigma=0.1 with sigma=0",
        "config": {"clean": CLEAN, "noisy": NOISY, "records": str(RECORDS.relative_to(ROOT))},
        "per_configuration": per_configuration,
        "mean_degradation_per_model": summary,
    }), encoding="utf-8")
    for model, values in summary.items():
        print(f"  {model:5s}: " + " ".join(f"{b}={values[b]:.1f}x" for b in BANDS_CF))
    print(f"wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Low-GER share of each estimator's test snapshots.

Splits every configuration of the main comparison at the median global error of its test snapshots and asks how often the better half fails to recover the three coarsest wavelet bands (``S_full < 3``). A large share means that the scale index does not simply follow the global error: reconstructions that are accurate on average still lose the coarse scales.

The three estimators of the main comparison are reported separately and jointly. The least-squares map is deterministic and contributes one run per configuration; the POD-coefficient network and the convolutional estimator contribute one run per training seed, and the seeds are pooled inside a configuration before the median split.

Inputs
    artifacts/statistics/band_error_records.json   per-snapshot global error and
                                                   S_full of every configuration
Output
    artifacts/statistics/low_ger_analysis.json     one block per estimator plus
                                                   ``all``, each holding ``low_ger_samples``, ``sfull_lt_3`` and ``pct``

Usage
    python -m applications.statistics.low_ger_analysis
    python -m applications.statistics.low_ger_analysis --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402

RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
OUTPUT = ROOT / "artifacts" / "statistics" / "low_ger_analysis.json"
# : Frozen artifact of the earlier analysis of the same quantity, read by --verify.
REFERENCE = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "s08b_low_ger_final.json")

# : Estimators reported separately, in the order of the artifact.
MODELS = ("ridge", "mlp", "vcnn")
# : A snapshot counts as resolved when it recovers the three coarsest bands.
RESOLVED_BANDS = 3
# : Fields the paper reads from every block of the artifact.
FIELDS = ("low_ger_samples", "sfull_lt_3", "pct")


def sigma_key(sigma: float) -> float:
    """Noise level rounded to the precision the configuration files carry."""
    return round(float(sigma), 6)


def low_ger_table(records: list[dict], cfg) -> dict:
    """Below-median share per estimator, with the median taken per configuration.

    Every configuration is one (model, sensor count, noise level) cell; the training seeds inside a cell are pooled, as the comparison is between configurations rather than between runs.
    """
    samples: dict[tuple[str, int, float], list[tuple[float, int]]] = defaultdict(list)
    seeds: dict[tuple[str, int, float], set[int]] = defaultdict(set)
    for record in records:
        if record["model"] not in MODELS:
            continue
        key = (record["model"], int(record["sensor_count"]), sigma_key(record["noise_sigma"]))
        samples[key].append((float(record["global_error"]), int(record["s_full"])))
        seeds[key].add(int(record["training_seed"]))

    table: dict[str, dict] = {}
    total_low = total_resolved = 0
    for model in MODELS:
        low = resolved = 0
        for sensor_count in cfg.M_values:
            for sigma in cfg.sigma_values:
                key = (model, int(sensor_count), sigma_key(sigma))
                cell = samples.get(key)
                if cell is None:
                    raise SystemExit(
                        f"no records for {model} M={sensor_count} sigma={sigma}: "
                        "run applications.statistics.band_error_records first"
                    )
                expected = 1 if model == "ridge" else len(cfg.seeds)
                if len(seeds[key]) != expected:
                    raise SystemExit(
                        f"{model} M={sensor_count} sigma={sigma}: "
                        f"{len(seeds[key])} training seeds on disk, {expected} configured"
                    )
                median = float(np.median([ger for ger, _ in cell]))
                for ger, s_full in cell:
                    if ger < median:
                        low += 1
                        if s_full < RESOLVED_BANDS:
                            resolved += 1
        table[model] = {
            "low_ger_samples": low,
            "sfull_lt_3": resolved,
            "pct": round(resolved / low * 100, 1) if low else 0.0,
        }
        total_low += low
        total_resolved += resolved

    table["all"] = {
        "low_ger_samples": total_low,
        "sfull_lt_3": total_resolved,
        "pct": round(total_resolved / total_low * 100, 1) if total_low else 0.0,
    }
    return table


def verify(table: dict) -> int:
    """Field-by-field comparison with the frozen artifact of the same quantity."""
    if not REFERENCE.exists():
        print(f"  [skip] {REFERENCE.relative_to(ROOT)} is not available")
        return 0

    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
    print(f"\ncomparison with {REFERENCE.relative_to(ROOT).as_posix()}")
    differences = 0
    for name in list(MODELS) + ["all"]:
        for field in FIELDS:
            new, old = table[name][field], frozen[name][field]
            if new != old:
                differences += 1
                print(f"  DIFF {name}.{field}: {new} vs {old}")
            else:
                print(f"  ok   {name}.{field}: {new}")
    print("  every field identical" if not differences
          else f"  {differences} fields differ")
    return 1 if differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    print("== low-GER statistics")
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    records = payload["records"]
    print(f"   {len(records)} records from {args.records.relative_to(ROOT).as_posix()}")

    recorded_tau = payload.get("config", {}).get("tau")
    if recorded_tau is not None and abs(float(recorded_tau) - cfg.tau) > 1e-12:
        raise SystemExit(
            f"the record set was built at tau={recorded_tau}, "
            f"the configuration says tau={cfg.tau}"
        )

    table = low_ger_table(records, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    print(f"\n   {'estimator':<10}{'low-GER':>10}{'S_full<3':>10}{'pct':>8}")
    for name in list(MODELS) + ["all"]:
        row = table[name]
        print(f"   {name:<10}{row['low_ger_samples']:>10}{row['sfull_lt_3']:>10}"
              f"{row['pct']:>8.1f}")
    print(f"\n[OK] wrote {args.output.relative_to(ROOT).as_posix()}")

    if args.verify:
        return verify(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Excess per-band error of the estimators over the rank-128 truncation.

For the reference configuration of the main comparison -- 20 sensor locations,
noise-free measurements, first training seed -- the module reports, band by
band, the mean relative error of a reconstruction minus the error of the
rank-128 POD truncation reference:

    delta_excess_band_error[model][band]
        = mean_snapshots(total(band)) - reference(band)

The estimator term is the mean of ``band_errors[band]["total"]`` over the 300
test snapshots of the run, taken from the per-snapshot record set; the reference
term is the NC rank-128 band mean of the truncation audit. Both terms therefore
describe the streamwise component in physical units, and the excess is the part
of the reconstruction error that remains once the rank-128 representation floor
is removed.

The POD-coefficient network and the convolutional estimator are recomputed here.
The least-squares estimator is computed as well, as a cross-check of the
reference: the manuscript takes its excess errors from the same record set
through the band-error decomposition.

Two reference conventions are reported. ``truncation_errors`` is the mean of the
record set's own per-snapshot truncation term, which pairs every estimator error
with the truncation of the same snapshot; ``delta`` subtracts the single
reference band mean of the truncation audit instead, which is the convention the
manuscript uses. The two differ by the spread of the truncation term over the
test snapshots.

Inputs
    artifacts/statistics/band_error_records.json         per-snapshot band errors
    artifacts/statistics/truncation_reference_audit.json rank-r truncation reference
Output
    artifacts/statistics/excess_error_recompute.json

Usage
    python -m applications.statistics.excess_error_recompute
    python -m applications.statistics.excess_error_recompute --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402

RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
TRUNCATION = ROOT / "artifacts" / "statistics" / "truncation_reference_audit.json"
OUTPUT = ROOT / "artifacts" / "statistics" / "excess_error_recompute.json"
#: Frozen artifact of the earlier recomputation of the same quantity, read by --verify.
REFERENCE = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "s02_recomputed_values.json")

#: Estimators of the reference comparison, in the order of the artifact.
MODELS = ("ridge", "mlp", "vcnn")
#: The configuration the manuscript reports.
SENSOR_COUNT = 20
NOISE_SIGMA = 0.0
TRAINING_SEED = 0
#: Dataset whose truncation reference is subtracted.
DATASET = "nc"


def truncation_reference(path: Path, dataset: str, rank: int, bands) -> dict:
    """Band means of the rank-``rank`` truncation reference of one dataset."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = next((s for s in payload["summaries"] if s["dataset"] == dataset), None)
    if summary is None:
        raise SystemExit(f"{path.name} has no summary for dataset {dataset!r}")
    table = summary["table"].get(str(rank))
    if table is None:
        raise SystemExit(
            f"{path.name} has no rank {rank} for dataset {dataset!r}; "
            "run applications.statistics.truncation_reference_audit first"
        )
    return {band: float(table["bands"][band]["mean"]) for band in bands}


def select_records(records: list[dict], model: str, sensor_count: int,
                   sigma: float, seed: int) -> list[dict]:
    """Records of one (estimator, sensor count, noise level, seed) run."""
    return [
        record for record in records
        if record["model"] == model
        and int(record["sensor_count"]) == int(sensor_count)
        and abs(float(record["noise_sigma"]) - float(sigma)) < 1e-12
        and int(record["training_seed"]) == int(seed)
    ]


def model_block(records: list[dict], model: str, reference: dict, bands) -> dict:
    """Mean estimator error, mean truncation term and excess of one estimator."""
    rows = select_records(records, model, SENSOR_COUNT, NOISE_SIGMA, TRAINING_SEED)
    if not rows:
        raise SystemExit(
            f"no records for {model} M={SENSOR_COUNT} sigma={NOISE_SIGMA} "
            f"seed={TRAINING_SEED}: run applications.statistics.band_error_records first"
        )
    totals = {}
    truncations = {}
    for band in bands:
        totals[band] = float(np.mean([r["band_errors"][band]["total"] for r in rows]))
        truncations[band] = float(
            np.mean([r["band_errors"][band]["truncation"] for r in rows])
        )
    return {
        "n_snapshots": len(rows),
        "total_errors": totals,
        "truncation_errors": truncations,
        "delta": {band: totals[band] - reference[band] for band in bands},
    }


def verify(table: dict, bands, tolerance: float = 1e-6) -> int:
    """Band-by-band comparison with the frozen artifact of the same quantity."""
    if not REFERENCE.exists():
        print(f"  [skip] {REFERENCE.relative_to(ROOT).as_posix()} is not available")
        return 0

    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))["delta_excess_band_error"]
    print(f"\ncomparison with {REFERENCE.relative_to(ROOT).as_posix()}")
    print(f"   {'model':<7}{'band':<6}{'this run':>12}{'frozen':>12}{'difference':>13}")
    worst = 0.0
    above = 0
    for model in MODELS:
        for band in bands:
            new = table[model]["delta"][band]
            old = frozen.get(model, {}).get("delta", {}).get(band)
            difference = abs(new - old) if old is not None else float("nan")
            worst = max(worst, difference)
            if difference > tolerance:
                above += 1
            print(f"   {model:<7}{band:<6}{new:>12.6f}{old if old is not None else float('nan'):>12.6f}"
                  f"{difference:>13.2e}")
    print(f"\n   largest difference {worst:.3e}; {above} bands above {tolerance:.0e}")
    if above == 0:
        print("   OK: reproduces the frozen artifact")
        return 0
    print("   The bands above the tolerance are the definitional differences "
          "documented in the module docstring.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--truncation", type=Path, default=TRUNCATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = get_config()
    bands = cfg.bands
    print("=" * 60)
    print(f"excess band error: M={SENSOR_COUNT}, sigma={NOISE_SIGMA}, "
          f"seed={TRAINING_SEED}, rank {cfg.pod_rank}")
    print("=" * 60)

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    records = payload["records"]
    print(f"   {len(records)} records from {args.records.relative_to(ROOT).as_posix()}")
    recorded_rank = payload.get("config", {}).get("pod_rank")
    if recorded_rank is not None and int(recorded_rank) != cfg.pod_rank:
        raise SystemExit(
            f"the record set was built at rank {recorded_rank}, "
            f"the configuration says rank {cfg.pod_rank}"
        )

    reference = truncation_reference(args.truncation, DATASET, cfg.pod_rank, bands)
    print(f"   truncation reference from {args.truncation.relative_to(ROOT).as_posix()}")
    print(f"   {DATASET} rank {cfg.pod_rank}: "
          + ", ".join(f"{band}={reference[band]:.6e}" for band in bands))

    table = {model: model_block(records, model, reference, bands) for model in MODELS}

    output = {
        "description": (
            "Excess per-band error over the rank-128 POD truncation reference "
            f"(M={SENSOR_COUNT}, sigma={NOISE_SIGMA}, seed={TRAINING_SEED}); "
            "delta_excess_band_error[model][delta][band] = "
            "mean_snapshots(total(band)) - reference(band)"
        ),
        "config": {
            "dataset": DATASET,
            "sensor_count": SENSOR_COUNT,
            "noise_sigma": NOISE_SIGMA,
            "training_seed": TRAINING_SEED,
            "pod_rank": cfg.pod_rank,
            "bands": list(bands),
            "tau": cfg.tau,
        },
        "truncation_reference": {
            "source": "artifacts/statistics/truncation_reference_audit.json",
            "dataset": DATASET,
            "rank": cfg.pod_rank,
            "band_mean": reference,
        },
        "delta_excess_band_error": table,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"\n   {'model':<7}" + "".join(f"{band:>12}" for band in bands))
    for model in MODELS:
        print(f"   {model:<7}" + "".join(
            f"{table[model]['delta'][band]:>12.6f}" for band in bands))
    print(f"\n[OK] wrote {args.output.relative_to(ROOT).as_posix()}")

    if args.verify:
        return verify(table, bands, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

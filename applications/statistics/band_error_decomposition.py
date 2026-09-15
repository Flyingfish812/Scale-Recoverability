"""Band-wise error decomposition of every trained reconstruction.

For each configuration (model, sensor count M, noise level sigma, training run)
this module compares the reconstruction with the rank-r POD truncation of the
same target field and reports, band by band,

    total(b)      = ‖W_b(u) − W_b(û)‖₂      / ‖W_b(u)‖₂
    truncation(b) = ‖W_b(u) − W_b(u_ref)‖₂  / ‖W_b(u)‖₂
    prediction(b) = ‖W_b(u_ref) − W_b(û)‖₂  / ‖W_b(u)‖₂

together with two scalars per output:

    global_error            relative L2 error over the full state (both velocity
                            components) — the "GER" reported throughout the paper
    truncation_global_error the same quantity for the rank-r POD truncation
                            itself, i.e. the representation floor of the snapshot

The band-wise terms and the scale indices (S_full) are evaluated on the
streamwise component u; the global errors use the full two-component state.

Inputs
    data/cylinder2d_q1.npy            raw snapshots
    artifacts/pod_bases/...           rank-r POD basis (both components)
    artifacts/<estimator runs>/       one test_raw.npz per configuration
Output
    artifacts/statistics/band_error_decomposition.json

Usage
    python -m applications.statistics.band_error_decomposition
    python -m applications.statistics.band_error_decomposition --verify

``--verify`` compares the recomputed records with the frozen baseline of the
submitted manuscript (band errors and S_full must match exactly; the global
error is reported for information because its definition was unified here).
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

from luna.core.constants import (  # noqa: E402
    BANDS_CF,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    DEFAULT_WAVELET,
    TAU_DEFAULT,
)
from luna.data.io import load_npz  # noqa: E402
from luna.pod.truncation_reference import truncation_reference_reconstruct  # noqa: E402
from luna.wavelet.metrics import (  # noqa: E402
    band_error_decomposition,
    compute_S_full,
    global_error,
)
from features.training.estimator_runs import load_run, run_path  # noqa: E402

# ── Experiment definition (paper: Methods, "Reconstruction experiments") ──

SENSOR_COUNTS = [10, 15, 20, 30, 50]
NOISE_SIGMAS = [0.0, 0.001, 0.01, 0.1]
TRAINING_SEEDS = [0, 101, 202]
SNAPSHOTS_PER_CONFIGURATION = 50
POD_RANK = 128
POD_BUNDLE = ROOT / "artifacts" / "pod_bases" / "cylinder2d_q1" / "pod_base_bundle.npz"
OUTPUT = ROOT / "artifacts" / "statistics" / "band_error_decomposition.json"
BASELINE = ROOT / "artifacts" / "derived" / "main" / "statistics" / "three_layer_fixed.json"

# ── Where the estimator runs live ───────────────────────────────────────
# The run layout and the unit convention of each estimator family are defined
# once, in features.training.estimator_runs, and shared by all producers.


def decompose_configuration(
    model: str,
    sensor_count: int,
    sigma: float,
    seed: int,
    run: Path,
    basis: np.ndarray,
    mean: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> list[dict]:
    """Records of the first ``SNAPSHOTS_PER_CONFIGURATION`` snapshots of a run."""
    target, recon = load_run(run)
    n = min(SNAPSHOTS_PER_CONFIGURATION, target.shape[0])
    records = []
    for index in range(n):
        truth = np.transpose(target[index], (1, 2, 0))
        prediction = np.transpose(recon[index], (1, 2, 0))
        reference = truncation_reference_reconstruct(truth, basis, mean, spatial_shape=spatial_shape)

        bands = band_error_decomposition(truth[:, :, 0], prediction[:, :, 0], reference[:, :, 0])
        records.append({
            "model": model,
            "sensor_count": sensor_count,
            "noise_sigma": sigma,
            "training_seed": seed,
            "snapshot_index": index,
            "global_error": global_error(truth, prediction),
            "truncation_global_error": global_error(truth, reference),
            "s_full": compute_S_full(truth[:, :, 0], prediction[:, :, 0], TAU_DEFAULT),
            "band_errors": bands,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true",
                        help="compare with the frozen manuscript baseline")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing output file without recomputing")
    args = parser.parse_args()

    if args.verify_only:
        records = json.loads(args.output.read_text(encoding="utf-8"))["records"]
        return verify(records)

    bundle = load_npz(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)
    mean = np.asarray(bundle["mean_field"], dtype=np.float64)
    height, width, channels = mean.shape
    basis = basis.reshape(basis.shape[0], -1)[:POD_RANK]
    mean = mean.ravel()
    print(f"POD basis: rank {basis.shape[0]}, grid {height}x{width}, {channels} components")

    started = time.time()
    records: list[dict] = []
    missing: list[str] = []
    for model in ("mlp", "vcnn", "ridge"):
        for sensor_count in SENSOR_COUNTS:
            for sigma in NOISE_SIGMAS:
                for seed in TRAINING_SEEDS:
                    if model == "ridge" and seed != TRAINING_SEEDS[0]:
                        continue  # deterministic estimator: one run only
                    run = run_path(model, sensor_count, sigma, seed)
                    if run is None:
                        missing.append(f"{model} M={sensor_count} sigma={sigma} seed={seed}")
                        continue
                    records += decompose_configuration(
                        model, sensor_count, sigma, seed, run,
                        basis, mean, (height, width, channels),
                    )
        print(f"  {model}: cumulative records = {len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "band-wise error decomposition of every reconstruction against the rank-128 POD truncation",
        "config": {
            "dataset": "nc",
            "sensor_counts": SENSOR_COUNTS,
            "noise_sigmas": NOISE_SIGMAS,
            "training_seeds": TRAINING_SEEDS,
            "snapshots_per_configuration": SNAPSHOTS_PER_CONFIGURATION,
            "pod_rank": POD_RANK,
            "wavelet": DEFAULT_WAVELET,
            "level": DEFAULT_LEVEL,
            "mode": DEFAULT_MODE,
            "tau": TAU_DEFAULT,
            "scale_component": "streamwise",
            "global_error_component": "both",
        },
        "records": records,
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output.relative_to(ROOT)} "
          f"({time.time() - started:.0f} s)")
    if missing:
        print(f"WARNING: {len(missing)} configurations not found, e.g. {missing[:3]}")

    if args.verify:
        return verify(records)
    return 0


def verify(records: list[dict]) -> int:
    """Compare against the frozen baseline of the submitted manuscript.

    The band-wise quantities of the POD-coefficient and convolutional estimators
    must reproduce the baseline exactly (to floating-point storage precision).
    The Ridge records are excluded: the baseline mixed in a deprecated
    AdamW-trained Ridge, which this refactor replaces with the closed-form Ridge
    used everywhere else in the paper.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]
    # The baseline stores three training runs per configuration in row blocks,
    # with the seed column left at zero; rows are ordered [seed0, seed101, seed202].
    expected: dict[tuple, list[dict]] = {}
    for row in baseline:
        key = (row["model_type"], row["mask_num"], row["noise_sigma"], row["sample_idx"])
        expected.setdefault(key, []).append(row)

    tolerance = 1e-6
    worst = {quantity: 0.0 for quantity in ("band error", "truncation error")}
    worst_s_full = 0
    compared = 0
    skipped = {"ridge": 0, "not in baseline": 0}
    for record in records:
        key = (record["model"], record["sensor_count"], record["noise_sigma"], record["snapshot_index"])
        rows = expected.get(key)
        if not rows:
            skipped["not in baseline"] += 1
            continue
        if record["model"] == "ridge":
            skipped["ridge"] += 1
            continue
        seed_order = TRAINING_SEEDS.index(record["training_seed"])
        reference = rows[min(seed_order, len(rows) - 1)]
        worst_s_full = max(worst_s_full, abs(record["s_full"] - reference["S_full_total"]))
        for band in BANDS_CF:
            worst["band error"] = max(worst["band error"],
                                      abs(record["band_errors"][band]["total"] - reference[f"E_total_{band}"]))
            worst["truncation error"] = max(worst["truncation error"],
                                            abs(record["band_errors"][band]["truncation"] - reference[f"E_trunc_{band}"]))
        compared += 1

    print("\nverification against the frozen baseline")
    print(f"  records compared            : {compared}")
    print(f"  max |band error difference| : {worst['band error']:.3e}")
    print(f"  max |truncation difference| : {worst['truncation error']:.3e}")
    print(f"  max |S_full difference|     : {worst_s_full}")
    print(f"  skipped                     : ridge {skipped['ridge']}, "
          f"not in baseline {skipped['not in baseline']}")
    ok = max(worst.values()) < tolerance and worst_s_full == 0
    print("  OK: band-wise quantities reproduce the baseline"
          if ok else "  FAILED: band-wise quantities differ from the baseline")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

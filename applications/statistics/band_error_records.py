"""Full band-error record set: every configuration, every test snapshot.

Extends the 50-snapshot sample of ``band_error_decomposition`` to all 300 test
snapshots of each training run, which is what the cross-model figures and the
cross-model scale-recoverability means are built from.

For each record the module reports, band by band,

    total(b)      = ‖W_b(u) − W_b(û)‖₂      / ‖W_b(u)‖₂
    truncation(b) = ‖W_b(u) − W_b(u_ref)‖₂  / ‖W_b(u)‖₂
    prediction(b) = ‖W_b(u_ref) − W_b(û)‖₂  / ‖W_b(u)‖₂

together with the global error of the reconstruction and of the rank-r POD
truncation, both on the full two-component state, and the index S_full computed
on the streamwise component. Fields are in physical units.

Because the truncation reference depends only on (training run, snapshot), it is
computed once per run and reused across the M x sigma grid.

Inputs
    data/cylinder2d_q1.npy, artifacts/pod_bases/..., artifacts/<estimator runs>/
Output
    artifacts/statistics/band_error_records.json

Usage
    python -m applications.statistics.band_error_records --jobs 9
    python -m applications.statistics.band_error_records --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
from luna.wavelet.transform import decompose_field_2d  # noqa: E402

from applications.statistics.band_error_decomposition import (  # noqa: E402
    NOISE_SIGMAS,
    POD_BUNDLE,
    POD_RANK,
    SENSOR_COUNTS,
    TRAINING_SEEDS,
    load_run,
    run_path,
)

OUTPUT = ROOT / "artifacts" / "statistics" / "band_error_records.json"
BASELINE = ROOT / "artifacts" / "derived" / "main" / "statistics" / "three_layer_errors_full.json"


def _basis() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    bundle = load_npz(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)
    mean = np.asarray(bundle["mean_field"], dtype=np.float64)
    shape = mean.shape
    return basis.reshape(basis.shape[0], -1)[:POD_RANK], mean.ravel(), shape


def process_run(args: tuple[str, int, int, float, int]) -> list[dict]:
    """Records of one (model, M, sigma, seed) run over all its snapshots."""
    model, sensor_count, sigma, seed = args
    run = run_path(model, sensor_count, sigma, seed)
    if run is None:
        return []
    basis, mean, shape = _basis()
    target, reconstruction = load_run(run)
    records = []
    for index in range(target.shape[0]):
        truth = np.transpose(target[index], (1, 2, 0))
        prediction = np.transpose(reconstruction[index], (1, 2, 0))
        reference = truncation_reference_reconstruct(truth, basis, mean, spatial_shape=shape)
        records.append({
            "model": model,
            "sensor_count": sensor_count,
            "noise_sigma": sigma,
            "training_seed": seed,
            "snapshot_index": index,
            "global_error": global_error(truth, prediction),
            "truncation_global_error": global_error(truth, reference),
            "s_full": compute_S_full(truth[:, :, 0], prediction[:, :, 0], TAU_DEFAULT),
            "band_errors": band_error_decomposition(
                truth[:, :, 0], prediction[:, :, 0], reference[:, :, 0]
            ),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        return verify(payload["records"])

    tasks = [(model, sensor_count, sigma, seed)
             for model in ("mlp", "vcnn", "ridge", "gappy")
             for sensor_count in SENSOR_COUNTS
             for sigma in NOISE_SIGMAS
             for seed in TRAINING_SEEDS
             if not (model in ("ridge", "gappy") and seed != TRAINING_SEEDS[0])]

    started = time.time()
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for count, chunk in enumerate(pool.map(process_run, tasks), start=1):
            records += chunk
            if count % 20 == 0:
                print(f"  {count}/{len(tasks)} runs, {len(records)} records, "
                      f"{time.time() - started:.0f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "band-wise error decomposition over every test snapshot",
        "config": {
            "dataset": "nc",
            "sensor_counts": SENSOR_COUNTS,
            "noise_sigmas": NOISE_SIGMAS,
            "training_seeds": TRAINING_SEEDS,
            "pod_rank": POD_RANK,
            "wavelet": DEFAULT_WAVELET,
            "level": DEFAULT_LEVEL,
            "mode": DEFAULT_MODE,
            "tau": TAU_DEFAULT,
            "fields": "physical units",
            "scale_component": "streamwise",
            "global_error_component": "both",
        },
        "records": records,
    }), encoding="utf-8")
    print(f"wrote {len(records)} records ({time.time() - started:.0f} s)")
    if args.verify:
        return verify(records)
    return 0


def verify(records: list[dict]) -> int:
    """Compare the band-wise terms with the frozen baseline of the manuscript.

    The baseline stores three training runs in row blocks ordered
    [seed0, seed101, seed202] and leaves the seed column at zero.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    rows = baseline["results"] if isinstance(baseline, dict) else baseline
    expected: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model_type"], row["mask_num"], row["noise_sigma"])
        expected.setdefault(key, []).append(row)

    worst = 0.0
    worst_s_full = 0
    compared, skipped = 0, 0
    for record in records:
        if record["model"] == "ridge":
            skipped += 1
            continue
        rows = expected.get((record["model"], record["sensor_count"], record["noise_sigma"]))
        if rows is None:
            skipped += 1
            continue
        offset = TRAINING_SEEDS.index(record["training_seed"]) * 300 + record["snapshot_index"]
        if offset >= len(rows):
            skipped += 1
            continue
        reference = rows[offset]
        worst_s_full = max(worst_s_full, abs(record["s_full"] - reference["S_full_total"]))
        for band in BANDS_CF:
            worst = max(worst,
                        abs(record["band_errors"][band]["total"] - reference[f"E_total_{band}"]),
                        abs(record["band_errors"][band]["truncation"] - reference[f"E_trunc_{band}"]))
        compared += 1

    print("\nverification against the frozen baseline")
    print(f"  records compared            : {compared} (skipped {skipped})")
    print(f"  max |band error difference| : {worst:.3e}")
    print(f"  max |S_full difference|     : {worst_s_full}")
    ok = worst < 1e-6 and worst_s_full == 0
    print("  OK: reproduces the baseline" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

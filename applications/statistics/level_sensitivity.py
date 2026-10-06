"""Sensitivity of the scale-recoverability index to the wavelet level.

The index counts bands, so a fair comparison across decomposition levels uses the normalised value S_full / (L + 1). This module evaluates that quantity for L = 3, 4, 5 on one representative run and reports, per level, the band errors and the index.

Two band-error definitions are computed:

    spatial     the band-limited spatial components, as everywhere else in the
                paper (band error = ‖W_b(u) − W_b(û)‖₂ / ‖W_b(u)‖₂, with S_full
                contiguous from the coarsest band)
    coefficient the wavelet coefficients themselves (the earlier diagnostic,
                kept for comparison)

Inputs
    artifacts/<estimator runs>/    one test_raw.npz per configuration
Output
    artifacts/statistics/level_sensitivity.json

Usage
    python -m applications.statistics.level_sensitivity
    python -m applications.statistics.level_sensitivity --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.core.constants import (  # noqa: E402
    DEFAULT_MODE,
    DEFAULT_WAVELET,
    EPS,
    TAU_DEFAULT,
)
from luna.wavelet.metrics import contiguous_recoverable_index  # noqa: E402

from applications.statistics.band_error_decomposition import load_run, run_path  # noqa: E402

OUTPUT = ROOT / "artifacts" / "statistics" / "level_sensitivity.json"
# : Reference values of the earlier level analysis, kept for the --verify path.
REFERENCE = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "mechanism_analysis" / "level_sensitivity.json")
MODEL, SENSOR_COUNT, SIGMA, SEED = "mlp", 20, 0.0, 0
N_SNAPSHOTS = 50
LEVELS = (3, 4, 5)


def band_names(level: int) -> list[str]:
    return [f"A{level}"] + [f"W{i}" for i in range(level, 0, -1)]


def spatial_components(field: np.ndarray, level: int) -> list[np.ndarray]:
    """Band-limited spatial components, coarse to fine."""
    coeffs = pywt.wavedec2(field, DEFAULT_WAVELET, level=level, mode=DEFAULT_MODE)
    height, width = field.shape
    components = []
    for index in range(len(coeffs)):
        kept = [np.zeros_like(c) if not isinstance(c, tuple)
                else tuple(np.zeros_like(p) for p in c) for c in coeffs]
        kept[index] = coeffs[index]
        reconstructed = pywt.waverec2(kept, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE)
        components.append(np.asarray(reconstructed[:height, :width], dtype=np.float64))
    return components


def coefficient_components(field: np.ndarray, level: int) -> list[np.ndarray]:
    """Wavelet coefficients, with the three detail sub-bands combined in magnitude."""
    coeffs = pywt.wavedec2(field, DEFAULT_WAVELET, level=level, mode=DEFAULT_MODE)
    components = [np.asarray(coeffs[0], dtype=np.float64)]
    for detail in coeffs[1:]:
        components.append(np.sqrt(sum(np.asarray(part, dtype=np.float64) ** 2 for part in detail)))
    return components


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm((reference - estimate).ravel())
                 / (np.linalg.norm(reference.ravel()) + EPS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        return verify(payload)

    run = run_path(MODEL, SENSOR_COUNT, SIGMA, SEED)
    target, reconstruction = load_run(run)
    n = min(N_SNAPSHOTS, target.shape[0])

    results = {}
    for level in LEVELS:
        names = band_names(level)
        spatial_errors = {name: [] for name in names}
        coefficient_errors = {name: [] for name in names}
        spatial_index, coefficient_index = [], []
        for snapshot in range(n):
            truth = target[snapshot, 0]
            estimate = reconstruction[snapshot, 0]
            for label, extractor, store, index_store in (
                ("spatial", spatial_components, spatial_errors, spatial_index),
                ("coefficient", coefficient_components, coefficient_errors, coefficient_index),
            ):
                bands = extractor(truth, level)
                predicted = extractor(estimate, level)
                errors = [relative_l2(b, p) for b, p in zip(bands, predicted)]
                for name, value in zip(names, errors):
                    store[name].append(value)
                index_store.append(contiguous_recoverable_index(np.array(errors), TAU_DEFAULT))
        results[f"level_{level}"] = {
            "n_snapshots": n,
            "bands": names,
            "spatial": {
                "s_full_mean": float(np.mean(spatial_index)),
                "s_full_std": float(np.std(spatial_index)),
                "s_full_normalised": float(np.mean(spatial_index) / (level + 1)),
                "per_band_mean_error": {k: float(np.mean(v)) for k, v in spatial_errors.items()},
            },
            "coefficient": {
                "s_full_mean": float(np.mean(coefficient_index)),
                "s_full_std": float(np.std(coefficient_index)),
                "s_full_normalised": float(np.mean(coefficient_index) / (level + 1)),
                "per_band_mean_error": {k: float(np.mean(v)) for k, v in coefficient_errors.items()},
            },
        }
        r = results[f"level_{level}"]
        print(f"  level {level}: S/(L+1) spatial={r['spatial']['s_full_normalised']:.3f} "
              f"coefficient={r['coefficient']['s_full_normalised']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "sensitivity of the scale-recoverability index to the wavelet level",
        "config": {
            "model": MODEL, "sensor_count": SENSOR_COUNT, "noise_sigma": SIGMA,
            "training_seed": SEED, "n_snapshots": n,
            "wavelet": DEFAULT_WAVELET, "tau": TAU_DEFAULT,
        },
        "levels": results,
    }), encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)}")
    if args.verify:
        return verify({"levels": results})
    return 0


def verify(payload: dict) -> int:
    """Compare the coefficient-domain variant with the earlier diagnostic."""
    if not REFERENCE.exists():
        print("\nbaseline not available, skipping verification")
        return 0
    baseline = json.loads(REFERENCE.read_text(encoding="utf-8"))
    print("\nverification against the earlier diagnostic (coefficient domain)")
    worst_index, worst_error = 0.0, 0.0
    for key, expected in baseline.items():
        got = payload["levels"].get(key)
        if got is None:
            continue
        worst_index = max(worst_index, abs(got["coefficient"]["s_full_mean"] - expected["s_full_mean"]))
        for band, value in expected["per_band_mean_error"].items():
            worst_error = max(worst_error, abs(got["coefficient"]["per_band_mean_error"][band] - value))
    print(f"  max |S_full mean difference|  : {worst_index:.3e}")
    print(f"  max |band error difference|    : {worst_error:.3e}")
    ok = worst_index < 1e-6 and worst_error < 1e-6
    print("  OK: reproduces the earlier diagnostic" if ok else "  DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Analytical multiscale benchmark: does the index recover the prescribed scales?

The benchmark builds synthetic fields with known coarse-scale content, removes
one or two prescribed Fourier-carrier groups, and asks whether the wavelet index
reports the change. Because the content is prescribed, the correct value of
S_full is known in advance, which makes the benchmark an absolute check on the
index rather than a relative comparison between models.

Two blocks are produced:

    cases      the six benchmark cases, with the expected and the measured
               scale count, the global error and the per-band errors
    real_nc    the same wavelet index applied to the NC reconstructions, for
               each wavelet family, which is the sensitivity check of the
               transform choice

Inputs
    nothing (the analytical fields are constructed here)
    artifacts/statistics/wavelet_sensitivity.json  (the transform-sensitivity block)
Output
    artifacts/statistics/analytical_benchmark.json

Usage
    python -m applications.statistics.analytical_benchmark
    python -m applications.statistics.analytical_benchmark --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.benchmarks.analytical_wake import (  # noqa: E402
    WakeParams,
    case_metrics,
    generate_ensemble,
    scale_u_components,
    snapshot,
)
from luna.core.constants import (  # noqa: E402
    BANDS_CF,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    DEFAULT_WAVELET,
    TAU_DEFAULT,
)
from luna.pod.band_pod import fit_band_pod  # noqa: E402
from luna.wavelet.transform import decompose_field_2d  # noqa: E402

SOURCE = ROOT / "artifacts" / "statistics"
OUTPUT = ROOT / "artifacts" / "statistics" / "analytical_benchmark.json"
#: Consolidated values of the submitted manuscript, frozen for --verify.
SUBMITTED = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "p0_consolidated.json")

#: Snapshots used for the band-POD fit behind the coherent-only index, and for the
#: ensemble statistics. The two groups must not overlap.
N_TRAIN_POD = 200
N_TEST = 100
SEED_OFFSET = 1000

WAVELETS = ["haar", "db2", "db4", "sym4", "coif1"]
MODELS = ["ridge", "mlp", "vcnn"]
CASES = ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3", "E1_del_W1_only",
         "E2_partial_W3"]
#: Scale count each case must report, by construction.
EXPECTED_SCALES = {"A_full": 5, "B_del_W1": 4, "C_del_W1W2": 3, "D_del_W3": 2,
                   "E1_del_W1_only": 4, "E2_partial_W3": 2}
#: Energy share of each band in the NC test fields, for reference.
NC_BAND_ENERGY_FRACTIONS = {"A4": 0.966, "W4": 0.029, "W3": 0.004, "W2": 0.001,
                            "W1": 0.000}

#: Fourier carriers carried by each scale group of the construction (band 1 = A4).
CARRIER_WAVENUMBERS = {
    1: [(1, 1), (2, 1), (3, 2)],
    2: [(6, 4), (7, 4)],
    3: [(14, 7)],
    4: [(28, 14), (24, 12)],
    5: [(48, 32), (56, 28)],
}


def expected_scales(case: str) -> int:
    """Scale count the construction prescribes for one case."""
    return EXPECTED_SCALES[case]


def ensemble_case_stats(params: WakeParams, n_snapshots: int, seed_offset: int,
                        band_pod: dict | None, tau: float) -> dict:
    """Mean and spread of every metric over an ensemble of analytical snapshots."""
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    accumulated = {case: {"GER": [], "S_full": [], "S_coh": [],
                          "E_direct": {band: [] for band in BANDS_CF}}
                   for case in CASES}

    for i in range(n_snapshots):
        seed = seed_offset + i
        target = snapshot(x, y, params, seed)
        components = scale_u_components(x, y, params, seed)
        reconstructions = case_metrics(target, components, band_pod=band_pod, tau=tau)
        for case in CASES:
            accumulated[case]["GER"].append(reconstructions[case]["GER"])
            accumulated[case]["S_full"].append(reconstructions[case]["S_full"])
            if reconstructions[case]["S_coh"] is not None:
                accumulated[case]["S_coh"].append(reconstructions[case]["S_coh"])
            for band in BANDS_CF:
                accumulated[case]["E_direct"][band].append(
                    reconstructions[case]["E_direct"][band])

    statistics = {}
    for case in CASES:
        entry = {
            "GER_mean": float(np.mean(accumulated[case]["GER"])),
            "GER_std": float(np.std(accumulated[case]["GER"])),
            "S_full_mean": float(np.mean(accumulated[case]["S_full"])),
            "S_full_std": float(np.std(accumulated[case]["S_full"])),
            "S_full_all": [int(v) for v in accumulated[case]["S_full"]],
            "S_full_correct_frac": float(np.mean(
                [v == expected_scales(case) for v in accumulated[case]["S_full"]])),
            "E_direct_mean": {band: float(np.mean(accumulated[case]["E_direct"][band]))
                              for band in BANDS_CF},
            "E_direct_std": {band: float(np.std(accumulated[case]["E_direct"][band]))
                             for band in BANDS_CF},
        }
        if accumulated[case]["S_coh"]:
            entry["S_coh_mean"] = float(np.mean(accumulated[case]["S_coh"]))
            entry["S_coh_std"] = float(np.std(accumulated[case]["S_coh"]))
        statistics[case] = entry
    return statistics


def run_benchmark(n_train: int = N_TRAIN_POD, n_test: int = N_TEST,
                  seed_offset: int = SEED_OFFSET, tau: float = TAU_DEFAULT) -> dict:
    """Construct the benchmark fields and measure the prescribed scale counts."""
    params = WakeParams()
    train_fields, _ = generate_ensemble(n_train, params, seed_offset=0)
    band_pod = fit_band_pod(train_fields, pod_energy_threshold=0.99)
    statistics = ensemble_case_stats(params, n_test, seed_offset, band_pod, tau)

    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    target = snapshot(x, y, params, 0)
    components = scale_u_components(x, y, params, 0)
    representative = case_metrics(target, components, band_pod=band_pod, tau=tau)

    decomposed = decompose_field_2d(target)
    total_energy = float(np.sum(target ** 2))
    return {
        "benchmark": "NC-inspired analytical multiscale wake",
        "grid": {"H": params.H, "W": params.W, "x0": params.x0, "y0": params.y0},
        "wavelet": {"family": DEFAULT_WAVELET, "level": DEFAULT_LEVEL,
                    "mode": DEFAULT_MODE},
        "tau": tau,
        "params": params.__dict__,
        "wavenum": {str(band): carriers
                    for band, carriers in CARRIER_WAVENUMBERS.items()},
        "n_train_pod": n_train,
        "n_test": n_test,
        "target_band_energy_fractions": {
            band: float(np.sum(decomposed[band] ** 2) / total_energy)
            for band in BANDS_CF},
        "representative_seed0": representative,
        "ensemble_stats": statistics,
    }


def consolidate(benchmark: dict, sensitivity: dict) -> dict:
    """Reduce the raw results to the numbers the paper quotes."""
    statistics = benchmark["ensemble_stats"]
    representative = benchmark["representative_seed0"]
    return {
        "meta": {
            "grid": benchmark.get("grid"),
            "wavelet": benchmark.get("wavelet"),
            "tau": benchmark.get("tau"),
            "params": benchmark.get("params"),
            "wavenumber": benchmark.get("wavenum"),
        },
        "cases": {
            case: {
                "expected_S_full": EXPECTED_SCALES[case],
                "GER_mean": statistics[case]["GER_mean"],
                "GER_std": statistics[case]["GER_std"],
                "S_full_mean": statistics[case]["S_full_mean"],
                "S_full_std": statistics[case]["S_full_std"],
                "S_full_correct_frac": statistics[case]["S_full_correct_frac"],
                "S_coh_mean": statistics[case].get("S_coh_mean"),
                "E_direct_mean": statistics[case]["E_direct_mean"],
                "E_direct_std": statistics[case]["E_direct_std"],
            }
            for case in CASES
        },
        "representative_run": {
            case: {
                "GER": representative[case]["GER"],
                "S_full": representative[case]["S_full"],
                "S_coh": representative[case]["S_coh"],
                "E_direct": representative[case]["E_direct"],
            }
            for case in CASES
        },
        "equal_ger_pair": {
            "E1_GER": statistics["E1_del_W1_only"]["GER_mean"],
            "E2_GER": statistics["E2_partial_W3"]["GER_mean"],
            "E1_S_full": statistics["E1_del_W1_only"]["S_full_mean"],
            "E2_S_full": statistics["E2_partial_W3"]["S_full_mean"],
        },
        "target_band_energy_fractions": benchmark.get("target_band_energy_fractions"),
        "nc_band_energy_fractions": NC_BAND_ENERGY_FRACTIONS,
        "wavelet_sensitivity": {
            "representative_case": sensitivity.get("representative_case"),
            "real_nc": {
                wavelet: {
                    model: {
                        "GER_mean": sensitivity["real_nc"][wavelet][model]["GER_mean"],
                        "S_full_mean": sensitivity["real_nc"][wavelet][model]["S_full_mean"],
                        "S_full_mode": sensitivity["real_nc"][wavelet][model]["S_full_mode"],
                        "S_full_distribution":
                            sensitivity["real_nc"][wavelet][model]["S_full_dist"],
                        "S_coh_mode": sensitivity["real_nc"][wavelet][model]["S_coh_mode"],
                        "E_direct_mean":
                            sensitivity["real_nc"][wavelet][model]["E_direct_mean"],
                    }
                    for model in MODELS
                }
                for wavelet in WAVELETS
            },
            "benchmark_detection": {
                wavelet: {case: sensitivity["analytical_benchmark"][wavelet][case]["correct_frac"]
                          for case in CASES}
                for wavelet in WAVELETS
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="compare with the consolidated values of the manuscript")
    args = parser.parse_args()

    start = time.time()
    print("== analytical multiscale benchmark")
    print(f"   constructing {N_TEST} snapshots and the band-POD basis ...")
    benchmark = run_benchmark()
    sensitivity = json.loads((SOURCE / "wavelet_sensitivity.json").read_text(encoding="utf-8"))
    result = consolidate(benchmark, sensitivity)

    if args.verify:
        return _verify(result)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[OK] {OUTPUT.name} ({time.time() - start:.1f}s)")
    for case, values in result["cases"].items():
        print(f"   {case:16s} expected {values['expected_S_full']} "
              f"measured {values['S_full_mean']:.2f} "
              f"(correct in {100 * values['S_full_correct_frac']:.0f}% of fields), "
              f"GER {values['GER_mean']:.4f}")
    for wavelet, entry in result["wavelet_sensitivity"]["benchmark_detection"].items():
        worst = min(entry.values())
        print(f"   detection with {wavelet:6s}: worst case {100 * worst:.0f}%")
    return 0


def _close(got, want, tolerance: float = 1e-12) -> bool:
    """Compare two numbers, or two dictionaries of numbers, field by field."""
    if isinstance(want, dict):
        return (isinstance(got, dict) and set(got) == set(want)
                and all(_close(got[key], want[key], tolerance) for key in want))
    return abs(float(got) - float(want)) <= tolerance


def _verify(result: dict) -> int:
    """Compare with the consolidated values used by the submitted manuscript."""
    if not SUBMITTED.exists():
        print("   consolidated values not available; nothing to compare")
        return 1
    old = json.loads(SUBMITTED.read_text(encoding="utf-8"))
    print("== verification against the consolidated values")
    failures = 0
    for case, values in result["cases"].items():
        for field in ("GER_mean", "S_full_mean", "S_full_correct_frac",
                      "E_direct_mean"):
            got, want = values[field], old["p0_1"]["cases"][case][field]
            if not _close(got, want):
                print(f"   [FAIL] {case}.{field}: {got} != {want}")
                failures += 1
    for wavelet, entry in result["wavelet_sensitivity"]["benchmark_detection"].items():
        for case, value in entry.items():
            want = old["p0_2"]["analytical_benchmark_detection"][wavelet][case]
            if not _close(value, want):
                print(f"   [FAIL] detection {wavelet}/{case}: {value} != {want}")
                failures += 1
    for wavelet, models in result["wavelet_sensitivity"]["real_nc"].items():
        for model, values in models.items():
            for field in ("GER_mean", "S_full_mean", "S_full_mode", "E_direct_mean"):
                want = old["p0_2"]["real_nc"][wavelet][model][field]
                if not _close(values[field], want):
                    print(f"   [FAIL] {wavelet}/{model}.{field}: "
                          f"{values[field]} != {want}")
                    failures += 1
    print("   all checks passed" if not failures else f"   {failures} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

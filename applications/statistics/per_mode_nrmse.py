"""Per-mode NRMSE of the recovered POD coefficients, its energy-decile table and
the band recovery rates.

For every configuration (estimator, sensor count M, noise level sigma) the
reconstruction is projected onto the rank-128 POD basis and the error of modal
coefficient j is reported as

    e_j = sqrt( Σ_i (â_ij − a_ij)² / Σ_i a_ij² )

summed over the test snapshots i, with λ_j² / λ_1² as the modal energy. Because
the modal energy spans five orders of magnitude, the mechanism table of the
paper groups the 128 modes into energy deciles and reports the mean per-mode
NRMSE and the energy share of each group.

Two groupings of the same partition appear in the paper and are both reported
here, because they place the two 12-mode groups at opposite ends:

    rows            13 modes in groups 1-8 and 12 in groups 9-10, counted from
                    the least energetic mode (the order of the table)
    rows_descending the same partition counted from the most energetic mode,
                    which is the orientation of the frozen cross-table artifact
                    this module reproduces numerically
    rows_manuscript the partition of the frozen table again, counted from the
                    least energetic mode, which is the orientation the
                    manuscript prints: its two 12-mode groups are groups 1-2
                    and its group 10 holds the 13 most energetic modes

The second block counts how often each wavelet band error stays below the
tolerance tau over the complete 42,000-record set of the model comparison,
which is the recovery-rate table. It reads the record set produced by
``applications.statistics.band_error_records`` rather than recomputing the band
errors (the same way the pre-refactor analysis consumed its upstream record
file).

The three families are not equivalent inputs: the POD-coefficient estimators
store physical fields, the convolutional estimator stores normalised fields
(``load_run`` de-normalises them), and the linear estimator of the paper is the
closed-form Ridge of the trained-run layout. The frozen per-mode artifact was
built before the unit convention and the Ridge estimator were unified, so its
convolutional and linear configurations are reported but not reproduced; only
the MLP configurations, the modal energies and the recovery rates are expected
to agree with the frozen values.

Inputs
    artifacts/pod_bases/...                       rank-128 POD basis (both components)
    artifacts/<estimator runs>/                   one test_raw.npz per configuration
    artifacts/statistics/band_error_records.json  band errors of every test snapshot
Output
    artifacts/statistics/per_mode_nrmse.json

Usage
    python -m applications.statistics.per_mode_nrmse
    python -m applications.statistics.per_mode_nrmse --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from applications.statistics.band_error_decomposition import POD_BUNDLE, POD_RANK  # noqa: E402
from applications.statistics.modal_coefficient_error import DECILE_SIZES, modal_errors  # noqa: E402
from features.training.estimator_runs import load_run, noise_code, run_path  # noqa: E402

OUTPUT = OUT_DIR / "per_mode_nrmse.json"
RECORD_SET = OUT_DIR / "band_error_records.json"

#: Estimators of the sweep, in the order of the per-mode listing.
ESTIMATORS = ["mlp", "ridge", "vcnn"]
#: The per-mode listing covers one representative training run per configuration.
REPRESENTATIVE_SEED = 0
#: Decile table of the paper: the MLP run at the middle sensor count, clean input.
EXAMPLE_CONFIG = ("mlp", 20, 0.0)
#: Modes per group in the per-configuration decile listing, which splits the 128
#: modes into ten equal groups and leaves the eight least energetic ones out.
MODES_PER_LISTING_DECILE = 12
#: Frozen values of the manuscript, checked by ``--verify``.
FROZEN_CORRELATION = -0.9772
FROZEN_RECOVERY = {"A4": 73.4, "W4": 56.6, "W3": 48.4, "W2": 36.3, "W1": 30.8}
#: Energy share of the 13 most energetic modes (the manuscript's decile 10).
FROZEN_TOP_DECILE_ENERGY = 98.64


def per_mode_errors(
    run: Path, projection: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, int]:
    """Per-mode NRMSE of one run and the number of snapshots it covers."""
    target, reconstruction = load_run(run)
    errors = modal_errors(target, reconstruction, projection, mean)
    return errors, int(target.shape[0])


def decile_records(
    errors: np.ndarray, mode_energy: np.ndarray, descending: bool
) -> list[dict]:
    """Statistics of the 128 modes per energy decile of the paper's grouping.

    ``descending`` counts the groups from the most energetic mode, which is the
    orientation of the pre-refactor cross-table artifact; the default counts
    them from the least energetic mode. The energy share is reported both as a
    fraction of the total modal energy (the column of the paper) and as a
    percentage of λ_1² (the value of the pre-refactor artifact).
    """
    order = np.argsort(mode_energy)
    if descending:
        order = order[::-1]
    total_energy = float(mode_energy.sum())
    records, start = [], 0
    for index, size in enumerate(DECILE_SIZES, start=1):
        selected = order[start:start + size]
        start += size
        energy = mode_energy[selected]
        records.append({
            "decile": index,
            "n_modes": int(size),
            "energy_range_pct": [float(energy.min() * 100.0), float(energy.max() * 100.0)],
            "energy_share_pct": float(energy.sum() / total_energy * 100.0),
            "energy_sum_pct": float(energy.sum() * 100.0),
            "nrmse_mean": float(np.mean(errors[selected])),
            "nrmse_median": float(np.median(errors[selected])),
            "nrmse_q25": float(np.percentile(errors[selected], 25)),
            "nrmse_q75": float(np.percentile(errors[selected], 75)),
        })
    return records


def listing_decile_records(errors: np.ndarray, mode_energy: np.ndarray) -> list[dict]:
    """Equal-size decile listing stored with every configuration.

    The listing groups the modes into ten equal groups of 12, most energetic
    first, so the eight least energetic modes are not covered; it is kept for
    compatibility with the frozen per-mode artifact, which reports the same
    list for all 60 configurations.
    """
    order = np.argsort(mode_energy)[::-1]
    records = []
    for index in range(10):
        selected = order[index * MODES_PER_LISTING_DECILE:
                         (index + 1) * MODES_PER_LISTING_DECILE]
        energy = mode_energy[selected]
        records.append({
            "decile": index + 1,
            "n_modes": int(selected.size),
            "energy_range_pct": [float(energy.min() * 100.0), float(energy.max() * 100.0)],
            "energy_sum_pct": float(energy.sum() * 100.0),
            "nrmse_mean": float(np.mean(errors[selected])),
            "nrmse_median": float(np.median(errors[selected])),
            "nrmse_q25": float(np.percentile(errors[selected], 25)),
            "nrmse_q75": float(np.percentile(errors[selected], 75)),
        })
    return records


def decile_rows(records: list[dict]) -> dict[str, list]:
    """Decile records as the flat row format of the paper's table."""
    return {
        f"d{record['decile']}": [
            [float(record["energy_range_pct"][0]), float(record["energy_range_pct"][1])],
            record["nrmse_mean"],
            record["nrmse_median"],
            record["nrmse_q25"],
            record["nrmse_q75"],
            float(record["energy_share_pct"]),
        ]
        for record in records
    }


def summary_of_model(records: list[dict], model: str) -> dict | None:
    """Statistics of one estimator over its part of the configuration grid."""
    subset = [record for record in records if record["model_type"] == model]
    if not subset:
        return None
    correlations = [record["spearman_r"] for record in subset]
    means = [record["nrmse_mean"] for record in subset]
    example = next(
        (record for record in subset
         if record["mask_num"] == EXAMPLE_CONFIG[1]
         and abs(record["sigma_val"] - EXAMPLE_CONFIG[2]) < 1e-12),
        None,
    )
    return {
        "n_configs": len(subset),
        "spearman_r_mean": float(np.mean(correlations)),
        "spearman_r_std": float(np.std(correlations)),
        "spearman_r_min": float(np.min(correlations)),
        "spearman_r_max": float(np.max(correlations)),
        "nrmse_mean_all": float(np.mean(means)),
        "representative_config": {
            "mask_num": EXAMPLE_CONFIG[1],
            "sigma": EXAMPLE_CONFIG[2],
            "spearman_r": example["spearman_r"] if example else None,
            "spearman_p": example["spearman_p"] if example else None,
            "nrmse_mean": example["nrmse_mean"] if example else None,
            "nrmse_median": example["nrmse_median"] if example else None,
        } if example else None,
    }


def recovery_rates(bands: list[str], tau: float) -> dict:
    """Share of the records whose band error stays below tau.

    The record set pools the MLP and convolutional runs (three training seeds
    each) with the closed-form Ridge runs (one deterministic run per
    configuration), 42,000 records in total, and every record carries the same
    weight.
    """
    payload = json.loads(RECORD_SET.read_text(encoding="utf-8"))
    records = payload["records"]
    total = len(records)

    def rates(subset: list[dict]) -> dict[str, dict]:
        summary = {}
        for band in bands:
            passed = sum(
                1 for record in subset
                if record["band_errors"][band]["total"] <= tau
            )
            summary[band] = {
                "rate": float(passed / len(subset)) if subset else 0.0,
                "rate_pct": round(passed / len(subset) * 100.0, 1) if subset else 0.0,
                "passed": passed,
                "total": len(subset),
            }
        return summary

    overall = rates(records)
    drop = (overall["A4"]["rate_pct"] / overall["W1"]["rate_pct"]
            if overall["W1"]["rate_pct"] else float("inf"))
    return {
        "tau": tau,
        "total_records": total,
        "overall": overall,
        "per_model": {model: rates([r for r in records if r["model"] == model])
                      for model in ESTIMATORS},
        "drop_factor": round(drop, 1),
        "source": RECORD_SET.relative_to(ROOT).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true",
                        help="compare the reported values with the manuscript")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing output file without recomputing")
    args = parser.parse_args()

    if args.verify_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        return verify(payload["config_results"], payload["deciles"], payload["recovery_rates"])

    config = get_config()
    print("=" * 60)
    print("per-mode NRMSE, energy deciles and band recovery rates")
    print("=" * 60)

    bundle = np.load(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)
    mean = np.asarray(bundle["mean_field"], dtype=np.float64)
    singular_values = np.asarray(bundle["singular_values"], dtype=np.float64)
    projection = basis.reshape(basis.shape[0], -1)[:POD_RANK].T
    mean_flat = mean.ravel()
    mode_energy = (singular_values[:POD_RANK] ** 2) / (singular_values[0] ** 2)
    print(f"POD basis: rank {projection.shape[1]}, grid {mean.shape}")
    print(f"modal energy spans [{mode_energy[-1]:.3e}, {mode_energy[0]:.3f}] in units of lambda_1^2")

    started = time.time()
    results, missing = [], []
    for model in ESTIMATORS:
        for mask_num in config.M_values:
            for sigma in config.sigma_values:
                run = run_path(model, mask_num, sigma, REPRESENTATIVE_SEED)
                if run is None:
                    missing.append(f"{model} M={mask_num} sigma={sigma}")
                    continue
                errors, n_snapshots = per_mode_errors(run, projection, mean_flat)
                correlation, p_value = spearmanr(mode_energy, errors)
                results.append({
                    "model_type": model,
                    "mask_num": mask_num,
                    "sigma_code": noise_code(sigma),
                    "sigma_val": sigma,
                    "n_test_samples": n_snapshots,
                    "nrmse_per_mode": errors.tolist(),
                    "nrmse_mean": float(np.mean(errors)),
                    "nrmse_median": float(np.median(errors)),
                    "nrmse_std": float(np.std(errors)),
                    "nrmse_min": float(np.min(errors)),
                    "nrmse_max": float(np.max(errors)),
                    "spearman_r": float(correlation),
                    "spearman_p": float(p_value),
                    "deciles": listing_decile_records(errors, mode_energy),
                })
        print(f"  {model}: {len(results)} configurations done")

    example = next(
        record for record in results
        if record["model_type"] == EXAMPLE_CONFIG[0]
        and record["mask_num"] == EXAMPLE_CONFIG[1]
        and abs(record["sigma_val"] - EXAMPLE_CONFIG[2]) < 1e-12
    )
    example_errors = np.asarray(example["nrmse_per_mode"], dtype=np.float64)

    rows = decile_records(example_errors, mode_energy, descending=False)
    rows_descending = decile_records(example_errors, mode_energy, descending=True)
    total_energy_pct = float(mode_energy.sum() * 100.0)
    bottom = [record for record in rows_descending if record["decile"] > 6]
    deciles = {
        "desc": ("per-mode NRMSE grouped into energy deciles of the "
                 f"{EXAMPLE_CONFIG[0].upper()} run at M={EXAMPLE_CONFIG[1]}, "
                 f"sigma={EXAMPLE_CONFIG[2]}, {POD_RANK} modes"),
        "columns": ["energy_range_pct", "nrmse_mean", "nrmse_median",
                    "nrmse_q25", "nrmse_q75", "energy_share_pct"],
        "grouping": ("ascending energy order: 13 modes in groups 1-8 and 12 modes in "
                     "groups 9-10, so group 10 holds the 12 most energetic modes"),
        "decile_sizes": list(DECILE_SIZES),
        "rows": decile_rows(rows),
        "rows_descending": decile_rows(rows_descending),
        "rows_manuscript": decile_rows(rows_descending[::-1]),
        "descending_grouping": ("descending energy order: 13 modes in groups 1-8 and "
                                "12 modes in groups 9-10, the orientation of the frozen "
                                "cross-table artifact (its group 1 is the most energetic)"),
        "manuscript_grouping": ("ascending energy order: 12 modes in groups 1-2 and 13 "
                                "modes in groups 3-10, the orientation of the frozen "
                                "facts of the manuscript"),
        "structured": rows,
        "structured_descending": rows_descending,
        "energy_share_definition": "share of the total modal energy of the 128 modes",
        "total_energy_pct": round(total_energy_pct, 4),
        "bottom_40pct_energy": f"{sum(record['energy_sum_pct'] for record in bottom):.4f}%",
        "bottom_40pct_mean_error": f"{np.mean([record['nrmse_mean'] for record in bottom]):.4f}",
    }

    rates = recovery_rates(config.bands, config.tau)

    summary = {model: summary_of_model(results, model) for model in ESTIMATORS}
    summary = {model: values for model, values in summary.items() if values is not None}
    config_summary = {}
    for record in results:
        key = (record["model_type"], record["mask_num"], record["sigma_val"])
        config_summary[str(key)] = {
            "model": record["model_type"],
            "mask_num": record["mask_num"],
            "sigma": record["sigma_val"],
            "nrmse_mean": record["nrmse_mean"],
            "spearman_r": record["spearman_r"],
            "n_test": record["n_test_samples"],
        }

    payload = {
        "task": "per-mode NRMSE",
        "description": ("per-mode NRMSE over the configuration grid, the energy-decile "
                        "table and the band recovery rates"),
        "n_configs_total": len(ESTIMATORS) * len(config.M_values) * len(config.sigma_values),
        "n_configs_completed": len(results),
        "n_modes": POD_RANK,
        "mode_energy_norm": mode_energy.tolist(),
        "config_results": results,
        "summary_by_model": summary,
        "config_summary": config_summary,
        "deciles": deciles,
        "recovery_rates": rates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {len(results)} configurations to {args.output.relative_to(ROOT)} "
          f"({time.time() - started:.0f} s)")
    if missing:
        print(f"WARNING: {len(missing)} configurations not found, e.g. {missing[:3]}")

    print(f"\n{EXAMPLE_CONFIG[0].upper()} M={EXAMPLE_CONFIG[1]} sigma={EXAMPLE_CONFIG[2]}: "
          f"rho = {example['spearman_r']:+.4f} (p = {example['spearman_p']:.2e}), "
          f"NRMSE mean = {example['nrmse_mean']:.4f}")
    print(f"{'decile':>7} {'modes':>6} {'energy share (%)':>17} {'NRMSE mean':>11}")
    for record in rows:
        print(f"{record['decile']:>7} {record['n_modes']:>6} "
              f"{record['energy_share_pct']:>17.4f} {record['nrmse_mean']:>11.4f}")

    print(f"\nrecovery rates (tau = {config.tau}, {rates['total_records']} records):")
    for band in config.bands:
        overall = rates["overall"][band]
        print(f"  {band}: {overall['rate_pct']:>5.1f}% "
              f"({overall['passed']}/{overall['total']})")
    print(f"  drop factor A4/W1: {rates['drop_factor']}")

    if args.verify:
        return verify(results, deciles, rates)
    return 0


def verify(results: list[dict], deciles: dict, rates: dict) -> int:
    """Compare the reported values with the frozen values of the manuscript."""
    print("\nverification against the manuscript")
    ok = True

    example = next(
        record for record in results
        if record["model_type"] == EXAMPLE_CONFIG[0]
        and record["mask_num"] == EXAMPLE_CONFIG[1]
        and abs(record["sigma_val"] - EXAMPLE_CONFIG[2]) < 1e-12
    )
    difference = abs(example["spearman_r"] - FROZEN_CORRELATION)
    print(f"  {EXAMPLE_CONFIG[0].upper()} M={EXAMPLE_CONFIG[1]} sigma={EXAMPLE_CONFIG[2]}: "
          f"rho = {example['spearman_r']:+.4f} (manuscript {FROZEN_CORRELATION:+.4f})  "
          f"{'OK' if difference < 5e-3 else 'DIFFERS'}")
    ok &= difference < 5e-3

    for band, expected in FROZEN_RECOVERY.items():
        reported = rates["overall"][band]["rate_pct"]
        difference = abs(reported - expected)
        print(f"  recovery {band}: {reported:>5.1f}% (manuscript {expected:>5.1f}%)  "
              f"{'OK' if difference < 0.2 else 'DIFFERS'}")
        ok &= difference < 0.2

    top_thirteen = deciles["structured_descending"][0]["energy_share_pct"]
    top_twelve = deciles["structured"][-1]["energy_share_pct"]
    difference = abs(top_thirteen - FROZEN_TOP_DECILE_ENERGY)
    print(f"  most energetic 13 modes: {top_thirteen:.2f}% of the modal energy "
          f"(manuscript {FROZEN_TOP_DECILE_ENERGY:.2f}%)  "
          f"{'OK' if difference < 0.05 else 'DIFFERS'}")
    print(f"  most energetic 12 modes: {top_twelve:.2f}% (the grouping of the per-configuration listing)")
    ok &= difference < 0.05
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

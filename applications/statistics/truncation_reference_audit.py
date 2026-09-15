"""Accuracy of the POD truncation used as the reconstruction reference.

Every band error in the paper is measured against a rank-r POD truncation of the
same snapshot. This module quantifies that reference: for each rank it reports
the per-band truncation error over a test block, the fraction of the energy the
truncation retains, and the smallest rank at which the truncation is accurate in
every band. It also states the resulting scale count S_full, i.e. how many bands
of the truncation itself sit below tau.

The test block is the last 20% of each sequence, which is the protocol of the
main experiments. For the two auxiliary datasets (RDB, SST) the block is capped
at 300 snapshots so that all three are summarised on comparable sample sizes.

Inputs
    data/{cylinder2d_q1,rdb_h5,sst_weekly}.npy
    artifacts/pod_bases/{...}/pod_base_bundle.npz
Output
    artifacts/statistics/truncation_reference_audit.json
    artifacts/statistics/truncation_reference_audit.csv

Usage
    python -m applications.statistics.truncation_reference_audit
    python -m applications.statistics.truncation_reference_audit --datasets nc
    python -m applications.statistics.truncation_reference_audit --verify

``--verify`` compares the result with the audit reported in the submitted
manuscript, which was produced by the same computation from a legacy script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.core.constants import (  # noqa: E402
    BANDS_CF,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    DEFAULT_WAVELET,
    TAU_DEFAULT,
)
from luna.data.io import load_npy, load_npz  # noqa: E402
from luna.wavelet.metrics import band_error  # noqa: E402
from luna.wavelet.transform import decompose_field_2d  # noqa: E402

#: Datasets of the paper, with the rank grid and test block of each.
DATASETS = {
    "nc": {
        "label": "numerical cylinder wake",
        "data": "data/cylinder2d_q1.npy",
        "pod_bundle": "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz",
        "grid": (80, 160),
        "channels": 2,
        "ranks": [16, 32, 64, 128, 200],
    },
    "rdb_h5": {
        "label": "radial dam break",
        "data": "data/rdb_h5.npy",
        "pod_bundle": "artifacts/pod_bases/rdb_h5/pod_base_bundle.npz",
        "grid": (128, 128),
        "channels": 1,
        "ranks": [16, 32, 64, 128, 256],
    },
    "sst_weekly": {
        "label": "weekly sea-surface temperature",
        "data": "data/sst_weekly.npy",
        "pod_bundle": "artifacts/pod_bases/sst_weekly/pod_base_bundle.npz",
        "grid": (180, 360),
        "channels": 1,
        "ranks": [32, 64, 128, 256, 512, 1024],
    },
}
TEST_RATIO = 0.2
#: Upper bound on the test block, so that all datasets use the same sample size.
MAX_TEST_SNAPSHOTS = 300
OUTPUT = ROOT / "artifacts" / "statistics" / "truncation_reference_audit.json"
OUTPUT_CSV = ROOT / "artifacts" / "statistics" / "truncation_reference_audit.csv"
#: Audit of the submitted manuscript, kept for --verify.
SUBMITTED = ROOT / "artifacts" / "derived" / "main" / "statistics" / "oracle_audit_testset.json"


def test_block(n_total: int) -> np.ndarray:
    """Indices of the held-out block: the last 20% of the sequence, capped in
    length so that every dataset is summarised on the same number of snapshots.

    The cap keeps the first snapshots of the block, which is the convention of
    the main experiments for the numerical cylinder wake, where the block is
    exactly 300 snapshots long.
    """
    n_test = int(n_total * TEST_RATIO)
    return np.arange(n_total - n_test, n_total - n_test + min(n_test, MAX_TEST_SNAPSHOTS),
                     dtype=int)


def truncation_band_errors(
    fields: np.ndarray,
    target_ch0: np.ndarray,
    basis: np.ndarray,
    mean_field: np.ndarray,
    channels: int,
    grid: tuple[int, int],
    ranks: list[int],
    wavelet: str,
    level: int,
    mode: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-snapshot band errors of the rank-r truncation, for every rank.

    For a multi-channel dataset the coefficients are obtained from the full
    state, which is what the main experiments do; the errors are then evaluated
    on the streamwise component, as everywhere else in the paper.
    """
    height, width = grid
    available = basis.shape[0]
    shape = (height, width, channels)
    basis_flat = basis.reshape(available, height * width * channels)
    mean_flat = mean_field.ravel()
    basis_ch0 = basis[:, :, :, 0].reshape(available, height * width) if channels > 1 else basis_flat
    mean_ch0 = mean_field[:, :, 0].ravel() if channels > 1 else mean_flat

    errors: dict[int, dict[str, np.ndarray]] = {}
    for rank in ranks:
        clamped = min(rank, available)
        per_band: dict[str, list[float]] = {band: [] for band in BANDS_CF}
        for index in range(target_ch0.shape[0]):
            if channels > 1:
                coefficients = basis_flat[:clamped] @ (fields[index].ravel() - mean_flat)
                reconstruction = (basis_flat[:clamped].T @ coefficients + mean_flat)
                reconstruction = reconstruction.reshape(shape)[:, :, 0]
            else:
                coefficients = basis_ch0[:clamped] @ (fields[index].ravel() - mean_ch0)
                reconstruction = basis_ch0[:clamped].T @ coefficients + mean_ch0
                reconstruction = reconstruction.reshape(height, width)
            target_bands = decompose_field_2d(target_ch0[index], wavelet, level, mode)
            recon_bands = decompose_field_2d(reconstruction, wavelet, level, mode)
            for band in BANDS_CF:
                per_band[band].append(band_error(target_bands[band], recon_bands[band]))
        errors[rank] = {band: np.asarray(values) for band, values in per_band.items()}
    return errors


def audit_dataset(name: str, config: dict, tau: float, wavelet: str, level: int,
                  mode: str) -> dict:
    """Truncation audit of one dataset over its rank grid."""
    print(f"\n== {name} ({config['label']})")
    fields = load_npy(str(ROOT / config["data"]))
    pod = load_npz(str(ROOT / config["pod_bundle"]))
    basis = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)

    indices = test_block(fields.shape[0])
    test_fields = fields[indices].astype(np.float64)
    channels = config["channels"]
    height, width = config["grid"]
    # Single-channel datasets store a trailing channel axis of length one; the
    # wavelets are applied to the two-dimensional field.
    target_ch0 = (test_fields[:, :, :, 0] if channels > 1
                  else test_fields.reshape(test_fields.shape[0], height, width))
    print(f"   {len(indices)} snapshots ({indices[0]}-{indices[-1]}), "
          f"POD rank available: {basis.shape[0]}")

    ranks = sorted({min(r, basis.shape[0]) for r in config["ranks"]} | {basis.shape[0]})
    start = time.time()
    errors = truncation_band_errors(
        test_fields, target_ch0, basis, mean_field, channels, config["grid"],
        ranks, wavelet, level, mode,
    )
    print(f"   truncation errors computed ({time.time() - start:.1f}s)")

    tau_q95 = tau / 5.0
    table = {}
    first_mean_ok = first_q95_ok = first_strict_ok = None
    for rank in ranks:
        bands = {}
        mean_ok = q95_ok = True
        for band in BANDS_CF:
            values = errors[rank][band]
            statistics = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
                "std": float(values.std()),
                "n_samples": int(values.size),
                "mean_ok": bool(values.mean() < tau),
                "q95_ok": bool(np.quantile(values, 0.95) < tau_q95),
            }
            mean_ok &= statistics["mean_ok"]
            q95_ok &= statistics["q95_ok"]
            bands[band] = statistics
        # Scale count of the truncation itself: how many of its bands are
        # accurate, which is the reference that the reconstructions are
        # compared against.
        accurate = sum(
            (errors[rank][band] < tau).astype(int) for band in BANDS_CF
        )
        table[str(rank)] = {
            "bands": bands,
            "mean_s_full": float(accurate.mean()),
            "mean_s_coh": float(accurate.mean()),
            "mean_ok_all_bands": bool(mean_ok),
            "q95_ok_all_bands": bool(q95_ok),
            "strict_ok": bool(mean_ok and q95_ok),
        }
        if first_mean_ok is None and mean_ok:
            first_mean_ok = rank
        if first_q95_ok is None and q95_ok:
            first_q95_ok = rank
        if first_strict_ok is None and mean_ok and q95_ok:
            first_strict_ok = rank

    return {
        "dataset": name,
        "label": config["label"],
        "config": {
            "grid": list(config["grid"]),
            "n_channels": channels,
            "n_test_samples": int(indices.size),
            "test_indices": [int(i) for i in indices],
            "test_ratio": TEST_RATIO,
            "tau": tau,
            "tau_q95": tau_q95,
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
            "ranks_tested": ranks,
        },
        "first_rank_mean_below_tau": first_mean_ok,
        "first_rank_q95_below_tau_over_five": first_q95_ok,
        "first_rank_strict": first_strict_ok,
        "table": table,
    }


def energy_retention(dataset: str, rank: int) -> float:
    """Fraction of the POD energy retained by the leading ``rank`` modes."""
    pod = load_npz(str(ROOT / DATASETS[dataset]["pod_bundle"]))
    if "cumulative_energy_ratio" in pod:
        cumulative = np.asarray(pod["cumulative_energy_ratio"], dtype=np.float64)
        return float(cumulative[min(rank, cumulative.size) - 1])
    singular_values = np.asarray(pod["singular_values"], dtype=np.float64)
    energy = singular_values**2
    return float(energy[:rank].sum() / energy.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--verify", action="store_true", help="compare with the submitted audit")
    args = parser.parse_args()

    tau = TAU_DEFAULT
    start = time.time()
    print(f"== POD truncation reference audit (tau={tau}, "
          f"{DEFAULT_WAVELET} level {DEFAULT_LEVEL} {DEFAULT_MODE})")

    summaries = [
        audit_dataset(name, DATASETS[name], tau, DEFAULT_WAVELET, DEFAULT_LEVEL,
                      DEFAULT_MODE)
        for name in args.datasets
    ]

    retention = {}
    for name, summary in zip(args.datasets, summaries):
        rank = 128 if 128 in summary["table"] else summary["config"]["ranks_tested"][-1]
        try:
            retention[name] = {"rank": rank, "energy_fraction": energy_retention(name, rank)}
        except KeyError:
            print(f"   [note] no singular values stored for {name}")

    if args.verify:
        return _verify(summaries)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({
            "quantity": "accuracy of the rank-r POD truncation reference",
            "tau": tau,
            "tau_q95": tau / 5.0,
            "energy_retention": retention,
            "summaries": summaries,
            "runtime_s": round(time.time() - start, 2),
        }, indent=2),
        encoding="utf-8",
    )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "rank", "band", "mean", "median", "q95", "max",
                         "std", "n_samples", "mean_ok", "q95_ok", "strict_ok"])
        for summary in summaries:
            for rank, entry in summary["table"].items():
                for band in BANDS_CF:
                    band_stats = entry["bands"][band]
                    writer.writerow([
                        summary["dataset"], int(rank), band,
                        f"{band_stats['mean']:.6f}", f"{band_stats['median']:.6f}",
                        f"{band_stats['q95']:.6f}", f"{band_stats['max']:.6f}",
                        f"{band_stats['std']:.6f}", band_stats["n_samples"],
                        band_stats["mean_ok"], band_stats["q95_ok"], entry["strict_ok"],
                    ])

    print(f"\n[OK] {OUTPUT.name} / {OUTPUT_CSV.name} ({time.time() - start:.1f}s)")
    for name, summary in zip(args.datasets, summaries):
        print(f"   {name}: first rank with all bands below tau = "
              f"{summary['first_rank_mean_below_tau']}, strict = "
              f"{summary['first_rank_strict']}")
    for name, item in retention.items():
        print(f"   {name}: rank {item['rank']} retains "
              f"{100 * item['energy_fraction']:.3f}% of the energy")
    return 0


def _verify(summaries: list[dict]) -> int:
    """Compare with the audit reported in the submitted manuscript."""
    if not SUBMITTED.exists():
        print("   submitted audit not available; nothing to compare")
        return 1
    old = {entry["dataset"]: entry for entry in
           json.loads(SUBMITTED.read_text(encoding="utf-8"))["summaries"]}
    print("== verification against the submitted audit")
    failures = 0
    for summary in summaries:
        reference = old.get(summary["dataset"])
        if reference is None:
            print(f"   [skip] {summary['dataset']} absent from the submitted audit")
            continue
        for rank, entry in summary["table"].items():
            expected = reference["audit_table"].get(rank)
            if expected is None:
                print(f"   [skip] {summary['dataset']} rank {rank} absent")
                continue
            for band in BANDS_CF:
                for field in ("mean", "median", "q95", "max"):
                    got = entry["bands"][band][field]
                    want = expected["bands"][band][field]
                    if abs(got - want) > 1e-12 * max(abs(want), 1e-6):
                        print(f"   [FAIL] {summary['dataset']} rank {rank} {band}.{field}: "
                              f"{got} != {want}")
                        failures += 1
        for field in ("first_rank_mean_below_tau", "first_rank_q95_below_tau_over_five",
                      "first_rank_strict"):
            print(f"   {summary['dataset']} {field}: "
                  f"new {summary[field]} vs submitted "
                  f"{reference[field.replace('first_rank', 'safe_rank')]}")
    print("   all checks passed" if not failures else f"   {failures} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

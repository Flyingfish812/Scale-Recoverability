"""Temporal dependence of the flow and the resulting bootstrap block length.

The 300 test snapshots are drawn from the 1501-snapshot sequence with a random stride of between 1 and 22 steps, so a block length measured in test snapshots would be meaningless. The shedding period is therefore estimated on the full equally spaced sequence, from the vortex-shedding pair of POD modes, and then converted into test-snapshot units for the block bootstrap used elsewhere.

This module also recomputes the two summary intervals quoted for the primary configuration, using time blocks as well as a snapshot-cluster resampling, so that the effect of the choice can be compared directly.

Inputs
    data/cylinder2d_q1.npy                 full snapshot sequence
    artifacts/pod_bases/cylinder2d_q1/     POD basis and coefficients
    artifacts/<estimator runs>/            the primary configuration
Output
    artifacts/statistics/temporal_dependence.json
    artifacts/statistics/tables/temporal_dependence.tex

Usage
    python -m applications.statistics.temporal_dependence
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.statistics.block_bootstrap import block_bootstrap_ci  # noqa: E402
from features.statistics.temporal_dependence import (  # noqa: E402
    report_temporal_series,
    suggest_block_length,
)
from features.training.estimator_runs import load_run, run_path  # noqa: E402

# : Configuration whose summary intervals are quoted in the paper.
PRIMARY = ("mlp", 20, 0.001, 0)
POD_BUNDLE = ROOT / "artifacts" / "pod_bases" / "cylinder2d_q1" / "pod_base_bundle.npz"
OUTPUT = ROOT / "artifacts" / "statistics" / "temporal_dependence.json"
OUTPUT_TEX = ROOT / "artifacts" / "statistics" / "tables" / "temporal_dependence.tex"


def full_sequence_diagnostics(snapshots: Path) -> tuple[dict, dict]:
    """Temporal diagnostics of the full sequence and the block length it implies."""
    fields = np.load(str(snapshots), mmap_mode="r")
    energy = np.empty(fields.shape[0], dtype=np.float64)
    for i in range(fields.shape[0]):
        energy[i] = float(np.sum(np.asarray(fields[i, :, :, 0], dtype=np.float64) ** 2))

    pod = np.load(str(POD_BUNDLE))
    coefficients = np.asarray(pod["coefficients"], dtype=np.float64)

    series = {"global_energy": energy}
    for mode in range(5):
        series[f"pod_mode{mode}"] = coefficients[:, mode]
    diagnostics = report_temporal_series(series, max_lag=50)
    return diagnostics, {"energy": energy, "coefficients": coefficients}


def main() -> int:
    cfg = get_config()
    bootstrap = cfg.block_bootstrap
    n_resamples = int(bootstrap["n_resamples"])
    seed = int(bootstrap["seed"])
    method = bootstrap["method"]
    candidates = [int(x) for x in bootstrap["candidate_block_lengths"]]

    start = time.time()
    print("== temporal dependence of the flow")

    print("[1] diagnostics of the full sequence")
    diagnostics, series = full_sequence_diagnostics(ROOT / "data" / "cylinder2d_q1.npy")
    suggestion = suggest_block_length(
        [series["energy"]] + [series["coefficients"][:, m] for m in range(2)],
        prefer_physical_period=bool(bootstrap["prefer_physical_period"]),
        candidate_block_lengths=candidates,
    )
    block_length = (
        int(suggestion["chosen_block_length"])
        if suggestion["chosen_block_length"] is not None
        else 62
    )
    print(f"   block length {block_length} ({suggestion['method']}); "
          f"physical period {suggestion['physical_period']:.1f} snapshots "
          f"(peak fraction {suggestion['peak_frac']:.2f})")

    print("[2] structure of the test snapshots")
    path = run_path(*PRIMARY)
    if path is None:
        raise SystemExit(f"missing run for the primary configuration {PRIMARY}")
    # Both fields come from load_run so that the estimator's stored unit convention (normalised for the convolutional model) is undone in one place; reading `target_nchw` directly would mismatch the reconstruction for any configuration that stores normalised fields.
    target, reconstruction = load_run(path)
    data = np.load(path, allow_pickle=True)
    test_indices = np.asarray(sorted(set(data["test_indices"].tolist())), dtype=np.int64)
    gaps = np.diff(test_indices)
    gap_stats = {
        "min": int(gaps.min()),
        "median": float(np.median(gaps)),
        "max": int(gaps.max()),
        "mean": float(gaps.mean()),
        "n_consecutive_pairs": int(np.sum(gaps == 1)),
    }
    mean_gap = float(gaps.mean())
    block_in_test_units = max(3, int(round(block_length / mean_gap))) if mean_gap else block_length

    order = np.argsort(test_indices)
    n = reconstruction.shape[0]
    if order.size != n:
        raise SystemExit(f"{n} snapshots but {order.size} snapshot times")
    metrics = [compute_sample_metrics(reconstruction, target, i, tau=cfg.tau) for i in range(n)]
    ger = np.asarray([m["GER"] for m in metrics])[order]
    s_full = np.asarray([m["S_full"] for m in metrics], dtype=float)[order]
    diagnostics_primary = report_temporal_series(
        {"GER": ger, "S_full": s_full}, max_lag=30
    )

    print(f"[3] recomputing the summary intervals (block={block_in_test_units} test units, "
          f"{method}, n_resamples={n_resamples})")
    ci_s_full = block_bootstrap_ci(
        s_full, lambda x: float(np.mean(x)), block_len=block_in_test_units,
        n_resamples=n_resamples, seed=seed, method=method,
    )
    ci_s_full_cluster = block_bootstrap_ci(
        s_full, lambda x: float(np.mean(x)), block_len=1,
        n_resamples=n_resamples, seed=seed, cluster_ids=np.arange(s_full.size),
    )
    ci_p3 = block_bootstrap_ci(
        s_full, lambda x: float(np.mean(x >= 3)), block_len=block_in_test_units,
        n_resamples=n_resamples, seed=seed, method=method,
    )

    report = {
        "quantity": "temporal dependence of the flow and bootstrap block length",
        "full_sequence_diagnostics": diagnostics,
        "test_snapshot_structure": {
            "n_test": int(test_indices.size),
            "gaps": gap_stats,
            "mean_test_gap": round(mean_gap, 3),
            "physical_period_snapshots": suggestion["physical_period"],
            "physical_peak_fraction": suggestion["peak_frac"],
            "block_length_snapshots": block_length,
            "block_length_test_units": block_in_test_units,
            "block_method": method,
        },
        "primary_configuration": {
            "model": PRIMARY[0], "sensor_count": PRIMARY[1],
            "sigma": PRIMARY[2], "training_seed": PRIMARY[3],
        },
        "primary_diagnostics": diagnostics_primary,
        "block_bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": method,
            "block_length_test_units": block_in_test_units,
            "mean_s_full": ci_s_full,
            "mean_s_full_snapshot_cluster": ci_s_full_cluster,
            "p_at_least_three_bands": ci_p3,
        },
        "suggested_block_length": {
            k: (round(v, 3) if isinstance(v, float) else v) for k, v in suggestion.items()
        },
        "runtime_s": round(time.time() - start, 2),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Quantity & n & ACF(1) & ACF(5) & IACT & ESS & period & peak frac \\",
        r"\midrule",
    ]
    for name, stats in diagnostics.items():
        lines.append(
            f"{name} & {stats['n']} & {stats['acf_lag1']:.3f} & {stats['acf_lag5']:.3f} "
            f"& {stats['iact']:.1f} & {stats['ess']:.1f} & "
            f"{stats['dominant_period'] if stats['dominant_period'] else '--'} "
            f"& {stats['dominant_peak_frac']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {OUTPUT.name} / {OUTPUT_TEX.name} ({time.time() - start:.1f}s)")
    for name in ["global_energy", "pod_mode0", "pod_mode1"]:
        stats = diagnostics[name]
        print(f"   {name}: IACT={stats['iact']:.1f} ESS={stats['ess']:.0f} "
              f"period={stats['dominant_period']} frac={stats['dominant_peak_frac']:.2f}")
    print(f"   mean S_full block CI [{ci_s_full['ci_5_low']:.4f}, {ci_s_full['ci_5_high']:.4f}] "
          f"| cluster CI [{ci_s_full_cluster['ci_5_low']:.4f}, {ci_s_full_cluster['ci_5_high']:.4f}]")
    print(f"   at least three bands CI [{ci_p3['ci_5_low']:.4f}, {ci_p3['ci_5_high']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

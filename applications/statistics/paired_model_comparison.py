"""Paired comparison of the two learned estimators on identical inputs.

The main text compares the POD-coefficient network (MLP) with the convolutional estimator (VCNN). Comparing two pools of runs with a rank test would treat snapshots of the same configuration as independent samples, so the comparison is made pair-wise instead: for every (snapshot, sensor count, noise level, training seed) the two estimators are evaluated on the very same measurement, and the per-snapshot difference of each metric is bootstrapped in time blocks to obtain a confidence interval that respects the temporal correlation of the flow. For the aggregate across configurations the resampling unit is the snapshot identity: moving blocks of consecutive test identities are drawn and every record of a drawn identity -- across sensor counts, noise levels and training seeds -- enters with its multiplicity, so the repeated use of the same targets is respected.

Reported metrics
    GER            relative L2 error over the full two-component state
    S_full         number of consecutive bands, from the coarsest, below tau
    S_coh          the same count on the POD-dominant band error
    W1             relative error of the finest wavelet band
    vorticity_RMSE RMSE of the discrete Laplacian on the streamwise component gradient_RMSE  RMSE of the first spatial derivatives on the streamwise component

Both counts and the band errors are evaluated on the streamwise velocity, and the band basis of S_coh is fitted to that same field, as in luna.wavelet.metrics.compute_S_coh.

Inputs
    artifacts/<estimator runs>/       one test_raw.npz per configuration
    data/cylinder2d_q1.npy            training snapshots behind the band basis
Output
    artifacts/statistics/paired_model_comparison.json
    artifacts/statistics/paired_model_comparison.csv
    artifacts/statistics/tables/paired_model_comparison.tex

Usage
    python -m applications.statistics.paired_model_comparison
    python -m applications.statistics.paired_model_comparison --verify

``--verify`` re-runs the analysis and compares the result with the stored one, so that the table in the paper can be regenerated and checked deterministically. When a stored summary from an earlier run is present, it is printed alongside for comparison.
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

from applications.config import get_config  # noqa: E402
from applications.statistics.scoh_vs_sfull import band_pod_models_for_seed  # noqa: E402
from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.statistics.block_bootstrap import block_bootstrap_ci  # noqa: E402
from features.training.estimator_runs import (  # noqa: E402
    load_run,
    run_path,
)
from luna.wavelet.metrics import compute_S_coh  # noqa: E402

METRICS = ["GER", "S_full", "S_coh", "W1", "vorticity_RMSE", "gradient_RMSE"]
BANDS = ["A4", "W4", "W3", "W2", "W1"]
OUTPUT = ROOT / "artifacts" / "statistics" / "paired_model_comparison.json"
OUTPUT_CSV = ROOT / "artifacts" / "statistics" / "paired_model_comparison.csv"
OUTPUT_TEX = ROOT / "artifacts" / "statistics" / "tables" / "paired_model_comparison.tex"
# Stored summary of an earlier run, printed by --verify when present.
SUBMITTED = ROOT / "artifacts" / "derived" / "supplementary" / "paired_mlp_vcnn_summary.json"
# Agreement required between the target fields of the two estimators, which differ only by the normalisation convention of the stored fields.
TARGET_TOLERANCE = 1e-5


def metrics_of(
    output_nchw: np.ndarray,
    target_nchw: np.ndarray,
    index: int,
    band_pod: dict,
    tau: float,
    wavelet: str,
    level: int,
    mode: str,
) -> dict:
    """All six metrics of one snapshot, on the physical fields."""
    values = compute_sample_metrics(output_nchw, target_nchw, index, tau=tau)
    values["W1"] = values.pop("E_W1")
    values["S_coh"] = int(compute_S_coh(
        np.asarray(target_nchw[index], dtype=np.float64)[0],
        np.asarray(output_nchw[index], dtype=np.float64)[0],
        band_pod, tau=tau, wavelet=wavelet, level=level, mode=mode,
    ))
    return values


def block_length(cfg) -> int:
    """Bootstrap block length in test units, from the physical shedding period."""
    candidates = [int(x) for x in cfg.block_bootstrap["candidate_block_lengths"]]
    mean_gap = 1501.0 / 300.0
    period = 62 if 62 in candidates else candidates[-1]
    return max(3, int(round(period / mean_gap)))


def paired_summary(
    diffs: np.ndarray, block_len: int, n_resamples: int, seed: int, method: str
) -> dict:
    """Mean paired difference with a block-bootstrap confidence interval."""
    ci = block_bootstrap_ci(
        diffs, lambda x: float(np.mean(x)), block_len=block_len,
        n_resamples=n_resamples, seed=seed, method=method,
    )
    std = float(np.std(diffs))
    mean = float(np.mean(diffs))
    return {
        "n_pairs": int(diffs.size),
        "mean_diff": mean,
        "median_diff": float(np.median(diffs)),
        "std_diff": std,
        "effect_size": mean / std if std > 0 else float("nan"),
        "win_rate_first": float(np.mean(diffs < 0)),
        "ci_low": float(ci["ci_5_low"]),
        "ci_high": float(ci["ci_5_high"]),
    }


def identity_block_length(identities: np.ndarray, cfg) -> int:
    """Block length in snapshot identities from the physical shedding period.

    The identities of the aggregate are the union of the test splits of the
training
    seeds, so their mean gap in the full sequence sets how many identities make
up
    one shedding period. Blocks drawn over the identity array are therefore time
    blocks, as in the single-run analysis.
    """
    candidates = [int(x) for x in cfg.block_bootstrap["candidate_block_lengths"]]
    mean_gap = float(np.mean(np.diff(identities)))
    period = 62 if 62 in candidates else candidates[-1]
    return max(3, int(round(period / mean_gap)))


def identity_block_summary(
    diffs: list[np.ndarray], indices: list[np.ndarray],
    block_length: int, n_resamples: int, seed: int,
) -> dict:
    """Mean paired difference with a snapshot-identity block-bootstrap interval.

    Records are grouped by snapshot identity; a moving block of ``block_length``
    consecutive identities is drawn with replacement, every record of a drawn
    identity enters with its multiplicity, and the interval is the percentile
    interval of the resampled mean over all pooled records.
    """
    unique = np.unique(np.concatenate(indices))
    total = np.zeros(unique.size, dtype=np.float64)
    count = np.zeros(unique.size, dtype=np.float64)
    for idx, values in zip(indices, diffs):
        position = np.searchsorted(unique, idx)
        np.add.at(total, position, values)
        np.add.at(count, position, 1.0)

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(unique.size / block_length))
    offsets = np.arange(block_length)
    distribution = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        starts = rng.randint(0, unique.size - block_length + 1, size=n_blocks)
        position = (starts[:, None] + offsets[None, :]).reshape(-1)[: unique.size]
        weight = np.bincount(position, minlength=unique.size)
        distribution[b] = (weight @ total) / (weight @ count)

    pooled = np.concatenate(diffs)
    mean = float(pooled.mean())
    std = float(pooled.std())
    return {
        "n_pairs": int(pooled.size),
        "mean_diff": mean,
        "median_diff": float(np.median(pooled)),
        "std_diff": std,
        "effect_size": mean / std if std > 0 else float("nan"),
        "win_rate_first": float(np.mean(pooled < 0)),
        "ci_low": float(np.percentile(distribution, 2.5)),
        "ci_high": float(np.percentile(distribution, 97.5)),
        "spread_low": float(np.percentile(pooled, 2.5)),
        "spread_high": float(np.percentile(pooled, 97.5)),
    }


def identity_block_length(identities: np.ndarray, cfg) -> int:
    """Block length in snapshot identities from the physical shedding period.

    The identities of the aggregate are the union of the test splits of the
    training seeds, so their mean gap in the full sequence sets how many
    identities make up one shedding period; blocks drawn over the identity
    array are therefore time blocks, as in the single-run analysis.
    """
    candidates = [int(x) for x in cfg.block_bootstrap["candidate_block_lengths"]]
    mean_gap = float(np.mean(np.diff(identities)))
    period = 62 if 62 in candidates else candidates[-1]
    return max(3, int(round(period / mean_gap)))


def identity_block_summary(
    diffs: list[np.ndarray], indices: list[np.ndarray],
    block_length: int, n_resamples: int, seed: int,
) -> dict:
    """Mean paired difference with a snapshot-identity block-bootstrap interval.

    The records are grouped by the snapshot identity they belong to; a moving
    block of ``block_length`` consecutive identities is drawn with replacement,
    every record of a drawn identity enters with its multiplicity, and the
    interval is the percentile interval of the resampled mean over all pooled
    records. A snapshot drawn several times therefore contributes its records
    several times, and the records of one snapshot stay together across sensor
    counts, noise levels and training seeds.
    """
    unique = np.unique(np.concatenate(indices))
    total = np.zeros(unique.size, dtype=np.float64)
    count = np.zeros(unique.size, dtype=np.float64)
    for idx, values in zip(indices, diffs):
        position = np.searchsorted(unique, idx)
        np.add.at(total, position, values)
        np.add.at(count, position, 1.0)

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(unique.size / block_length))
    offsets = np.arange(block_length)
    distribution = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        starts = rng.randint(0, unique.size - block_length + 1, size=n_blocks)
        position = (starts[:, None] + offsets[None, :]).reshape(-1)[: unique.size]
        weight = np.bincount(position, minlength=unique.size)
        distribution[b] = (weight @ total) / (weight @ count)

    pooled = np.concatenate(diffs)
    mean = float(pooled.mean())
    std = float(pooled.std())
    return {
        "n_pairs": int(pooled.size),
        "mean_diff": mean,
        "median_diff": float(np.median(pooled)),
        "std_diff": std,
        "effect_size": mean / std if std > 0 else float("nan"),
        "win_rate_first": float(np.mean(pooled < 0)),
        "ci_low": float(np.percentile(distribution, 2.5)),
        "ci_high": float(np.percentile(distribution, 97.5)),
        "spread_low": float(np.percentile(pooled, 2.5)),
        "spread_high": float(np.percentile(pooled, 97.5)),
    }


def compare_configuration(job: tuple) -> dict | None:
    """Metrics and paired differences of every snapshot of one configuration.

    Runs in a worker process when ``--jobs`` is used, hence the flat argument tuple: the band basis is fitted once per worker process, because it is far larger than the returned data.
    """
    sensors, sigma, seed, n_samples, settings = job
    wavelet, level, mode, tau, block_len, n_resamples, method, bootstrap_seed = settings

    mlp_path = run_path("mlp", sensors, sigma, seed)
    vcnn_path = run_path("vcnn", sensors, sigma, seed)
    if mlp_path is None or vcnn_path is None:
        print(f"   [skip] run absent (M={sensors}, sigma={sigma}, seed={seed})")
        return None

    # Pairs are formed within a training seed, so the diagnostic band basis is
    # the one fitted on that seed's training split.
    band_pod = band_pod_models_for_seed(seed)
    target_mlp, out_mlp = load_run(mlp_path)
    target_vcnn, out_vcnn = load_run(vcnn_path)

    drift = float(np.max(np.abs(target_mlp - target_vcnn)))
    if drift > TARGET_TOLERANCE:
        raise RuntimeError(
            f"target fields disagree by {drift:.2e} for M={sensors}, sigma={sigma}, "
            f"seed={seed}; check the unit convention"
        )

    n = out_mlp.shape[0] if not n_samples else min(out_mlp.shape[0], n_samples)
    diffs = {m: np.empty(n) for m in METRICS}
    model_values = {model: {m: np.empty(n) for m in METRICS} for model in ("mlp", "vcnn")}
    for i in range(n):
        first = metrics_of(out_mlp, target_mlp, i, band_pod, tau, wavelet, level, mode)
        second = metrics_of(out_vcnn, target_vcnn, i, band_pod, tau, wavelet, level, mode)
        for m in METRICS:
            diffs[m][i] = first[m] - second[m]
            model_values["mlp"][m][i] = float(first[m])
            model_values["vcnn"][m][i] = float(second[m])

    # The stored matrices follow the unsorted ``test_indices`` as saved, so the
    # metric series is reordered into time before any block resampling; blocks
    # over the stored order would resample unrelated snapshots.
    stored = np.asarray(np.load(mlp_path, allow_pickle=True)["test_indices"], dtype=np.int64)[:n]
    order = np.argsort(stored)
    sorted_indices = stored[order]
    diffs = {m: diffs[m][order] for m in METRICS}
    model_values = {
        model: {m: model_values[model][m][order] for m in METRICS}
        for model in ("mlp", "vcnn")
    }
    rows = []
    for m in METRICS:
        stats = paired_summary(
            diffs[m], block_len, min(n_resamples, 5000),
            seed=int(bootstrap_seed) + sensors + int(sigma * 1000) + seed, method=method,
        )
        rows.append({
            "mask_family": "family_01", "sensor_count": sensors, "sigma": sigma,
            "training_seed": seed, "metric": m, **stats,
        })
    return {
        "config": (sensors, sigma, seed),
        "rows": rows,
        "diffs": diffs,
        "model_values": model_values,
        "test_indices": sorted_indices,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all snapshots")
    parser.add_argument("--configs", type=int, default=0, help="0 = all configurations")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--verify", action="store_true", help="compare with the stored result")
    args = parser.parse_args()

    cfg = get_config()
    tau = cfg.tau
    wavelet, level, mode = cfg.wavelet_family, cfg.wavelet_level, cfg.wavelet_mode
    bootstrap = cfg.block_bootstrap
    n_resamples = int(bootstrap["n_resamples"])
    method = bootstrap["method"]
    block_len = block_length(cfg)

    first_mlp = run_path("mlp", cfg.M_values[0], cfg.sigma_values[0], cfg.mlp_seeds[0])
    if first_mlp is None:
        raise SystemExit("no trained runs found; run the training pipeline first")

    start = time.time()
    print(f"== paired MLP-VCNN comparison (block={block_len} test units, {method}, "
          f"n_resamples={n_resamples})")

    jobs = [
        (sensors, sigma, seed, args.max_samples,
         (wavelet, level, mode, tau, block_len, n_resamples, method, int(bootstrap["seed"])))
        for sensors in cfg.M_values
        for sigma in cfg.sigma_values
        for seed in cfg.mlp_seeds
    ]
    if args.configs:
        jobs = jobs[: args.configs]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = [r for r in pool.map(compare_configuration, jobs) if r is not None]
    else:
        results = [r for r in map(compare_configuration, jobs) if r is not None]

    if not results:
        raise SystemExit("no configurations could be compared")

    rows: list[dict] = []
    collected: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    collected_idx: list[np.ndarray] = []
    model_values: dict[str, dict[str, list[np.ndarray]]] = {
        model: {m: [] for m in METRICS} for model in ("mlp", "vcnn")
    }
    for result in results:
        rows.extend(result["rows"])
        collected_idx.append(result["test_indices"])
        for m in METRICS:
            collected[m].append(result["diffs"][m])
            for model in ("mlp", "vcnn"):
                model_values[model][m].append(result["model_values"][model][m])
    n_configs = len(results)
    print(f"   configurations: {n_configs}")

    identities = np.unique(np.concatenate(collected_idx))
    identity_block = identity_block_length(identities, cfg)

    aggregate = {}
    for m in METRICS:
        # ``ci_low``/``ci_high`` are the snapshot-identity block interval of the
        # mean, as quoted in the paper; the spread of the individual paired
        # differences is reported separately because it answers a different
        # question.
        stats = identity_block_summary(
            collected[m], collected_idx, identity_block, n_resamples,
            int(bootstrap["seed"]),
        )
        for model in ("mlp", "vcnn"):
            values = np.concatenate(model_values[model][m])
            stats[f"{model}_mean"] = float(values.mean())
            stats[f"{model}_median"] = float(np.median(values))
        aggregate[m] = stats

    result = {
        "quantity": "paired difference of MLP and VCNN metrics on identical inputs",
        "convention": {
            "pairing": "(snapshot, sensor count, noise level, training seed, mask family)",
            "fields": "physical units for both estimators",
            "ger": "two velocity components; other metrics on the streamwise component",
        },
        "mask_family": "family_01",
        "n_configs": n_configs,
        "block_bootstrap": {
            "block_length_test_units": block_len,
            "method": method,
            "n_resamples": n_resamples,
            "seed": int(bootstrap["seed"]),
        },
        "aggregate_resampling": {
            "unit": "snapshot identity (union of the test splits of the training seeds)",
            "block_length_identities": identity_block,
            "n_identities": int(identities.size),
            "mean_gap_snapshots": float(np.mean(np.diff(identities))),
            "n_resamples": n_resamples,
            "seed": int(bootstrap["seed"]),
        },
        "metrics": aggregate,
    }

    if args.verify:
        return _verify(aggregate, n_configs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(_format_table(aggregate), encoding="utf-8")

    print(f"[OK] {OUTPUT.name} / {OUTPUT_CSV.name} / {OUTPUT_TEX.name} "
          f"({time.time() - start:.1f}s)")
    for m in METRICS:
        a = aggregate[m]
        print(f"   {m:15s} MLP={a['mlp_mean']:.5f} VCNN={a['vcnn_mean']:.5f} "
              f"diff={a['mean_diff']:+.5f} CI=[{a['ci_low']:+.5f}, {a['ci_high']:+.5f}] "
              f"MLP<VCNN={a['win_rate_first']:.3f}")
    return 0


def _format_table(aggregate: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & MLP & VCNN & difference & 95\% CI & MLP<VCNN \\",
        r"\midrule",
    ]
    for metric, a in aggregate.items():
        lines.append(
            f"{metric} & {a['mlp_mean']:.5f} & {a['vcnn_mean']:.5f} & {a['mean_diff']:+.5f} "
            f"& [{a['ci_low']:+.5f}, {a['ci_high']:+.5f}] & {a['win_rate_first']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _verify(aggregate: dict, n_configs: int) -> int:
    """Recompute and compare with the stored result and the submitted summary."""
    print("== verification")
    if not OUTPUT.exists():
        print("   no stored result yet; nothing to compare")
        return 1
    stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
    failures = 0
    if stored["n_configs"] != n_configs:
        print(f"   [FAIL] n_configs {n_configs} != stored {stored['n_configs']}")
        failures += 1
    for metric, a in aggregate.items():
        b = stored["metrics"][metric]
        delta = abs(a["mean_diff"] - b["mean_diff"])
        ok = delta < 1e-9
        print(f"   [{'ok' if ok else 'FAIL'}] {metric:15s} diff={a['mean_diff']:+.5f} "
              f"stored={b['mean_diff']:+.5f} |delta|={delta:.2e}")
        failures += int(not ok)

    if SUBMITTED.exists():
        old = json.loads(SUBMITTED.read_text(encoding="utf-8"))["metrics"]
        print("   stored summary from an earlier run:")
        for metric in aggregate:
            if metric in old:
                print(f"      {metric:15s} old={old[metric]['mean_diff']:+.5f} "
                      f"new={aggregate[metric]['mean_diff']:+.5f}")
    print("   all checks passed" if not failures else f"   {failures} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

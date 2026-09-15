"""Paired comparison of the two learned estimators on identical inputs.

The main text compares the POD-coefficient network (MLP) with the convolutional
estimator (VCNN). Comparing two pools of runs with a rank test would treat
snapshots of the same configuration as independent samples, so the comparison is
made pair-wise instead: for every (snapshot, sensor count, noise level, training
seed) the two estimators are evaluated on the very same measurement, and the
per-snapshot difference of each metric is bootstrapped in time blocks to obtain
a confidence interval that respects the temporal correlation of the flow.

Reported metrics
    GER            relative L2 error over the full two-component state
    S_full         number of wavelet bands whose relative error is below tau
    S_coh          number of bands recoverable from the band-POD subspace
    W1             relative error of the finest wavelet band
    vorticity_RMSE RMSE of the discrete Laplacian on the streamwise component
    gradient_RMSE  RMSE of the first spatial derivatives on the streamwise
                   component

Inputs
    artifacts/<estimator runs>/       one test_raw.npz per configuration
    artifacts/wavelet_pod_nc_2000/    band-POD bundle used by S_coh
Output
    artifacts/statistics/paired_model_comparison.json
    artifacts/statistics/paired_model_comparison.csv
    artifacts/statistics/tables/paired_model_comparison.tex

Usage
    python -m applications.statistics.paired_model_comparison
    python -m applications.statistics.paired_model_comparison --verify

``--verify`` re-runs the analysis and compares the result with the stored one, so
that the table in the paper can be regenerated and checked deterministically.
The previously submitted summary was computed on the normalised convolutional
outputs and is therefore reported separately for the record.
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
from features.metrics.band_error.coherent_scoh import (  # noqa: E402
    compute_scoh_with_target_bands,
    compute_target_bands_bundle,
)
from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.statistics.block_bootstrap import block_bootstrap_ci  # noqa: E402
from features.training.estimator_runs import (  # noqa: E402
    load_run,
    run_path,
    snapshot_indices,
)
from luna.pod.band_pod import load_band_pod_bundle  # noqa: E402

METRICS = ["GER", "S_full", "S_coh", "W1", "vorticity_RMSE", "gradient_RMSE"]
BANDS = ["A4", "W4", "W3", "W2", "W1"]
BAND_POD_BUNDLE = ROOT / "artifacts" / "wavelet_pod_nc_2000" / "band_pod_bundle.npz"
TARGET_BANDS_CACHE = ROOT / "artifacts" / "statistics" / "cache" / "test_target_bands.npz"
OUTPUT = ROOT / "artifacts" / "statistics" / "paired_model_comparison.json"
OUTPUT_CSV = ROOT / "artifacts" / "statistics" / "paired_model_comparison.csv"
OUTPUT_TEX = ROOT / "artifacts" / "statistics" / "tables" / "paired_model_comparison.tex"
# Summary of the submitted manuscript, kept for the record in --verify.
SUBMITTED = ROOT / "artifacts" / "derived" / "supplementary" / "paired_mlp_vcnn_summary.json"
# Agreement required between the target fields of the two estimators, which
# differ only by the normalisation convention of the stored fields.
TARGET_TOLERANCE = 1e-5


def metrics_of(
    output_nchw: np.ndarray,
    target_nchw: np.ndarray,
    index: int,
    band_pod,
    target_bands_i: np.ndarray,
    tau: float,
    wavelet: str,
    level: int,
    mode: str,
) -> dict:
    """All six metrics of one snapshot, on the physical fields."""
    values = compute_sample_metrics(output_nchw, target_nchw, index, tau=tau)
    values["W1"] = values.pop("E_W1")
    prediction_hwc = np.asarray(output_nchw[index], dtype=np.float64).transpose(1, 2, 0)
    values["S_coh"] = int(
        compute_scoh_with_target_bands(
            prediction_hwc, target_bands_i, band_pod, tau=tau,
            wavelet=wavelet, level=level, mode=mode,
        )
    )
    return values


def load_target_bands(
    target_nchw: np.ndarray, wavelet: str, level: int, mode: str, n_samples: int
) -> np.ndarray:
    """Wavelet-band components of the test targets, cached across configurations.

    The targets do not depend on the estimator, so the decomposition is computed
    once for the full test split and reused for every configuration.
    """
    if TARGET_BANDS_CACHE.exists() and not n_samples:
        cached = np.load(str(TARGET_BANDS_CACHE))["bands"]
        if cached.shape[0] == target_nchw.shape[0]:
            return cached
        print(f"   [cache] stale ({cached.shape[0]} snapshots), recomputing")
        TARGET_BANDS_CACHE.unlink()
    print("   [cache] decomposing test targets into wavelet bands ...")
    start = time.time()
    targets = target_nchw[:n_samples] if n_samples else target_nchw
    bands = compute_target_bands_bundle(
        targets, bands=BANDS, wavelet=wavelet, level=level, mode=mode
    )
    if not n_samples:
        TARGET_BANDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(TARGET_BANDS_CACHE), bands=bands)
    print(f"   [cache] ready ({time.time() - start:.1f}s)")
    return bands


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


def compare_configuration(job: tuple) -> dict | None:
    """Metrics and paired differences of every snapshot of one configuration.

    Runs in a worker process when ``--jobs`` is used, hence the flat argument
    tuple: the band-POD bundle and the target-band cache are loaded once per
    worker process, because together they are far larger than the returned data.
    """
    sensors, sigma, seed, n_samples, settings, cache_path = job
    wavelet, level, mode, tau, block_len, n_resamples, method, bootstrap_seed = settings

    mlp_path = run_path("mlp", sensors, sigma, seed)
    vcnn_path = run_path("vcnn", sensors, sigma, seed)
    if mlp_path is None or vcnn_path is None:
        print(f"   [skip] run absent (M={sensors}, sigma={sigma}, seed={seed})")
        return None

    band_pod = load_band_pod_bundle(str(BAND_POD_BUNDLE))
    target_bands = np.load(str(cache_path))["bands"]
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
        first = metrics_of(
            out_mlp, target_mlp, i, band_pod, target_bands[i], tau, wavelet, level, mode
        )
        second = metrics_of(
            out_vcnn, target_vcnn, i, band_pod, target_bands[i], tau, wavelet, level, mode
        )
        for m in METRICS:
            diffs[m][i] = first[m] - second[m]
            model_values["mlp"][m][i] = float(first[m])
            model_values["vcnn"][m][i] = float(second[m])

    order = np.argsort(snapshot_indices(mlp_path)[:n])
    rows = []
    for m in METRICS:
        stats = paired_summary(
            diffs[m][order], block_len, min(n_resamples, 5000),
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
    target, _ = load_run(first_mlp)
    load_target_bands(target, wavelet, level, mode, args.max_samples)

    start = time.time()
    print(f"== paired MLP-VCNN comparison (block={block_len} test units, {method}, "
          f"n_resamples={n_resamples})")

    jobs = [
        (sensors, sigma, seed, args.max_samples,
         (wavelet, level, mode, tau, block_len, n_resamples, method, int(bootstrap["seed"])),
         TARGET_BANDS_CACHE)
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
    model_values: dict[str, dict[str, list[np.ndarray]]] = {
        model: {m: [] for m in METRICS} for model in ("mlp", "vcnn")
    }
    for result in results:
        rows.extend(result["rows"])
        for m in METRICS:
            collected[m].append(result["diffs"][m])
            for model in ("mlp", "vcnn"):
                model_values[model][m].append(result["model_values"][model][m])
    n_configs = len(results)
    print(f"   configurations: {n_configs}")

    aggregate = {}
    for m in METRICS:
        pooled = np.concatenate(collected[m])
        stats = paired_summary(
            pooled, block_len, n_resamples, int(bootstrap["seed"]), method=method
        )
        for model in ("mlp", "vcnn"):
            values = np.concatenate(model_values[model][m])
            stats[f"{model}_mean"] = float(values.mean())
            stats[f"{model}_median"] = float(np.median(values))
        # ``ci_low``/``ci_high`` are the time-block interval of the mean, as
        # quoted in the paper; the spread of the individual paired differences
        # is reported separately because it answers a different question.
        stats["spread_low"] = float(np.percentile(pooled, 2.5))
        stats["spread_high"] = float(np.percentile(pooled, 97.5))
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
        print("   submitted summary (normalised convolutional outputs, superseded):")
        for metric in aggregate:
            if metric in old:
                print(f"      {metric:15s} old={old[metric]['mean_diff']:+.5f} "
                      f"new={aggregate[metric]['mean_diff']:+.5f}")
    print("   all checks passed" if not failures else f"   {failures} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

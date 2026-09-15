#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equal-GER pairs over the full 42,000-configuration search space.

Search space:
  MLP   20 configurations x 3 training seeds = 60 runs
  VCNN  20 configurations x 3 training seeds = 60 runs
  Ridge 20 configurations, closed form   = 20 runs

The AdamW ridge runs of the retired sweep are no longer used; the closed-form
ridge replaces them.

Output
------
artifacts/statistics/equal_ger_pairs.json         pairs, statistics,
                                                 representative pair
artifacts/statistics/equal_ger_pairs_strict.json  per-configuration summary,
                                                 cross-model pairs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.statistics.paired_matching import (  # noqa: E402
    compute_cluster_bootstrap_ci,
    compute_paired_stats,
    find_internal_pairs,
)

OUT_DIR = ROOT / "artifacts" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W, C = 80, 160, 2
MASK_NUMS = [10, 15, 20, 30, 50]
NOISE_SIGMAS = [0.0, 0.001, 0.01, 0.1]
SEEDS = [0, 101, 202]
EPS = 1e-12


def sigma_to_code(sigma):
    if sigma == 0.0:
        return "s0000"
    elif sigma == 0.001:
        return "s0010"
    elif sigma == 0.01:
        return "s0100"
    elif sigma == 0.1:
        return "s1000"
    return f"s{int(sigma*10000):04d}"


def get_npz_path(model_type, mask_num, seed, sigma):
    """Learned models (MLP/VCNN) use the existing NPZ; closed-form Ridge the new one."""
    sigma_code = sigma_to_code(sigma)
    if model_type == "vcnn":
        if seed == 0:
            return (ROOT / "artifacts" / "vcnn_results" / "vcnn_sweep_nc_2000"
                    / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
        return (ROOT / "artifacts" / "vcnn_results"
                / f"vcnn_sweep_nc_2000_seed{seed:03d}"
                / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
    if model_type == "ridge":
        # Closed-form Ridge: deterministic, seed000 only; independent of the training seed
        return (ROOT / "artifacts" / "ridge_closed_form_sweep_nc"
                / f"ridge_n{mask_num:04d}" / "seed000" / "tests" / sigma_code / "test_raw.npz")
    # mlp
    return (ROOT / "artifacts" / "pod_model_sweep_nc"
            / f"{model_type}_n{mask_num:04d}" / f"seed{seed:03d}"
            / "tests" / sigma_code / "test_raw.npz")


def main() -> None:
    t_start = time.time()
    print("=" * 72)
    print("  42,000-record equal-GER full recomputation (closed-form Ridge)")
    print("=" * 72)

    # Skip when the artifacts already exist (local make only checks + plots; --force recomputes)
    json_fig3 = OUT_DIR / "equal_ger_pairs.json"
    json_s33 = OUT_DIR / "equal_ger_pairs_strict.json"
    if json_fig3.exists() and json_s33.exists() and "--force" not in sys.argv:
        print(f"[skip] {json_fig3.name} / {json_s33.name} already exist (--force to recompute)")
        return 0

    # ── Phase 1: load the 140 configuration metrics ────────────
    print("\n[Phase 1] scanning the 140 configs (MLP 60 + VCNN 60 + closed-form Ridge 20)...")
    all_config_metrics = {}
    skipped = 0
    n_searched = 0
    for model in ("mlp", "vcnn", "ridge"):
        for mask in MASK_NUMS:
            for sigma in NOISE_SIGMAS:
                seeds = [0] if model == "ridge" else SEEDS   # closed-form Ridge: 1 run only
                for seed in seeds:
                    npz_path = get_npz_path(model, mask, seed, sigma)
                    if not npz_path.exists():
                        skipped += 1
                        print(f"  [skip] missing: {npz_path}")
                        continue
                    data = dict(np.load(str(npz_path)))
                    out = data["output_nchw"]
                    tgt = data["target_nchw"]
                    B = out.shape[0]
                    key = f"{model}_M{mask}_σ{sigma}_seed{seed}"
                    ger_list, sfull_list, metrics_list = [], [], []
                    for i in range(B):
                        m = compute_sample_metrics(out, tgt, i)
                        ger_list.append(m["GER"])
                        sfull_list.append(m["S_full"])
                        metrics_list.append(m)
                    all_config_metrics[key] = {
                        "model": model, "mask": mask, "sigma": sigma, "seed": seed,
                        "n_samples": B, "ger": ger_list, "sfull": sfull_list,
                        "metrics": metrics_list,
                    }
                    n_searched += B
    print(f"  configurations: {len(all_config_metrics)} (skipped {skipped}), samples: {n_searched}")

    # ── Phase 2: within-config matching ────────────────────────
    print("\n[Phase 2] within-config matching...")
    all_pairs_data = []
    config_pair_counts = {}
    for key, cfg in all_config_metrics.items():
        pairs = find_internal_pairs(cfg["ger"], cfg["sfull"])
        config_pair_counts[key] = len(pairs)
        for p in pairs:
            p["metrics_low"] = cfg["metrics"][p["idx_low"]]
            p["metrics_high"] = cfg["metrics"][p["idx_high"]]
            p["config_key"] = key
            all_pairs_data.append(p)
    n_total_pairs = len(all_pairs_data)
    print(f"  within-config pairs: {n_total_pairs}")

    # Per-model counts
    from collections import defaultdict
    by_model = defaultdict(int)
    for p in all_pairs_data:
        by_model[p["config_key"].split("_")[0]] += 1
    print(f"  by model: {dict(by_model)}")

    # ── Phase 3: cross-model matching (same as s33) ───────────
    print("\n[Phase 3] cross-model matching (same snapshot)...")
    cross_model_pairs = []
    for mask in MASK_NUMS:
        for sigma in NOISE_SIGMAS:
            for seed in SEEDS:
                md = {}
                for model in ("mlp", "vcnn", "ridge"):
                    key = f"{model}_M{mask}_σ{sigma}_seed{seed}"
                    if key in all_config_metrics:
                        md[model] = all_config_metrics[key]
                    elif model == "ridge":
                        # Closed-form Ridge: deterministic, comparable with any VCNN seed
                        key0 = f"ridge_M{mask}_σ{sigma}_seed0"
                        if key0 in all_config_metrics:
                            md["ridge"] = all_config_metrics[key0]
                if len(md) < 2:
                    continue
                for (mA, mB) in (("mlp", "vcnn"), ("ridge", "vcnn")):
                    if mA in md and mB in md:
                        g1 = np.asarray(md[mA]["ger"])
                        s1 = np.asarray(md[mA]["sfull"])
                        g2 = np.asarray(md[mB]["ger"])
                        s2 = np.asarray(md[mB]["sfull"])
                        for i in range(len(g1)):
                            max_ger = max(g1[i], g2[i], EPS)
                            ger_diff = abs(g1[i] - g2[i]) / max_ger
                            sfull_diff = abs(int(s1[i]) - int(s2[i]))
                            if ger_diff <= 0.01 and sfull_diff >= 2:
                                cross_model_pairs.append({
                                    "model_low": mA, "model_high": mB,
                                    "M": mask, "sigma": sigma, "seed": seed,
                                    "snapshot": int(i),
                                    "ger_low": float(g1[i]), "ger_high": float(g2[i]),
                                    "sfull_low": int(s1[i]), "sfull_high": int(s2[i]),
                                    "ger_diff": float(ger_diff),
                                })
    print(f"  cross-model pairs: {len(cross_model_pairs)}")

    # ── Phase 4: paired statistics + bootstrap CI ─────────────
    print("\n[Phase 4] paired statistics...")
    paired_stats = compute_paired_stats(all_pairs_data)
    cluster_ci = compute_cluster_bootstrap_ci(all_pairs_data, n_bootstrap=10000)
    for t in paired_stats["tests"]:
        print(f"    {t['label']}: median_low={t['median_low_S_full']:.5f} "
              f"median_high={t['median_high_S_full']:.5f} "
              f"median_diff={t['median_diff']:.5f} "
              f"wilcoxon_p={t.get('wilcoxon_p_value')}")
    for r in cluster_ci["results"]:
        print(f"    {r['label']}: obs={r['observed_median_diff']:.5f} "
              f"CI=[{r['ci_lower']:.5f}, {r['ci_upper']:.5f}] p={r['p_value_bootstrap']:.4f}")

    # pct_pairs_lower (fraction of pairs where the high S_full sample has lower error)
    for lo_key, hi_key, name in [("E_W1", "E_W1", "W1"), ("vorticity_RMSE", "vorticity_RMSE", "vort"),
                                 ("gradient_RMSE", "gradient_RMSE", "grad")]:
        pct = round(100 * sum(p["metrics_high"][hi_key] < p["metrics_low"][lo_key]
                              for p in all_pairs_data) / len(all_pairs_data), 1)
        print(f"    pct_pairs_lower_{name}: {pct}%")

    # ── Phase 5: representative pair ───────────────────────────
    print("\n[Phase 5] representative pair (smallest GER_diff)...")
    all_pairs_data.sort(key=lambda p: p["GER_diff"])
    best = all_pairs_data[0]
    rep_pair = {
        "GER_low": best["GER_low"], "GER_high": best["GER_high"],
        "S_full_low": best["S_full_low"], "S_full_high": best["S_full_high"],
        "S_full_diff": best["S_full_diff"], "GER_diff": best["GER_diff"],
        "vorticity_RMSE_low": best["metrics_low"]["vorticity_RMSE"],
        "vorticity_RMSE_high": best["metrics_high"]["vorticity_RMSE"],
        "gradient_RMSE_low": best["metrics_low"]["gradient_RMSE"],
        "gradient_RMSE_high": best["metrics_high"]["gradient_RMSE"],
        "config_key": best["config_key"],
    }
    print(f"    {best['config_key']}: GER {rep_pair['GER_low']:.6f}/{rep_pair['GER_high']:.6f}, "
          f"S_full {rep_pair['S_full_low']}/{rep_pair['S_full_high']}")

    # ── Phase 6: write fig03_data.json ─────────────────────────
    output_pairs = [{
        "config_key": p["config_key"], "idx_low": p["idx_low"], "idx_high": p["idx_high"],
        "GER_low": p["GER_low"], "GER_high": p["GER_high"],
        "S_full_low": p["S_full_low"], "S_full_high": p["S_full_high"],
        "GER_diff": p["GER_diff"], "S_full_diff": p["S_full_diff"],
        "W1_low": p["metrics_low"]["E_W1"], "W1_high": p["metrics_high"]["E_W1"],
        "vorticity_RMSE_low": p["metrics_low"]["vorticity_RMSE"],
        "vorticity_RMSE_high": p["metrics_high"]["vorticity_RMSE"],
        "gradient_RMSE_low": p["metrics_low"]["gradient_RMSE"],
        "gradient_RMSE_high": p["metrics_high"]["gradient_RMSE"],
    } for p in all_pairs_data]

    result = {
        "task": "S01b_42k_ridge_closedform",
        "description": "Strict within-configuration equal-GER matching, 42,000 records, closed-form Ridge",
        "matching_criteria": {
            "same_M": True, "same_sigma": True, "same_seed": True, "same_model": True,
            "different_snapshots": True, "ger_tolerance": "1% (relative)",
            "min_sfull_gap": 2, "one_to_one_no_reuse": True,
        },
        "search_scope": "42,000 records = (MLP+VCNN: 40 configs x 3 seeds) + (Ridge: 20 closed-form deterministic)",
        "n_configs_searched": len(all_config_metrics),
        "n_samples_searched": n_searched,
        "n_configs_skipped": skipped,
        "n_pairs_found": n_total_pairs,
        "config_pair_counts": config_pair_counts,
        "representative_pair": rep_pair,
        "paired_statistics": paired_stats,
        "cluster_bootstrap_ci": cluster_ci,
        "all_pairs": output_pairs,
        "supersedes": "artifacts/derived/main/statistics/fig03_data.json "
                      "(AdamW-Ridge pairs retired)",
    }
    json_path = OUT_DIR / "equal_ger_pairs.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: {json_path}")

    # ── Phase 7: write s33 summary (within + cross-model) ──────
    internal_examples = {}
    for key, val in all_config_metrics.items():
        n_int = config_pair_counts.get(key, 0)
        if n_int > 0:
            pairs = find_internal_pairs(val["ger"], val["sfull"])
            first = pairs[0] if pairs else None
            internal_examples[key] = {
                "n_pairs": n_int,
                "ger_mean": float(np.mean(val["ger"])),
                "sfull_mean": float(np.mean(val["sfull"])),
                "first_pair": first,
            }
    s33_report = {
        "task": "S3.3b",
        "description": "Strict within-configuration equal-GER matching (42,000-configuration space, closed-form Ridge)",
        "matching_criteria": {
            "same_M": True, "same_sigma": True, "same_seed": True,
            "same_model_type": "for internal pairs; cross-model for cross pairs",
            "same_snapshot": "for cross-model pairs; different for internal",
            "ger_tolerance": "1% (relative)", "min_sfull_gap": 2,
            "one_to_one": True, "no_reuse": True,
        },
        "n_internal_pairs": n_total_pairs,
        "n_cross_model_pairs": len(cross_model_pairs),
        "internal_pair_examples": internal_examples,
        "cross_model_examples": cross_model_pairs[:20],
        "supersedes": "artifacts/derived/main/statistics/s33_strict_equal_ger.json "
                      "(AdamW-Ridge pairs retired)",
        "evaluation": (
            f"Found {n_total_pairs} strict within-configuration internal pairs "
            f"and {len(cross_model_pairs)} cross-model pairs (42,000 records, closed-form Ridge)."
        ),
    }
    s33_path = OUT_DIR / "equal_ger_pairs_strict.json"
    s33_path.write_text(json.dumps(s33_report, indent=2), encoding="utf-8")
    print(f"  JSON: {s33_path}")

    # The figure itself is drawn by applications/figures/fig04_equal_ger.py from
    # the JSON written above; this producer only computes the numbers.

    print(f"\n{'=' * 72}")
    print(f"  done ({time.time() - t_start:.1f}s)")
    print(f"  within-config: {n_total_pairs} pairs (MLP {by_model.get('mlp', 0)} "
          f"/ VCNN {by_model.get('vcnn', 0)} / Ridge {by_model.get('ridge', 0)})")
    print(f"  cross-model: {len(cross_model_pairs)} pairs")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

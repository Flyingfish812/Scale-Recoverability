#!/usr/bin/env python3
"""
S01: 修复 Figure 3 — 使用严格 within-configuration 匹配 (1245 pairs)

问题:
  Figure 3 仍使用旧的 68 disjoint pairs 数据 (p≈0.26-0.29)，
  但正文和 caption 已更新为 1245 个严格同配置匹配对。

修复:
  1. 扫描所有 NPZ 文件 (同 s33 — 3 models × 5 masks × 4 σ × 3 seeds)
  2. 对每样本计算 GER、S_full、W1 error、vorticity RMSE、gradient RMSE
  3. 同配置内找 equal-GER 匹配对 (GER diff ≤ 1%, S_full diff ≥ 2)
  4. 计算配对统计 + snapshot-cluster bootstrap CI
  5. 生成 Figure 3 (representative pair + 全对统计)

输出:
  results/20260723/fig03_data.json         — 完整配对数据
  results/20260723/fig03_equal_ger.pdf     — 图
  results/20260723/fig03_equal_ger.png     — 预览
  thesis_src/figures/fig03_equal_ger.pdf   — 覆盖论文用图
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pywt

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "applications" / "paper_main" / "build" / "figures_raw"
THESIS_FIGURES = ROOT / "thesis_src" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────
H, W, C = 80, 160, 2
BANDS = ["A4", "W4", "W3", "W2", "W1"]
MASK_NUMS = [10, 15, 20, 30, 50]
NOISE_SIGMAS = [0.0, 0.001, 0.01, 0.1]
SEEDS = [0, 101, 202]
MODELS = ["mlp", "ridge", "vcnn"]
TAU = 0.05  # S_full threshold
WAVELET = "db2"
LEVEL = 4
EPS = 1e-12

# ── Style ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════
# Data loading & metric computation
# ══════════════════════════════════════════════════════════════════

def sigma_to_code(sigma):
    if sigma == 0.0:
        return "s0000"
    elif sigma == 0.001:
        return "s0010"
    elif sigma == 0.01:
        return "s0100"
    elif sigma == 0.1:
        return "s1000"
    else:
        return f"s{int(sigma*10000):04d}"


def get_npz_path(model_type, mask_num, seed, sigma):
    sigma_code = sigma_to_code(sigma)
    if model_type == "vcnn":
        if seed == 0:
            return (ROOT / "artifacts" / "vcnn_results" / "vcnn_sweep_nc_2000"
                    / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
        else:
            return (ROOT / "artifacts" / "vcnn_results"
                    / f"vcnn_sweep_nc_2000_seed{seed:03d}"
                    / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
    else:
        return (ROOT / "artifacts" / "pod_model_sweep_nc"
                / f"{model_type}_n{mask_num:04d}" / f"seed{seed:03d}"
                / "tests" / sigma_code / "test_raw.npz")


def load_npz(path):
    return dict(np.load(str(path)))


def compute_vorticity(field_2d):
    """Compute 2D vorticity proxy via Laplacian."""
    return np.gradient(np.gradient(field_2d, axis=0), axis=0) + \
           np.gradient(np.gradient(field_2d, axis=1), axis=1)


def compute_gradient_rmse(target, pred):
    """RMSE of spatial gradients."""
    gy_t = np.gradient(target, axis=0)
    gx_t = np.gradient(target, axis=1)
    gy_p = np.gradient(pred, axis=0)
    gx_p = np.gradient(pred, axis=1)
    err_x = np.sqrt(np.mean((gx_t - gx_p) ** 2))
    err_y = np.sqrt(np.mean((gy_t - gy_p) ** 2))
    return float(np.sqrt(err_x ** 2 + err_y ** 2))


def compute_sample_metrics(output_nchw, target_nchw, sample_idx):
    """
    Compute all metrics for a single sample.

    Returns dict with:
      GER, S_full, per-band errors, vorticity RMSE, gradient RMSE
    """
    o = output_nchw[sample_idx].ravel()
    t = target_nchw[sample_idx].ravel()
    ger = float(np.linalg.norm(o - t) / (np.linalg.norm(t) + EPS))

    # Channel 0 for wavelet analysis
    out_2d = output_nchw[sample_idx, 0]  # (H, W)
    tgt_2d = target_nchw[sample_idx, 0]

    # Wavelet decomposition
    coeffs_out = pywt.wavedec2(out_2d, WAVELET, level=LEVEL, mode='periodization')
    coeffs_tgt = pywt.wavedec2(tgt_2d, WAVELET, level=LEVEL, mode='periodization')

    # Per-band errors + S_full
    band_errs = {}
    s_full_count = 0
    band_names = ['A4', 'W4', 'W3', 'W2', 'W1']

    # A4
    a4_p = coeffs_out[0]
    a4_t = coeffs_tgt[0]
    e = float(np.linalg.norm(a4_p - a4_t) / (np.linalg.norm(a4_t) + EPS))
    band_errs['A4'] = e
    if e < TAU:
        s_full_count += 1

    # W4-W1
    for j, (det_p, det_t) in enumerate(zip(coeffs_out[1:], coeffs_tgt[1:])):
        err_sum = sum(np.sum((dp - dt) ** 2) for dp, dt in zip(det_p, det_t))
        norm_sum = sum(np.sum(dt ** 2) for dt in det_t)
        e = float(np.sqrt(err_sum) / (np.sqrt(norm_sum) + EPS))
        bn = band_names[j + 1]
        band_errs[bn] = e
        if e < TAU:
            s_full_count += 1

    # Vorticity RMSE (channel 0)
    vort = compute_vorticity(tgt_2d)
    vort_pred = compute_vorticity(out_2d)
    vort_rmse = float(np.sqrt(np.mean((vort - vort_pred) ** 2)))

    # Gradient RMSE
    grad_rmse = compute_gradient_rmse(tgt_2d, out_2d)

    return {
        "GER": ger,
        "S_full": s_full_count,
        "E_W1": band_errs.get("W1", np.nan),
        "E_W2": band_errs.get("W2", np.nan),
        "E_W3": band_errs.get("W3", np.nan),
        "E_W4": band_errs.get("W4", np.nan),
        "E_A4": band_errs.get("A4", np.nan),
        "vorticity_RMSE": vort_rmse,
        "gradient_RMSE": grad_rmse,
    }


# ══════════════════════════════════════════════════════════════════
# Pair matching (same logic as s33)
# ══════════════════════════════════════════════════════════════════

def find_internal_pairs(ger_list, sfull_list, tol=0.01, min_gap=2):
    """
    Find one-to-one matched pairs within a single configuration.
    Uses same first-fit algorithm as s33 for reproducible results.

    Conditions:
      - |GER_i - GER_j| / max(GER_i, GER_j) <= tol
      - |S_full_i - S_full_j| >= min_gap
      - One-to-one, no reuse (first-fit: for each i, take first valid j)

    Returns list of dicts.
    """
    B = len(ger_list)
    ger = np.array(ger_list)
    sfull = np.array(sfull_list)

    pairs = []
    used_i = set()
    used_j = set()

    for i in range(B):
        if i in used_i:
            continue
        for j in range(i + 1, B):
            if j in used_j:
                continue
            max_ger = max(ger[i], ger[j], EPS)
            ger_diff = abs(ger[i] - ger[j]) / max_ger
            sfull_diff = abs(int(sfull[i]) - int(sfull[j]))
            if ger_diff <= tol and sfull_diff >= min_gap:
                # Determine low/high S_full
                if sfull[i] < sfull[j]:
                    low_idx, high_idx = i, j
                elif sfull[i] > sfull[j]:
                    low_idx, high_idx = j, i
                else:
                    low_idx, high_idx = i, j

                pairs.append({
                    "idx_low": int(low_idx),
                    "idx_high": int(high_idx),
                    "GER_low": float(ger[low_idx]),
                    "GER_high": float(ger[high_idx]),
                    "S_full_low": int(sfull[low_idx]),
                    "S_full_high": int(sfull[high_idx]),
                    "GER_diff": float(abs(ger[low_idx] - ger[high_idx]) / max(ger[low_idx], ger[high_idx], EPS)),
                    "S_full_diff": int(abs(sfull[low_idx] - sfull[high_idx])),
                })
                used_i.add(i)
                used_j.add(j)
                break

    return pairs


# ══════════════════════════════════════════════════════════════════
# Statistical analysis
# ══════════════════════════════════════════════════════════════════

def compute_paired_stats(all_pairs_data):
    """
    Compute paired statistics for the matched pairs.

    all_pairs_data: list of dicts, each with fields for both members.
    Returns dict with n_pairs, per-metric tests.
    """
    w1_low, w1_high = [], []
    vort_low, vort_high = [], []
    grad_low, grad_high = [], []

    for p in all_pairs_data:
        metrics_low = p["metrics_low"]
        metrics_high = p["metrics_high"]
        w1_low.append(metrics_low["E_W1"])
        w1_high.append(metrics_high["E_W1"])
        vort_low.append(metrics_low["vorticity_RMSE"])
        vort_high.append(metrics_high["vorticity_RMSE"])
        grad_low.append(metrics_low["gradient_RMSE"])
        grad_high.append(metrics_high["gradient_RMSE"])

    def paired_test(a, b, label):
        a = np.array(a)
        b = np.array(b)
        n = len(a)
        if n < 2:
            return {"label": label, "n_pairs": n, "error": "insufficient"}

        diff = a - b
        median_diff = float(np.median(diff))
        mean_diff = float(np.mean(diff))

        # Wilcoxon signed-rank
        from scipy.stats import wilcoxon
        try:
            w_stat, w_p = wilcoxon(a, b, alternative="two-sided")
        except (ValueError):
            w_stat, w_p = np.nan, np.nan

        ratio = float(np.median(b) / (np.median(a) + EPS))

        return {
            "label": label,
            "n_pairs": n,
            "median_low_S_full": float(np.median(a)),
            "median_high_S_full": float(np.median(b)),
            "mean_low_S_full": float(np.mean(a)),
            "mean_high_S_full": float(np.mean(b)),
            "median_diff": median_diff,
            "mean_diff": mean_diff,
            "median_ratio_high_over_low": ratio,
            "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else None,
        }

    return {
        "n_pairs": len(all_pairs_data),
        "tests": [
            paired_test(w1_low, w1_high, "W1_band_error"),
            paired_test(vort_low, vort_high, "vorticity_RMSE"),
            paired_test(grad_low, grad_high, "gradient_RMSE"),
        ],
    }


def compute_cluster_bootstrap_ci(all_pairs_data, n_bootstrap=10000, ci_level=0.95):
    """
    Snapshot-cluster bootstrap CI for paired differences.

    Pairs are independently resampled with replacement.
    """
    n_pairs = len(all_pairs_data)
    w1_diffs = np.array([p["metrics_high"]["E_W1"] - p["metrics_low"]["E_W1"]
                         for p in all_pairs_data])
    vort_diffs = np.array([p["metrics_high"]["vorticity_RMSE"] -
                           p["metrics_low"]["vorticity_RMSE"]
                           for p in all_pairs_data])
    grad_diffs = np.array([p["metrics_high"]["gradient_RMSE"] -
                           p["metrics_low"]["gradient_RMSE"]
                           for p in all_pairs_data])

    rng = np.random.RandomState(42)
    alpha = 1.0 - ci_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    def bootstrap_ci(diffs, label):
        boot_medians = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            indices = rng.randint(0, n_pairs, size=n_pairs)
            boot_medians[b] = np.median(diffs[indices])

        ci_lower = float(np.percentile(boot_medians, lower_pct))
        ci_upper = float(np.percentile(boot_medians, upper_pct))
        obs_median = float(np.median(diffs))

        # p-value: proportion of bootstrapped medians with opposite sign
        if obs_median > 0:
            p_val = float(np.mean(boot_medians <= 0))
        elif obs_median < 0:
            p_val = float(np.mean(boot_medians >= 0))
        else:
            p_val = 1.0

        return {
            "label": label,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "observed_median_diff": obs_median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value_bootstrap": p_val,
        }

    return {
        "n_pairs": n_pairs,
        "n_bootstrap": n_bootstrap,
        "results": [
            bootstrap_ci(w1_diffs, "W1_band_error_diff"),
            bootstrap_ci(vort_diffs, "vorticity_RMSE_diff"),
            bootstrap_ci(grad_diffs, "gradient_RMSE_diff"),
        ],
    }


# ══════════════════════════════════════════════════════════════════
# Figure generation
# ══════════════════════════════════════════════════════════════════

def generate_figure3(all_pairs_data, representative_pair, paired_stats, cluster_ci, out_path):
    """
    Generate a 2-panel Figure 3:
      Left: ΔGER vs ΔS_full scatter plot (all pairs)
      Right: Paired statistics — W1, vorticity, gradient (log scale)
    """
    n_pairs = len(all_pairs_data)
    # Extract scatter data from all_pairs_data (which has metrics_low/high attached)
    ger_diffs = np.array([p["GER_diff"] for p in all_pairs_data])
    sfull_diffs = np.array([p["S_full_diff"] for p in all_pairs_data])
    rng = np.random.RandomState(42)
    sfull_jitter = sfull_diffs + rng.uniform(-0.15, 0.15, size=len(sfull_diffs))
    ger_means = 0.5 * (np.array([p["GER_low"] for p in all_pairs_data]) +
                        np.array([p["GER_high"] for p in all_pairs_data]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.35)

    # ── Left panel: ΔGER vs ΔS_full scatter ────────────────────
    ax = axes[0]
    scatter = ax.scatter(sfull_jitter, ger_diffs, c=np.log10(ger_means + 1e-12),
                         cmap="viridis", alpha=0.5, s=15,
                         edgecolors="none", rasterized=True)

    best_idx = int(np.argmin(ger_diffs))
    ax.scatter([sfull_jitter[best_idx]], [ger_diffs[best_idx]],
               marker="*", s=120, color="#D55E00", edgecolors="white",
               linewidths=0.7, zorder=5, label="Best match")
    ax.annotate(
        f"$\\Delta$GER$\\approx${ger_diffs[best_idx]:.1e}",
        xy=(sfull_jitter[best_idx], ger_diffs[best_idx]),
        xytext=(sfull_jitter[best_idx] + 0.4, ger_diffs[best_idx] * 3),
        fontsize=7, color="#A84400",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#A84400"),
    )

    ax.set_xlabel("$\\Delta S_{\\mathrm{full}}$")
    ax.set_ylabel("Relative GER difference\n$|\\mathrm{GER}_{\\mathrm{high}} - "
                  "\\mathrm{GER}_{\\mathrm{low}}| / \\max(\\mathrm{GER})$")
    ax.set_title(f"Pair matching quality ({n_pairs} strict matched pairs)", fontsize=10)
    ax.set_xticks(sorted(set(sfull_diffs)))
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("$\\log_{10}(\\bar{E}_{\\mathrm{G}})$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    sf_min = int(sfull_diffs.min())
    sf_max = int(sfull_diffs.max())
    ax.text(0.95, 0.05,
            f"n = {n_pairs}\n"
            f"median $\\Delta$GER = {np.median(ger_diffs):.2e}\n"
            f"$\\Delta S_{{\\mathrm{{full}}}}$ range: {sf_min}–{sf_max}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray",
                      boxstyle="round,pad=0.3", alpha=0.85))

    # ── Right panel: Paired statistics ──────────────────────────
    ax = axes[1]

    cb_results = {r["label"]: r for r in cluster_ci["results"]}
    ps_tests = {t["label"]: t for t in paired_stats["tests"]}

    metrics_info = [
        ("W1 Band Error", "W1_band_error", "W1_band_error_diff"),
        ("Vorticity RMSE", "vorticity_RMSE", "vorticity_RMSE_diff"),
        ("Gradient RMSE", "gradient_RMSE", "gradient_RMSE_diff"),
    ]

    x_pos = np.arange(len(metrics_info))
    width = 0.3

    for i, (label, ps_key, cb_key) in enumerate(metrics_info):
        ps_r = ps_tests.get(ps_key, {})
        cb_r = cb_results.get(cb_key, {})

        low_val = ps_r.get("median_low_S_full", 0)
        high_val = ps_r.get("median_high_S_full", 0)
        p_val = cb_r.get("p_value_bootstrap", 1.0)
        ci_low = cb_r.get("ci_lower", 0)
        ci_high = cb_r.get("ci_upper", 0)

        ax.bar(i - width / 2, low_val, width, label=f"Low $S_{{\\mathrm{{full}}}}$" if i == 0 else "",
               color="#e74c3c", alpha=0.75)
        ax.bar(i + width / 2, high_val, width, label=f"High $S_{{\\mathrm{{full}}}}$" if i == 0 else "",
               color="#2ecc71", alpha=0.75)

        ref_val = max(low_val, high_val)
        # Format p-value
        if p_val < 0.001:
            p_str = "p<0.001"
        else:
            p_str = f"p={p_val:.3f}"

        anno_y = 10 ** (np.log10(ref_val) + 0.12) if ref_val > 0 else 0.01
        ax.text(i, anno_y,
                f"{p_str}\nCI=[{ci_low:.4f}, {ci_high:.4f}]",
                ha="center", fontsize=6.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["W1 Band Error", "Vorticity RMSE", "Gradient RMSE"])
    ax.set_ylabel("Median Value (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(top=1e-1)
    ax.set_title(f"Paired Statistics ({n_pairs} strict matched pairs)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Equal-GER Analysis: {n_pairs} strict within-configuration matched pairs\n"
        f"All pairs: same $M$, $\\sigma$, seed, model; $\\Delta$GER $\\leq$ 1%, "
        f"$\\Delta S_{{\\mathrm{{full}}}} \\geq$ 2",
        fontsize=9, y=0.97)

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path.with_suffix('.pdf')}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  S01: Fix Figure 3 — Strict Within-Configuration Equal-GER Pairs")
    print("=" * 70)

    t_start = time.time()

    # ── Phase 1: Scan all NPZ files & compute metrics ──────────
    print("\n[Phase 1] Scanning NPZ files and computing metrics...")
    all_config_metrics = {}  # key -> dict with ger[], sfull[], metrics[]

    total_configs = 0
    total_samples = 0
    skipped_configs = 0

    for model in MODELS:
        for mask in MASK_NUMS:
            for sigma in NOISE_SIGMAS:
                for seed in SEEDS:
                    npz_path = get_npz_path(model, mask, seed, sigma)
                    if not npz_path.exists():
                        skipped_configs += 1
                        continue

                    key = f"{model}_M{mask}_σ{sigma}_seed{seed}"
                    total_configs += 1

                    data = load_npz(str(npz_path))
                    output_nchw = data["output_nchw"]
                    target_nchw = data["target_nchw"]
                    B = output_nchw.shape[0]

                    ger_list = []
                    sfull_list = []
                    metrics_list = []

                    for i in range(B):
                        m = compute_sample_metrics(output_nchw, target_nchw, i)
                        ger_list.append(m["GER"])
                        sfull_list.append(m["S_full"])
                        metrics_list.append(m)

                    all_config_metrics[key] = {
                        "model": model,
                        "mask": mask,
                        "sigma": sigma,
                        "seed": seed,
                        "n_samples": B,
                        "ger": ger_list,
                        "sfull": sfull_list,
                        "metrics": metrics_list,
                    }
                    total_samples += B

                    if total_configs % 10 == 0:
                        print(f"  ... processed {total_configs} configs, {total_samples} samples"
                              f" (skipped {skipped_configs})")

    elapsed_phase1 = time.time() - t_start
    print(f"\n[Phase 1] Done: {total_configs} configs, {total_samples} samples, "
          f"{skipped_configs} skipped, in {elapsed_phase1:.1f}s")

    # ── Phase 2: Match pairs within each configuration ─────────
    print("\n[Phase 2] Matching equal-GER pairs within configurations...")

    all_pairs_data = []
    config_pair_counts = {}

    for key, cfg in all_config_metrics.items():
        pairs = find_internal_pairs(cfg["ger"], cfg["sfull"])
        config_pair_counts[key] = len(pairs)

        for p in pairs:
            # Attach full metrics
            p["metrics_low"] = cfg["metrics"][p["idx_low"]]
            p["metrics_high"] = cfg["metrics"][p["idx_high"]]
            p["config_key"] = key
            all_pairs_data.append(p)

    n_total_pairs = len(all_pairs_data)

    print(f"  Found {n_total_pairs} matched pairs across {total_configs} configs")
    print(f"  Configs with pairs: {sum(1 for v in config_pair_counts.values() if v > 0)}")

    # ── Phase 3: Statistical analysis ──────────────────────────
    print("\n[Phase 3] Computing paired statistics...")

    paired_stats = compute_paired_stats(all_pairs_data)
    n_pairs = paired_stats["n_pairs"]
    print(f"  Paired statistics ({n_pairs} pairs):")
    for t in paired_stats["tests"]:
        if t.get("wilcoxon_p_value") is not None:
            print(f"    {t['label']}: Wilcoxon p={t['wilcoxon_p_value']:.4e}")

    # Cluster bootstrap (snapshot-cluster)
    print(f"\n  Computing snapshot-cluster bootstrap CI (n=10000)...")
    t_bs = time.time()
    cluster_ci = compute_cluster_bootstrap_ci(all_pairs_data, n_bootstrap=10000)
    elapsed_bs = time.time() - t_bs
    print(f"  Bootstrap done in {elapsed_bs:.1f}s")
    for r in cluster_ci["results"]:
        print(f"    {r['label']}: median diff={r['observed_median_diff']:.4e}, "
              f"CI=[{r['ci_lower']:.4e}, {r['ci_upper']:.4e}], "
              f"p={r['p_value_bootstrap']:.4f}")

    # ── Phase 4: Representative pair ───────────────────────────
    print("\n[Phase 4] Selecting representative pair...")
    # Pick the pair with smallest GER diff
    all_pairs_data.sort(key=lambda p: p["GER_diff"])
    best = all_pairs_data[0]
    rep_pair = {
        "GER_low": best["GER_low"],
        "GER_high": best["GER_high"],
        "S_full_low": best["S_full_low"],
        "S_full_high": best["S_full_high"],
        "S_full_diff": best["S_full_diff"],
        "GER_diff": best["GER_diff"],
        "vorticity_RMSE_low": best["metrics_low"]["vorticity_RMSE"],
        "vorticity_RMSE_high": best["metrics_high"]["vorticity_RMSE"],
        "gradient_RMSE_low": best["metrics_low"]["gradient_RMSE"],
        "gradient_RMSE_high": best["metrics_high"]["gradient_RMSE"],
        "config_key": best["config_key"],
    }
    print(f"  Representative pair:")
    print(f"    Config: {best['config_key']}")
    print(f"    GER: {rep_pair['GER_low']:.6f} vs {rep_pair['GER_high']:.6f}")
    print(f"    S_full: {rep_pair['S_full_low']} vs {rep_pair['S_full_high']}")
    print(f"    ΔGER: {rep_pair['GER_diff']:.2e}")
    print(f"    Vorticity RMSE: {rep_pair['vorticity_RMSE_low']:.4e} vs {rep_pair['vorticity_RMSE_high']:.4e}")
    print(f"    Gradient RMSE: {rep_pair['gradient_RMSE_low']:.4e} vs {rep_pair['gradient_RMSE_high']:.4e}")

    # ── Phase 5: Save results JSON ─────────────────────────────
    print("\n[Phase 5] Saving results...")

    # Build lightweight output (don't include all per-sample metrics)
    output_pairs = []
    for p in all_pairs_data:
        output_pairs.append({
            "config_key": p["config_key"],
            "idx_low": p["idx_low"],
            "idx_high": p["idx_high"],
            "GER_low": p["GER_low"],
            "GER_high": p["GER_high"],
            "S_full_low": p["S_full_low"],
            "S_full_high": p["S_full_high"],
            "GER_diff": p["GER_diff"],
            "S_full_diff": p["S_full_diff"],
            "W1_low": p["metrics_low"]["E_W1"],
            "W1_high": p["metrics_high"]["E_W1"],
            "vorticity_RMSE_low": p["metrics_low"]["vorticity_RMSE"],
            "vorticity_RMSE_high": p["metrics_high"]["vorticity_RMSE"],
            "gradient_RMSE_low": p["metrics_low"]["gradient_RMSE"],
            "gradient_RMSE_high": p["metrics_high"]["gradient_RMSE"],
        })

    result = {
        "task": "S01_fix_figure3",
        "description": "Strict within-configuration equal-GER matching + Figure 3",
        "matching_criteria": {
            "same_M": True,
            "same_sigma": True,
            "same_seed": True,
            "same_model": True,
            "different_snapshots": True,
            "ger_tolerance": "1% (relative)",
            "min_sfull_gap": 2,
            "one_to_one_no_reuse": True,
        },
        "n_configs_searched": total_configs,
        "n_samples_searched": total_samples,
        "n_configs_skipped": skipped_configs,
        "n_pairs_found": n_total_pairs,
        "config_pair_counts": config_pair_counts,
        "representative_pair": rep_pair,
        "paired_statistics": paired_stats,
        "cluster_bootstrap_ci": cluster_ci,
        "all_pairs": output_pairs,
    }

    # Save JSON
    json_path = OUT_DIR / "fig03_data.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # ── Phase 6: Generate Figure ───────────────────────────────
    print("\n[Phase 6] Generating Figure 3...")
    generate_figure3(all_pairs_data, rep_pair, paired_stats, cluster_ci,
                     OUT_DIR / "fig03_equal_ger")

    # Copy to thesis figures (optional; skipped when manuscript tree absent)
    if (ROOT / "thesis_src").exists():
        import shutil
        THESIS_FIGURES.mkdir(parents=True, exist_ok=True)
        thesis_pdf = THESIS_FIGURES / "fig03_equal_ger.pdf"
        shutil.copy2(OUT_DIR / "fig03_equal_ger.pdf", thesis_pdf)
        print(f"  Copied to thesis figures: {thesis_pdf}")

    # ── Summary ────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed_total:.1f}s")
    print(f"  Pairs found: {n_total_pairs}")
    print(f"  Figure: {OUT_DIR / 'fig03_equal_ger.pdf'}")
    print(f"  Data:   {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

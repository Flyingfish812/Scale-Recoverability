#!/usr/bin/env python3
"""
P05: Figure 7 — Per-band error comparison (M=20, σ=0, closed-form Ridge)

修复: 将 Ridge 数据从 AdamW 替换为闭式 Ridge（s05_true_ridge.json）
数据源:
  - MLP/VCNN: three_layer_fixed.json (不变)
  - Ridge (closed-form): results/20260723/s05_true_ridge.json
  - Oracle: paper_facts.yaml 固定值

输出:
  results/20260723/fig07_per_band_comparison.pdf / .png
  thesis_src/figures/fig07_per_band_comparison.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
THESIS_FIGURES = ROOT / "thesis_src" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

BANDS = ["A4", "W4", "W3", "W2", "W1"]

# Oracle per-band errors (from paper_facts.yaml / FOracle* macros)
ORACLE_BAND = {"A4": 0.000096, "W4": 0.001114, "W3": 0.001966,
               "W2": 0.003813, "W1": 0.005081}


def _clipped_yerr(means, stds):
    """Asymmetric error bars clipped at zero (E_direct >= 0).

    Returns a (2, N) array of [lower, upper] error lengths, where the
    lower extent cannot take the interval below zero.
    """
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    lower = np.minimum(stds, means)
    return np.vstack([lower, stds])


def main():
    print("=" * 60)
    print("  P05: Figure 7 — Per-band error (closed-form Ridge)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n1. Loading data...")

    # Ridge (closed-form) from s05 (statistics pool)
    s05_path = Path(__file__).resolve().parents[4] / "artifacts" / "derived" / "main" / "statistics" / "s05_true_ridge.json"
    with open(s05_path) as f:
        s05 = json.load(f)
    ridge_closed = None
    for r in s05["results"]:
        if r["mask_num"] == 20 and r["sigma"] == 0.0:
            ridge_closed = r
            break
    if ridge_closed is None:
        print("[ERROR] No closed-form Ridge M=20, σ=0 data")
        return
    ridge_band = ridge_closed["per_band_mean"]
    print(f"  Ridge (closed-form): {ridge_band}")

    # MLP/VCNN from three_layer_fixed
    with open(ROOT / "artifacts/derived/main/statistics/three_layer_fixed.json") as f:
        tlf = json.load(f)
    results = tlf["results"]

    model_band_data = {}
    for mt in ["mlp", "ridge", "vcnn"]:
        samples = [r for r in results if r["model_type"] == mt
                   and r["mask_num"] == 20 and r["noise_sigma"] == 0.0]
        if samples:
            model_band_data[mt] = {}
            for b in BANDS:
                vals = []
                for r in samples:
                    e = r.get(f"E_total_{b}")
                    if e is not None and e > 0:
                        vals.append(e)
                model_band_data[mt][b] = (np.mean(vals), np.std(vals)) if vals else (0, 0)
            ger = np.mean([r.get("GER", 0) for r in samples])
            print(f"  {mt}: GER={ger:.4f}, bands={[f'{model_band_data[mt][b][0]:.4f}' for b in BANDS]}")

    # ── 2. Draw figure ────────────────────────────────────────
    print("\n2. Drawing figure...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    x = np.arange(len(BANDS))
    width = 0.18
    tau = 0.05

    # MLP bar
    mlp_means = [model_band_data["mlp"][b][0] for b in BANDS]
    mlp_stds = [model_band_data["mlp"][b][1] for b in BANDS]
    ax.bar(x - width, mlp_means, width, color="#348ABD", alpha=0.8, label="MLP")
    ax.errorbar(x - width, mlp_means, yerr=_clipped_yerr(mlp_means, mlp_stds),
                fmt="none", color="black", capsize=2, capthick=1, linewidth=0.8)

    # VCNN bar
    vcnn_means = [model_band_data["vcnn"][b][0] for b in BANDS]
    vcnn_stds = [model_band_data["vcnn"][b][1] for b in BANDS]
    ax.bar(x + width, vcnn_means, width, color="#988ED5", alpha=0.8, label="VCNN")
    ax.errorbar(x + width, vcnn_means, yerr=_clipped_yerr(vcnn_means, vcnn_stds),
                fmt="none", color="black", capsize=2, capthick=1, linewidth=0.8)

    # Ridge (closed-form) bar — 无标准差 (单次闭式解)
    ridge_means_closed = [ridge_band[b] for b in BANDS]
    ax.bar(x, ridge_means_closed, width, color="#E24A33", alpha=0.8,
           label="Ridge (closed-form)")

    # τ threshold line + fail region (P1-8: pass/fail at a glance)
    ax.axhline(y=tau, color="red", linestyle="--", linewidth=1.2)
    ax.axhspan(tau, 0.14, color="red", alpha=0.07, zorder=0)
    ax.text(0.02, tau + 0.002, f"fail $>\\tau={tau}$", color="red", fontsize=9,
            va="bottom", transform=ax.get_yaxis_transform())

    # First-failed-band markers (P1-8: conclusion-first)
    def first_fail(means):
        for k, m in enumerate(means):
            if m > tau:
                return k
        return None

    for off, means, color, name in [(-width, mlp_means, "#348ABD", "MLP"),
                                    (0.0, ridge_means_closed, "#E24A33", "Ridge"),
                                    (width, vcnn_means, "#988ED5", "VCNN")]:
        k = first_fail(means)
        if k is not None:
            ax.annotate(f"{name} fails at {BANDS[k]}", xy=(x[k] + off, means[k]),
                        xytext=(x[k] + off, 0.135), arrowprops=dict(arrowstyle="->",
                        color=color, lw=1.0), fontsize=8, color=color, ha="center")
        else:
            ax.annotate(f"{name}: all pass", xy=(x[0] + off, means[0]),
                        xytext=(x[-1] + off, 0.135), fontsize=8, color=color,
                        ha="center")

    # GER annotation for Ridge
    ger_text = f"Ridge GER = {ridge_closed['GER_mean']:.4f}"
    ax.text(0.98, 0.95, ger_text, transform=ax.transAxes, fontsize=8,
            ha="right", va="top", color="#E24A33",
            bbox=dict(facecolor="white", alpha=0.7, pad=2))

    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("Mean $E_{\\mathrm{direct}}(b)$")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylim(0, 0.14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Per-band reconstruction error ($M=20$, $\\sigma=0$):\n"
                 "Ridge fails from W3; MLP and VCNN pass every band")

    fig.tight_layout()

    # ── 3. Save ───────────────────────────────────────────────
    out_pdf = OUT_DIR / "fig07_per_band_comparison.pdf"
    out_png = OUT_DIR / "fig07_per_band_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\n  Saved: {out_pdf}")
    plt.close(fig)
    # 发布由 publish_figures 统一处理 (fig07_per_band_comparison → fig07_cross_model_bands)

    # ── 4. Delta table data ───────────────────────────────────
    print("\n3. Delta table (excess error over Oracle):")
    print(f"  {'Band':>5} | {'Oracle':>8} {'Ridge (closed)':>16} {'ΔE':>10}")
    print(f"  {'-'*5}-+-{'-'*8}-{'-'*16}-{'-'*10}")
    for b in BANDS:
        oracle_e = ORACLE_BAND[b]
        ridge_e = ridge_band[b]
        delta = ridge_e - oracle_e
        print(f"  {b:>5} | {oracle_e:>8.5f} {ridge_e:>16.4f} {delta:>+10.4f}")

    print("\n  ✓ Done")


if __name__ == "__main__":
    main()

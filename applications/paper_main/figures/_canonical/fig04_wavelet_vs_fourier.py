#!/usr/bin/env python3
"""
=============================================================================
20260725 重画 Figure 4 — 修复 P0-13 标注问题
=============================================================================

根据 20260725 审计 P0-13:
  - "Wavelet-band limited targets" → "Wavelet-truncated reconstructions"
  - "Fourier-annulus limited targets" → "Fourier-annulus-truncated reconstructions"
  - "Controlled Scale Validation" → "Controlled Truncation Consistency Check"

基础代码来自 scripts/20260716/regenerate_figures.py (fig04_wavelet_vs_fourier_physics)
数据源: results/20260715/ (symmetric_control.json, fourier_spectral_baseline.csv)

Output: results/20260725/fig04_wavelet_vs_fourier_physics.pdf/png
=============================================================================
"""

from __future__ import annotations

import json
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR_V5 = PROJECT_ROOT / "artifacts/derived/main/statistics"
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    print("=" * 60)
    print("  Figure 4 — 重画 (P0-13 修复)")
    print("=" * 60)

    # ── Panel (a): S_full vs S_FFT scatter from fb2 CSV ────────────
    csv_path = DATA_DIR_V5 / "fourier_spectral_baseline.csv"
    s_full_vals = []
    s_fft_vals = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_full_vals.append(int(float(row["S_full"])))
            s_fft_vals.append(int(float(row["S_FFT"])))

    s_full = np.array(s_full_vals)
    s_fft = np.array(s_fft_vals)

    # ── Panel (b): Controlled truncation check from symmetric_control ──
    with open(DATA_DIR_V5 / "symmetric_control.json") as f:
        sc = json.load(f)

    wbl_results = [r for r in sc["results"] if r["experiment"] == "wavelet_band_limited"]
    fal_results = [r for r in sc["results"] if r["experiment"] == "fourier_annulus_limited"]

    def accuracy_by_field(results):
        fields = set(r["field_idx"] for r in results if r["expected_recoverable"] is not None)
        accs = {}
        for fid in sorted(fields):
            fsubset = [r for r in results if r["field_idx"] == fid and r["expected_recoverable"] is not None]
            s_full_acc = sum(1 for r in fsubset if r["S_full_correct"]) / len(fsubset) * 100
            s_fft_acc = sum(1 for r in fsubset if r["S_FFT_correct"]) / len(fsubset) * 100
            accs[fid] = (s_full_acc, s_fft_acc)
        return accs

    wbl_acc = accuracy_by_field(wbl_results)
    fal_acc = accuracy_by_field(fal_results)

    # ── Build figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # ================================================================
    # Panel (a): S_full vs S_FFT scatter
    # ================================================================
    ax = axes[0]
    jit = np.random.RandomState(42).uniform(-0.15, 0.15, size=len(s_full))
    ax.scatter(s_full + jit, s_fft + jit, alpha=0.5, s=20, c="#2c3e50",
               edgecolors="white", linewidth=0.3)
    ax.plot([-0.5, 5.5], [-0.5, 5.5], "r--", linewidth=0.8, alpha=0.5,
            label="perfect agreement")

    for sv in range(6):
        for fv in range(6):
            count = ((s_full == sv) & (s_fft == fv)).sum()
            if count > 3:
                ax.text(sv + 0.2, fv + 0.2, str(count), fontsize=7,
                        ha="center", va="center", color="blue", alpha=0.6)

    with open(DATA_DIR_V5 / "fourier_spectral_baseline.json") as f:
        fb_json = json.load(f)
    corr = fb_json.get("correlations", {})

    ax.set_xlabel("S$_{\\mathrm{full}}$ (wavelet)")
    ax.set_ylabel("S$_{\\mathrm{FFT}}$ (Fourier dyadic)")
    ax.set_title("(a) Wavelet vs Fourier Scale Recoverability")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.legend(fontsize=7, loc="upper left")

    sfull_vs_sfft = corr.get("S_full_vs_S_FFT", {})
    sfull_vs_spec = corr.get("S_full_vs_spectral_loss", {})
    info_text = (
        f"$\\rho$(S$_{{\\mathrm{{full}}}}$, S$_{{\\mathrm{{FFT}}}}$) = {sfull_vs_sfft.get('spearman_rho', 0):.3f}\n"
        f"$\\rho$(S$_{{\\mathrm{{full}}}}$, L$_{{\\mathrm{{spectral}}}}$) = {sfull_vs_spec.get('spearman_rho', 0):.3f}"
    )
    ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

    # ================================================================
    # Panel (b): Controlled Truncation Consistency Check
    # ================================================================
    ax = axes[1]
    x = np.arange(2)
    width = 0.3

    wbl_sf_mean = np.mean([v[0] for v in wbl_acc.values()])
    wbl_fft_mean = np.mean([v[1] for v in wbl_acc.values()])
    fal_sf_mean = np.mean([v[0] for v in fal_acc.values()])
    fal_fft_mean = np.mean([v[1] for v in fal_acc.values()])

    ax.bar(x[0] - width / 2, wbl_sf_mean, width, label="S$_{\\mathrm{full}}$",
           color="#3498db", alpha=0.85)
    ax.bar(x[0] + width / 2, wbl_fft_mean, width, label="S$_{\\mathrm{FFT}}$",
           color="#e74c3c", alpha=0.85)
    ax.bar(x[1] - width / 2, fal_sf_mean, width, color="#3498db", alpha=0.85)
    ax.bar(x[1] + width / 2, fal_fft_mean, width, color="#e74c3c", alpha=0.85)

    for xi, (sf, ff) in enumerate([(wbl_sf_mean, wbl_fft_mean), (fal_sf_mean, fal_fft_mean)]):
        ax.text(xi - width / 2, sf + 2, f"{sf:.0f}%", ha="center", fontsize=7, color="#3498db")
        ax.text(xi + width / 2, ff + 2, f"{ff:.0f}%", ha="center", fontsize=7, color="#e74c3c")

    # ═══════════════════════════════════════════════════════════════
    # P0-13 fixes: update x-tick labels and title
    # ═══════════════════════════════════════════════════════════════
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Wavelet-truncated\nreconstructions",
        "Fourier-annulus-\ntruncated reconstructions",
    ])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Controlled Truncation Consistency Check\n(S$_{\\mathrm{full}}$ vs S$_{\\mathrm{FFT}}$)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Save ──────────────────────────────────────────────────────
    fig.savefig(OUT_DIR / "fig04_wavelet_vs_fourier_physics.pdf")
    fig.savefig(OUT_DIR / "fig04_wavelet_vs_fourier_physics.png", dpi=200)
    plt.close(fig)
    print(f"  ✓ Figure 4 saved to {OUT_DIR}/fig04_wavelet_vs_fourier_physics.pdf/png")


if __name__ == "__main__":
    main()

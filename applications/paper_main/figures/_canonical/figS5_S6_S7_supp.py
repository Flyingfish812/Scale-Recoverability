#!/usr/bin/env python3
"""
20260725 重画 Figure S5, S6, S7 — 修复 P0-27 及标题问题

修复内容:
  1. 所有图删除内部 "S-Fig N:" 标题前缀
  2. S5 (原 sfig6): 标题简化, 标注 "coarse bands show larger relative degradation"
  3. S6 (原 sfig7): Panel (a) 改为归一化 S_norm = S_full / (L+1)
  4. S7 (原 sfig8): "coherent-only" → "POD-component-only", "physical meaning" → "interpretive meaning"

基础代码来自 _legacy/tools_active/plot_supplementary_figures.py
数据源: _legacy/results/mechanism_analysis/

Output: results/20260725/sfig5_*.png, sfig6_*.png, sfig7_*.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
LEGACY_DATA = PROJECT_ROOT / "_legacy" / "results_old" / "results" / "mechanism_analysis"
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS_CF = ["A4", "W4", "W3", "W2", "W1"]
DATASET_COLORS = {"NC": "#2196F3", "RDB": "#FF5722", "SST": "#4CAF50"}
MODEL_COLORS = {"MLP": "#2196F3", "VCNN": "#FF5722", "Ridge": "#4CAF50"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


# ═══════════════════════════════════════════════════════════════════════════
# S5: Noise propagation (原 S-Fig 6)
# ═══════════════════════════════════════════════════════════════════════════
def plot_sfig5():
    """Noise propagation — per-band degradation ratio."""
    path = LEGACY_DATA / "noise_propagation.json"
    d = json.loads(path.read_text())

    model_degradation = {}
    for config_key, info in d["per_config"].items():
        mt = config_key.split("_")[0]
        if mt not in model_degradation:
            model_degradation[mt] = []
        model_degradation[mt].append(info["degradation_ratio"])

    model_labels = {"mlp": "MLP", "vcnn": "VCNN", "ridge": "Ridge"}
    model_order = ["mlp", "vcnn", "ridge"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(BANDS_CF))
    n_m = len(model_order)
    w = 0.25

    for i, mt in enumerate(model_order):
        if mt not in model_degradation:
            continue
        ratios_list = model_degradation[mt]
        band_means = []
        for b in BANDS_CF:
            vals = [r[b] for r in ratios_list]
            band_means.append(np.mean(vals))

        offset = (i - (n_m - 1) / 2) * w
        bars = ax.bar(x + offset, band_means, w, color=MODEL_COLORS[model_labels[mt]],
                      alpha=0.85, edgecolor='white', label=model_labels[mt])
        for bar, val in zip(bars, band_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}x", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_CF, fontsize=11)
    ax.set_ylabel("Degradation ratio\n(σ=0.1 error / σ=0 error)", fontsize=12)
    ax.set_title("Noise propagation across bands", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 修复 caption 用语: coarse bands show larger RELATIVE degradation
    ax.text(0.98, 0.95,
            "Coarse bands show larger relative\ndegradation ratios, partly because\ntheir clean-error denominators\nare much smaller.",
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle="round", facecolor='#FFF9C4', alpha=0.6))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sfig5_noise_propagation.png", dpi=200)
    fig.savefig(OUT_DIR / "sfig5_noise_propagation_hires.png", dpi=400)
    plt.close(fig)
    print("  ✓ S5 (noise propagation) saved.")


# ═══════════════════════════════════════════════════════════════════════════
# S6: Wavelet level sensitivity (原 S-Fig 7) — 修复归一化
# ═══════════════════════════════════════════════════════════════════════════
def plot_sfig6():
    """Wavelet decomposition level sensitivity — normalized S_full + band errors.

    P0-27 fix: Panel (a) uses S_norm = S_full / (L+1) so that levels
    with different max values (L3=4, L4=5, L5=6) are comparable.
    """
    path = LEGACY_DATA / "level_sensitivity.json"
    d = json.loads(path.read_text())

    levels = ["level_3", "level_4", "level_5"]
    level_labels = ["L3\n(4 bands)", "L4\n(5 bands)", "L5\n(6 bands)"]
    max_per_level = {"level_3": 4, "level_4": 5, "level_5": 6}
    x = np.arange(len(levels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[1, 2])

    # ═══════════════════════════════════════════════════════════════
    # Panel (a): Normalized S_full
    # ═══════════════════════════════════════════════════════════════
    s_full_means = [d[lv]["s_full_mean"] for lv in levels]
    # Normalize: S_norm = S_full / max_possible
    s_full_norm = [s_full_means[i] / max_per_level[levels[i]] for i in range(len(levels))]
    colors = ["#E3F2FD", "#2196F3", "#1565C0"]

    bars = ax1.bar(x, s_full_norm, 0.5, color=colors, edgecolor='white')
    for bar, val in zip(bars, s_full_norm):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(level_labels, fontsize=10)
    ax1.set_ylabel("Normalized S$_{\\mathrm{full}}$ / max", fontsize=12)
    ax1.set_title("(a) Normalized recoverability index", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # Add reference lines for s_full = max (1.0) at each level
    # Show raw values in annotation
    raw_text = "Raw S_full:  "
    for lv, lname in zip(levels, level_labels):
        raw = d[lv]["s_full_mean"]
        norm = raw / max_per_level[lv]
        raw_text += f"{lname.split(chr(10))[0]}={raw:.2f}(→{norm:.2f})  "
    ax1.text(0.5, -0.25, raw_text, transform=ax1.transAxes, fontsize=7,
             ha='center', va='top', style='italic')

    # ═══════════════════════════════════════════════════════════════
    # Panel (b): Per-band errors for L4
    # ═══════════════════════════════════════════════════════════════
    ax2.set_title("(b) Per-band error (level 4)", fontsize=12, fontweight='bold')
    l4 = d["level_4"]
    bands = l4["bands"]
    errs = [l4["per_band_mean_error"][b] for b in bands]
    band_colors = ["#1B5E20", "#43A047", "#81C784", "#A5D6A7", "#C8E6C9"]
    bars = ax2.bar(range(len(bands)), errs, 0.5, color=band_colors, edgecolor='white')
    ax2.axhline(y=0.05, color='red', linestyle='--', lw=1.0, alpha=0.7, label='τ=0.05')
    for bar, val, band in zip(bars, errs, bands):
        y_pos = bar.get_height() + 0.001 if val < 0.03 else bar.get_height() * 0.6
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:.4f}", ha='center', va='bottom' if val < 0.03 else 'top',
                 fontsize=8, fontweight='bold', color='white' if val > 0.03 else 'black')
    ax2.set_xticks(range(len(bands)))
    ax2.set_xticklabels(bands, fontsize=11)
    ax2.set_ylabel("Mean relative error", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 修复 caption: the increase in per-band error from coarse to fine bands
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle("Wavelet decomposition level sensitivity", fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(OUT_DIR / "sfig6_level_sensitivity.png", dpi=200)
    fig.savefig(OUT_DIR / "sfig6_level_sensitivity_hires.png", dpi=400)
    plt.close(fig)
    print("  ✓ S6 (level sensitivity) saved.")


# ═══════════════════════════════════════════════════════════════════════════
# S7: POD-component-only sample (原 S-Fig 8)
# ═══════════════════════════════════════════════════════════════════════════
def _zero_coeff_like(coeffs):
    out = [np.zeros_like(coeffs[0])]
    for c_h, c_v, c_d in coeffs[1:]:
        out.append((np.zeros_like(c_h), np.zeros_like(c_v), np.zeros_like(c_d)))
    return out


def wavelet_components_2d(field_2d, wavelet="db2", level=4, mode="periodization"):
    coeffs = pywt.wavedec2(field_2d, wavelet=wavelet, level=level, mode=mode)
    comp = {}
    c_a = _zero_coeff_like(coeffs)
    c_a[0] = coeffs[0]
    a4 = pywt.waverec2(c_a, wavelet=wavelet, mode=mode)
    comp["A4"] = np.asarray(a4[:field_2d.shape[0], :field_2d.shape[1]], dtype=np.float64)
    for i in [1, 2, 3, 4]:
        idx = len(coeffs) - i
        c_w = _zero_coeff_like(coeffs)
        c_w[idx] = coeffs[idx]
        wi = pywt.waverec2(c_w, wavelet=wavelet, mode=mode)
        comp[f"W{i}"] = np.asarray(wi[:field_2d.shape[0], :field_2d.shape[1]], dtype=np.float64)
    return comp


def plot_sfig7():
    """POD-component-only sample — target, VCNN output, direct residual, band error."""
    import torch

    npz_path = PROJECT_ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz"
    if not npz_path.exists():
        print("  ⚠️  VCNN NPZ not found, skipping S7.")
        return

    with np.load(npz_path) as z:
        out_nchw = np.asarray(z["output_nchw"], dtype=np.float64)
        tgt_nchw = np.asarray(z["target_nchw"], dtype=np.float64)

    ckpt_dir = PROJECT_ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n0020_seed000_custom"
    ckpt = torch.load(str(ckpt_dir / "vcnn_best.pt"), map_location="cpu", weights_only=False)
    mean_c = np.asarray(ckpt.get("norm_mean_c", [0.0, 0.0]), dtype=np.float64)
    std_c = np.asarray(ckpt.get("norm_std_c", [1.0, 1.0]), dtype=np.float64)
    tgt_phys = tgt_nchw * std_c[None, :, None, None] + mean_c[None, :, None, None]
    out_phys = out_nchw * std_c[None, :, None, None] + mean_c[None, :, None, None]

    tau = 0.05
    sample_idx = None
    for si in range(min(out_phys.shape[0], 100)):
        tgt_c0 = tgt_phys[si, 0]
        out_c0 = out_phys[si, 0]
        tgt_comps = wavelet_components_2d(tgt_c0)
        out_comps = wavelet_components_2d(out_c0)

        def rel_l2(a, b):
            return np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(b.ravel()) + 1e-12)

        directs = [rel_l2(out_comps[b], tgt_comps[b]) for b in BANDS_CF]
        def s_from_errs(errs):
            s = 0
            for v in errs:
                if v <= tau: s += 1
                else: break
            return s
        s = s_from_errs(directs)
        if s <= 3:
            sample_idx = si
            break

    if sample_idx is None:
        sample_idx = 0

    tgt = tgt_phys[sample_idx, 0]
    out = out_phys[sample_idx, 0]
    tgt_comps = wavelet_components_2d(tgt)
    out_comps = wavelet_components_2d(out)

    errs = []
    for b in BANDS_CF:
        errs.append(np.linalg.norm(out_comps[b].ravel() - tgt_comps[b].ravel()) /
                    (np.linalg.norm(tgt_comps[b].ravel()) + 1e-12))

    s_val = s_from_errs(errs)
    if s_val < 5:
        fail_band = BANDS_CF[s_val]
    else:
        fail_band = "W1"

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    vmin, vmax = -5, 5

    # Row 1: Target, Output, Direct residual
    axes[0, 0].imshow(tgt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Target (sample {sample_idx})", fontsize=10)
    axes[0, 1].imshow(out, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("VCNN output", fontsize=10)
    direct_res = out - tgt
    axes[0, 2].imshow(direct_res, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0, 2].set_title(f"Direct residual\n(S_full={s_val})", fontsize=10)

    # Row 2: Band-limited target, Band-limited output, Band error
    fail_comp_tgt = tgt_comps[fail_band]
    fail_comp_out = out_comps[fail_band]
    axes[1, 0].imshow(fail_comp_tgt, cmap='RdBu_r')
    axes[1, 0].set_title(f"Target: {fail_band} band only", fontsize=10)
    axes[1, 1].imshow(fail_comp_out, cmap='RdBu_r')
    axes[1, 1].set_title(f"Output: {fail_band} band only", fontsize=10)

    fail_res = fail_comp_out - fail_comp_tgt
    im = axes[1, 2].imshow(fail_res, cmap='RdBu_r')
    axes[1, 2].set_title(f"Band error ({fail_band})\nRelL2={errs[BANDS_CF.index(fail_band)]:.3f}", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    # 修复: "coherent-only" → "POD-component-only", 删除 "S-Fig 8:"
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle("POD-component-only reconstruction example (VCNN)", fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(OUT_DIR / "sfig7_coherent_only_sample.png", dpi=200)
    fig.savefig(OUT_DIR / "sfig7_coherent_only_sample_hires.png", dpi=400)
    plt.close(fig)
    print("  ✓ S7 (POD-component-only sample) saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Redrawing S5, S6, S7 (P0-27 fixes)")
    print("=" * 60)
    print("[1/3] S5: Noise propagation")
    plot_sfig5()
    print("[2/3] S6: Level sensitivity (normalized)")
    plot_sfig6()
    print("[3/3] S7: POD-component-only sample")
    plot_sfig7()
    print("\n  All figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
=============================================================================
论文配图生成脚本 (Thesis Figures — Unified Data Source)
=============================================================================

从 results/20260714/thesis_data_audit.json 统一数据源生成所有论文配图。

Figures:
  1. fig01_oracle_audit.pdf       — Oracle 审计热力图 (rank sweep)
  2. fig02_three_layer.pdf        — 三层误差分解柱状图
  3. fig03_compensation.pdf       — 补偿效应对比 (修复：柱状图替代折线图)
  4. fig04_phase_mlp.pdf          — MLP 相位图热力图
  5. fig05_ger_baseline.pdf       — GER vs M 折线图 + 幂律拟合
  6. fig06_recoverability_chain.pdf — 跨模型恢复力链
  7. fig07_cross_model_bands.pdf  — 跨模型 per-band 误差对比
  8. fig08_tau_sensitivity.pdf    — τ 敏感性
  9. fig09_wavelet_sensitivity.pdf — 小波基敏感性
 10. fig10_noise_propagation.pdf  — 噪声传播退化比
 11. sfig_vcnn_phase.pdf          — VCNN 相位图 (附录)
 12. sfig_ridge_phase.pdf         — Ridge 相位图 (附录)

Output: results/20260714/figures/*.pdf

Usage:
  conda run -n sana python3 scripts/20260714/generate_figures.py
=============================================================================
"""

from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "artifacts/derived/main/statistics"
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["A4", "W4", "W3", "W2", "W1"]
BAND_COLORS = {"A4": "#d62728", "W4": "#ff7f0e", "W3": "#2ca02c", "W2": "#1f77b4", "W1": "#9467bd"}
MODEL_COLORS = {"ridge": "#e74c3c", "mlp": "#2ecc71", "vcnn": "#3498db", "gappy_pod": "#9b59b6"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_data():
    with open(DATA_DIR / "thesis_data_audit.json") as f:
        return json.load(f)["results"]


# ═══════════════════════════════════════════════════════════════════
# Fig 1 — Oracle Audit
# ═══════════════════════════════════════════════════════════════════
def fig01_oracle_audit(res):
    oa = res["oracle_audit"]
    datasets = ["nc", "rdb_h5", "sst_weekly"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        od = oa[ds]
        audit = od["audit"]
        ranks = sorted([int(r) for r in audit.keys()])
        matrix = np.array([[audit[str(r)]["max"][b] for b in BANDS] for r in ranks])
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.05)
        ax.set_xticks(range(5)); ax.set_xticklabels(BANDS)
        ax.set_yticks(range(len(ranks))); ax.set_yticklabels(ranks)
        ax.set_title(f"{ds.upper()} (safe_rank_MEAN={od.get('safe_rank_mean','N/A')})", fontsize=9)
        for i in range(len(ranks)):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=6, color="black" if matrix[i,j] < 0.03 else "white")
    fig.colorbar(im, ax=axes, shrink=0.6, label="max band E_trunc")
    fig.suptitle("Oracle Audit: Band-wise Max Truncation Error vs POD Rank", fontsize=11, y=1.02)
    fig.savefig(OUT_DIR / "fig01_oracle_audit.pdf"); plt.close(fig); print("  ✓ fig01_oracle_audit")


# ═══════════════════════════════════════════════════════════════════
# Fig 2 — Three-Layer Error Decomposition
# ═══════════════════════════════════════════════════════════════════
def fig02_three_layer(res):
    comp = res["compensation_effect"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(BANDS))
    w = 0.25
    etot = [max(comp[b]["E_total_mean"], 1e-6) for b in BANDS]
    etru = [max(comp[b]["E_trunc_mean"], 1e-6) for b in BANDS]
    epre = [max(comp[b]["E_pred_mean"], 1e-6) for b in BANDS]
    ax.bar(x - w, etot, w, label="E_total", color="#e74c3c", alpha=0.9)
    ax.bar(x, etru, w, label="E_trunc (oracle)", color="#3498db", alpha=0.9)
    ax.bar(x + w, epre, w, label="E_pred (model)", color="#9b59b6", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("Relative L2 Error (log scale)"); ax.set_yscale("log")
    ax.set_xlabel("Wavelet Band")
    ax.set_title(f"Three-Layer Error Decomposition (VCNN, M=10, σ=0, n={comp['n_samples']})")
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="τ=0.05")
    ax.legend(); ax.grid(axis="y", alpha=0.3, which="both")
    fig.savefig(OUT_DIR / "fig02_three_layer.pdf"); plt.close(fig); print("  ✓ fig02_three_layer (log y)")


# ═══════════════════════════════════════════════════════════════════
# Fig 3 — Compensation (FIXED: bar chart + log y for errors)
# ═══════════════════════════════════════════════════════════════════
def fig03_compensation(res):
    comp = res["compensation_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    # Left: bar chart of E_total, E_trunc, E_pred — LOG Y
    ax = axes[0]
    x = np.arange(len(BANDS)); w = 0.25
    etot = [max(comp[b]["E_total_mean"], 1e-6) for b in BANDS]
    etru = [max(comp[b]["E_trunc_mean"], 1e-6) for b in BANDS]
    epre = [max(comp[b]["E_pred_mean"], 1e-6) for b in BANDS]
    ax.bar(x - w, etot, w, label="E_total", color="#e74c3c", alpha=0.9)
    ax.bar(x, etru, w, label="E_trunc", color="#3498db", alpha=0.9)
    ax.bar(x + w, epre, w, label="E_pred", color="#9b59b6", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("Relative L2 Error (log scale)"); ax.set_yscale("log")
    ax.set_title("Three-Layer Decomposition")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, which="both")

    # Right: compensation ratio per band (linear scale)
    ax2 = axes[1]
    ratios = [comp[b]["compensation_ratio"] for b in BANDS]
    bars = ax2.bar(BANDS, ratios, color=[BAND_COLORS[b] for b in BANDS], alpha=0.85)
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5, label="No compensation (ratio=1)")
    ax2.set_ylabel("E_total / E_trunc")
    ax2.set_title("Compensation Magnitude\n(higher = stronger anti-correlation)")
    ax2.legend(fontsize=7); ax2.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{r:.1f}×", ha="center", fontsize=8)

    fig.suptitle(f"Compensation Effect (VCNN, M=10, σ=0, n={comp['n_samples']})", fontsize=10)
    fig.savefig(OUT_DIR / "fig03_compensation.pdf"); plt.close(fig); print("  ✓ fig03_compensation (log y)")


# ═══════════════════════════════════════════════════════════════════
# Fig 4 — MLP Phase Diagram (multi-line chart, NOT heatmap)
# ═══════════════════════════════════════════════════════════════════
def fig04_phase_mlp(res):
    mlp = res["phase_diagram"]["mlp"]
    M_vals = [10, 15, 20, 30, 50]
    S_vals = [0.0, 0.001, 0.01, 0.1]

    sigma_configs = [
        ("0.0", "o-", "#2ecc71", "0"),
        ("0.001", "s-", "#3498db", "10⁻³"),
        ("0.01", "D-", "#e67e22", "10⁻²"),
        ("0.1", "v-", "#e74c3c", "10⁻¹"),
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s_key, marker, color, s_label in sigma_configs:
        vals = []
        for m in M_vals:
            d = mlp.get(str(m), {}).get(s_key)
            vals.append(d["mean"] if d else np.nan)
        ax.plot(M_vals, vals, marker, color=color, label=f"σ={s_label}", lw=1.5, markersize=7)

    ax.set_xlabel("M (sensor count)"); ax.set_ylabel("mean S_full")
    ax.set_title("MLP Phase Diagram: S_full vs Sensor Count")
    ax.set_xticks(M_vals); ax.set_xlim(5, 55)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fig04_phase_mlp.pdf"); plt.close(fig); print("  ✓ fig04_phase_mlp (multi-line chart)")


# ═══════════════════════════════════════════════════════════════════
# Fig 5 — GER Baseline (power-law fits)
# ═══════════════════════════════════════════════════════════════════
def fig05_ger_baseline(res):
    ger = res["ger_baseline_fits"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    M_ref = np.array([10, 15, 20, 30, 50], dtype=float)
    for mt, color in [("ridge", "#e74c3c"), ("mlp", "#2ecc71"), ("vcnn", "#3498db")]:
        gd = ger.get(mt)
        if not gd or "GER_by_M" not in gd: continue
        gers = [gd["GER_by_M"].get(str(m), np.nan) for m in [10, 15, 20, 30, 50]]
        ax.plot(M_ref, gers, "o-", color=color, label=f"{mt} (α={gd['alpha']:.2f}, R²={gd['R2']:.3f})", lw=1.5, markersize=6)
    # Add Gappy POD if available
    gappy_path = DATA_DIR / "gappy_pod_baseline.json"
    if gappy_path.exists():
        with open(gappy_path) as f: gp = json.load(f)
        gers_gp = [(r["mask_num"], r["GER_mean"]) for r in gp["results"] if r["noise_sigma"] == 0.0]
        if gers_gp:
            mg, gg = zip(*sorted(gers_gp))
            ax.plot(mg, gg, "s--", color="#9b59b6", label=f"GappyPOD (α={gp['power_law_fit_sigma0']['alpha']:.2f})", lw=1.5, markersize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M (sensor count)"); ax.set_ylabel("GER")
    ax.set_title("Global Error vs Sensor Count (σ=0)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")
    fig.savefig(OUT_DIR / "fig05_ger_baseline.pdf"); plt.close(fig); print("  ✓ fig05_ger_baseline")


# ═══════════════════════════════════════════════════════════════════
# Fig 6 — Recoverability Chain (cross-model S_full + GER comparison)
# ═══════════════════════════════════════════════════════════════════
def fig06_recoverability_chain(res):
    xmodel = res["cross_model_comparison"]
    models = ["oracle", "mlp", "vcnn", "ridge"]
    labels = ["Oracle", "MLP", "VCNN", "Ridge"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    sfull_vals = [5.0, xmodel["mlp"]["mean_S_full"], xmodel["vcnn"]["mean_S_full"], xmodel["ridge"]["mean_S_full"]]
    ger_vals = [0.0006, xmodel["mlp"]["mean_GER"], xmodel["vcnn"]["mean_GER"], xmodel["ridge"]["mean_GER"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    ax1.bar(labels, sfull_vals, color=colors, alpha=0.85)
    ax1.set_ylabel("S_full"); ax1.set_title("Scale Recoverability (M=20, σ=0)")
    ax1.axhline(5, color="gray", ls="--", alpha=0.5)
    for i, v in enumerate(sfull_vals): ax1.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)

    ax2.bar(labels, ger_vals, color=colors, alpha=0.85)
    ax2.set_ylabel("GER (log scale)"); ax2.set_yscale("log")
    ax2.set_title("Global Error (M=20, σ=0)")
    for i, v in enumerate(ger_vals): ax2.text(i, v * 1.3, f"{v:.4f}", ha="center", fontsize=8)
    ax2.grid(axis="y", alpha=0.3, which="both")
    fig.savefig(OUT_DIR / "fig06_recoverability_chain.pdf"); plt.close(fig); print("  ✓ fig06_recoverability_chain (log GER)")


# ═══════════════════════════════════════════════════════════════════
# Fig 7 — Cross-Model Per-Band Error Comparison (log y)
# ═══════════════════════════════════════════════════════════════════
def fig07_cross_model_bands(res):
    xmodel = res["cross_model_comparison"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(BANDS)); w = 0.2
    for i, (mt, color) in enumerate([("ridge", "#e74c3c"), ("mlp", "#2ecc71"), ("vcnn", "#3498db")]):
        errs = [max(xmodel[mt]["per_band_E_total"][b], 1e-6) for b in BANDS]
        ax.bar(x + (i-1)*w, errs, w, label=mt.upper(), color=color, alpha=0.85)
    oracle_errs = [max(v, 1e-6) for v in [0.00009, 0.00096, 0.00224, 0.00525, 0.00703]]
    ax.plot(x, oracle_errs, "ko-", label="Oracle", markersize=4, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("E_total (per band, log scale)"); ax.set_yscale("log")
    ax.set_title("Cross-Model Per-Band Error (M=20, σ=0)")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5, label="τ=0.05")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, which="both")
    fig.savefig(OUT_DIR / "fig07_cross_model_bands.pdf"); plt.close(fig); print("  ✓ fig07_cross_model_bands (log y)")


# ═══════════════════════════════════════════════════════════════════
# Fig 8 — Tau Sensitivity
# ═══════════════════════════════════════════════════════════════════
def fig08_tau_sensitivity(res):
    ts = res["tau_sensitivity"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    M_vals = [10, 20, 30, 50]
    for tau_v, color in [("0.03", "#e74c3c"), ("0.05", "#2ecc71"), ("0.08", "#3498db")]:
        vals = [ts[tau_v][str(m)]["mean_S_full"] for m in M_vals if str(m) in ts[tau_v]]
        ax.plot(M_vals[:len(vals)], vals, "o-", color=color, label=f"τ={tau_v}", lw=1.5, markersize=6)
    ax.set_xlabel("M (sensor count)"); ax.set_ylabel("mean S_full")
    ax.set_title("Tau Sensitivity (MLP, σ=0)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fig08_tau_sensitivity.pdf"); plt.close(fig); print("  ✓ fig08_tau_sensitivity")


# ═══════════════════════════════════════════════════════════════════
# Fig 9 — Wavelet Basis Sensitivity
# ═══════════════════════════════════════════════════════════════════
def fig09_wavelet_sensitivity(res):
    ws = res["wavelet_sensitivity"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(BANDS)); w = 0.25
    for i, (wname, color) in enumerate([("db2", "#2ecc71"), ("sym2", "#3498db"), ("haar", "#e74c3c")]):
        errs = [ws[wname]["per_band_E_total_mean"][b] for b in BANDS]
        ax.bar(x + (i-1)*w, errs, w, label=wname, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("E_total"); ax.set_title("Wavelet Basis Sensitivity (MLP, M=20, σ=0)")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "fig09_wavelet_sensitivity.pdf"); plt.close(fig); print("  ✓ fig09_wavelet_sensitivity")


# ═══════════════════════════════════════════════════════════════════
# Fig 10 — Noise Propagation
# ═══════════════════════════════════════════════════════════════════
def fig10_noise_propagation(res):
    np_data = res["noise_propagation"]["mlp_M20"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ratios = [np_data[b] for b in BANDS]
    ax.bar(BANDS, ratios, color=[BAND_COLORS[b] for b in BANDS], alpha=0.85)
    ax.set_ylabel("Degradation Ratio (σ=0.1 / σ=0)"); ax.set_xlabel("Band")
    ax.set_title("Noise Propagation (MLP, M=20)")
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(ratios): ax.text(i, r + 1, f"{r:.1f}×", ha="center", fontsize=9)
    fig.savefig(OUT_DIR / "fig10_noise_propagation.pdf"); plt.close(fig); print("  ✓ fig10_noise_propagation")


# ═══════════════════════════════════════════════════════════════════
# Supplementary — VCNN Phase Diagram (multi-line chart)
# ═══════════════════════════════════════════════════════════════════
def sfig_vcnn_phase(res):
    vcnn = res["phase_diagram"]["vcnn"]
    M_vals = [10, 15, 20, 30, 50]
    sigma_configs = [
        ("0.0", "o-", "#2ecc71", "0"),
        ("0.001", "s-", "#3498db", "10⁻³"),
        ("0.01", "D-", "#e67e22", "10⁻²"),
        ("0.1", "v-", "#e74c3c", "10⁻¹"),
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s_key, marker, color, s_label in sigma_configs:
        vals = []
        for m in M_vals:
            d = vcnn.get(str(m), {}).get(s_key)
            vals.append(d["mean"] if d else np.nan)
        ax.plot(M_vals, vals, marker, color=color, label=f"σ={s_label}", lw=1.5, markersize=7)
    ax.set_xlabel("M (sensor count)"); ax.set_ylabel("mean S_full")
    ax.set_title("VCNN Phase Diagram: S_full vs Sensor Count")
    ax.set_xticks(M_vals); ax.set_xlim(5, 55)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "sfig_vcnn_phase.pdf"); plt.close(fig); print("  ✓ sfig_vcnn_phase (multi-line)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating thesis figures from unified data source")
    print(f"  Data: {DATA_DIR / 'thesis_data_audit.json'}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    res = load_data()

    # Plot all figures
    fig01_oracle_audit(res)
    fig02_three_layer(res)
    fig03_compensation(res)
    fig04_phase_mlp(res)
    fig05_ger_baseline(res)
    fig06_recoverability_chain(res)
    fig07_cross_model_bands(res)
    fig08_tau_sensitivity(res)
    fig09_wavelet_sensitivity(res)
    fig10_noise_propagation(res)
    sfig_vcnn_phase(res)

    # 发布由 publish_figures 统一处理 (仅 fig04_phase_mlp → thesis fig05_phase_mlp.pdf)
    print("\nDone. 输出位于 build/figures_raw (fig05_phase_mlp.pdf 由 publish_figures 发布)")


if __name__ == "__main__":
    main()

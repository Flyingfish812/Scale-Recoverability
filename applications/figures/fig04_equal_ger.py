#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_equal_ger.py — Fig. 4 Equal-GER pair analysis (v3-3, 2026-09-02)

Two panels:
  (a) cumulative distribution (ECDF) of the relative GER difference, grouped by
      ΔS_full, with the representative pair marked
  (b) statistics: median comparison for W1 / vorticity / gradient + bootstrap CI
Data source: artifacts/statistics/equal_ger_pairs.json (42,000-record convention, verified)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
STATS = _ROOT / "artifacts" / "statistics" / "equal_ger_pairs.json"


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    d = json.loads(STATS.read_text(encoding="utf-8"))
    pairs = d["all_pairs"]
    rep = d["representative_pair"]
    stats = d["paired_statistics"]
    ci = d["cluster_bootstrap_ci"]
    n = len(pairs)

    ger_diffs = np.array([p["GER_diff"] for p in pairs])
    sfull_diffs = np.array([p["S_full_diff"] for p in pairs])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2),
                             gridspec_kw=dict(wspace=0.5))

    # ── (a) ECDF of relative ΔGER, grouped by ΔS_full gap ──────
    # Pairs are grouped by ΔS_full, which keeps the curve readable: for
    # ΔS_full=2 and 3 alike, nearly all pairs differ by 0.1%–1%, all of them
    # below 1%; a few reach 1e-4, and the representative pair is ≈ 6e-7 (star).
    ax = axes[0]
    cats = [2, 3]
    for v in cats:
        yv = np.sort(ger_diffs[sfull_diffs == v])
        nv = len(yv)
        ax.step(yv, np.arange(1, nv + 1) / nv, where="post", lw=1.3,
                color=(ps.MODEL_COLORS["MLP"] if v == 2
                       else ps.MODEL_COLORS["Ridge"]),
                label=f"$\\Delta S_{{\\mathrm{{full}}}}$={v}  (n={nv})")
    # representative pair (smallest relative GER difference): star + label
    best = int(np.argmin(ger_diffs))
    m_best = sfull_diffs == sfull_diffs[best]
    best_frac = float(np.sum(ger_diffs[m_best] <= ger_diffs[best])) / int(m_best.sum())
    ax.scatter([ger_diffs[best]], [best_frac], marker="*", s=130,
               color=ps.ACCENT, edgecolors="white", linewidths=0.8, zorder=6)
    ax.annotate("best pair", xy=(ger_diffs[best], best_frac),
                xytext=(1.6e-6, 0.12), fontsize=7.5, color=ps.ACCENT,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ps.ACCENT, lw=0.9))
    ax.set_xscale("log")
    ax.set_xlim(3e-7, 2e-2)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("relative $\\Delta$GER")
    ax.set_ylabel("cumulative fraction of pairs")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ps.panel_label(ax, "a")

    # ── (b) statistics ───────────────────────────────────────────
    ax = axes[1]
    cb_results = {r["label"]: r for r in ci["results"]}
    ps_tests = {t["label"]: t for t in stats["tests"]}
    metrics_info = [
        ("W1 band error", "W1_band_error", "W1_band_error_diff"),
        ("Laplacian RMSE", "vorticity_RMSE", "vorticity_RMSE_diff"),
        ("Gradient RMSE", "gradient_RMSE", "gradient_RMSE_diff"),
    ]
    x_pos = np.arange(len(metrics_info))
    width = 0.3
    for i, (label, ps_key, cb_key) in enumerate(metrics_info):
        ps_r = ps_tests.get(ps_key, {})
        cb_r = cb_results.get(cb_key, {})
        low = ps_r.get("median_low_S_full", 0)
        high = ps_r.get("median_high_S_full", 0)
        p_val = cb_r.get("p_value_bootstrap", 1.0)
        ax.bar(i - width / 2, low, width, color=ps.MODEL_COLORS["Ridge"],
               alpha=0.85, label="low $S_{\\mathrm{full}}$" if i == 0 else "")
        ax.bar(i + width / 2, high, width, color=ps.MODEL_COLORS["Gappy"],
               alpha=0.85, label="high $S_{\\mathrm{full}}$" if i == 0 else "")
        ref = max(low, high)
        anno_y = 10 ** (np.log10(ref) + 0.12) if ref > 0 else 0.01
        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        # label the p value only; the CI stays in the caption
        ax.text(i, anno_y, p_str, ha="center", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["W1", "Laplacian", "Gradient"], fontsize=8)
    ax.set_ylabel("Median value (log)")
    ax.set_yscale("log")
    ax.set_ylim(top=1e-1)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ps.panel_label(ax, "b")

    ps.save(fig, OUT_DIR, "fig04_equal_ger")
    print(f"  [equal-GER] n={n} pairs; rep: GER {rep['GER_low']:.4f} vs "
          f"{rep['GER_high']:.4f}, S_full {rep['S_full_low']} vs {rep['S_full_high']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

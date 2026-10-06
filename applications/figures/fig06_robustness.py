#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig06_robustness.py — main figure 6: robustness of the scale count.

Two compact panels that support the robustness discussion with main-text figures rather than with the supplement alone:

  (a) mean S_full of Ridge, MLP and VCNN in the five tested wavelet bases
      (NC, M = 30, sigma = 0, tau = 0.05). The modal counts of the same configurations are printed above the Ridge bars because the text cites them; the complete table stays in the supplement.
  (b) mean S_full against sensor count for tau = 0.03, 0.05 and 0.08 (MLP),
      solid lines for clean measurements and dashed lines for sigma = 10^-2.

Data: artifacts/statistics/wavelet_sensitivity.json (a),
      artifacts/statistics/threshold_sensitivity.json (b).
Both artifacts are also the sources of the published sensitivity tables.

Output: artifacts/figures/fig06_robustness.pdf
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
STATS = _ROOT / "artifacts" / "statistics"

WAVELETS = ["haar", "db2", "db4", "sym4", "coif1"]
MODELS = [("ridge", "Ridge"), ("mlp", "MLP"), ("vcnn", "VCNN")]
TAUS = [0.03, 0.05, 0.08]
TAU_COLORS = {0.03: ps.MODEL_COLORS["Ridge"], 0.05: ps.MODEL_COLORS["MLP"],
              0.08: ps.MODEL_COLORS["Gappy"]}


def panel_wavelets(ax, data: dict) -> None:
    """(a) mean (and modal) scale count per wavelet and model."""
    x = np.arange(len(WAVELETS))
    width = 0.26
    for i, (key, label) in enumerate(MODELS):
        means = [data["real_nc"][w][key]["S_full_mean"] for w in WAVELETS]
        off = (i - 1) * width
        ax.bar(x + off, means, width, color=ps.MODEL_COLORS[label], alpha=0.9,
               label=label)
        if key == "ridge":
            for xi, w in zip(x, WAVELETS):
                mode = int(round(data["real_nc"][w][key]["S_full_mode"]))
                ax.text(xi + off, means[WAVELETS.index(w)] + 0.08, f"{mode}",
                        ha="center", va="bottom", fontsize=6.0, color="#333333")
    ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xticks(x)
    ax.set_xticklabels(WAVELETS, fontsize=6.5)
    ax.set_xlabel("wavelet", fontsize=7)
    ax.set_ylabel("mean $S_{\\mathrm{full}}$", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.0, ncol=3, loc="upper center", frameon=False,
              handlelength=1.0, columnspacing=0.7, borderaxespad=0.1,
              bbox_to_anchor=(0.5, 1.03))
    ps.panel_label(ax, "a", x=-0.22, y=1.02)


def panel_tau(ax, results: list) -> None:
    """(b) sensor-count trend for three recovery thresholds."""
    masks = sorted({r["mask_num"] for r in results})
    for tau in TAUS:
        for sigma, ls in [(0.0, "-"), (0.01, "--")]:
            row = []
            for m in masks:
                hit = [r for r in results if r["model"] == "mlp"
                       and r["mask_num"] == m and r["sigma"] == sigma
                       and r["tau"] == tau]
                row.append(hit[0]["mean_S_full"] if hit else np.nan)
            ax.plot(masks, row, color=TAU_COLORS[tau], ls=ls, marker="o",
                    ms=3.2, lw=1.2,
                    label=f"$\\tau$ = {tau:g}" if sigma == 0.0 else None)
    ax.set_xticks(masks)
    ax.set_ylim(-0.2, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("sensor count $M$", fontsize=7)
    ax.set_ylabel("mean $S_{\\mathrm{full}}$ (MLP)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.0, ncol=3, loc="lower right", frameon=False,
              handlelength=1.2, columnspacing=0.7, borderaxespad=0.2)
    ax.set_title("solid: $\\sigma$ = 0; dashed: $\\sigma$ = $10^{-2}$",
                 fontsize=6.0, pad=3)
    ps.panel_label(ax, "b", x=-0.22, y=1.02)


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    wavelet = json.loads((STATS / "wavelet_sensitivity.json").read_text(encoding="utf-8"))
    threshold = json.loads(
        (STATS / "threshold_sensitivity.json").read_text(encoding="utf-8"))["results"]

    fig, axes = plt.subplots(1, 2, figsize=(5.33, 2.15),
                             gridspec_kw=dict(wspace=0.30, left=0.085,
                                              right=0.985, top=0.88,
                                              bottom=0.185))

    panel_wavelets(axes[0], wavelet)
    panel_tau(axes[1], threshold)

    ps.save(fig, OUT_DIR, "fig06_robustness")

    ridge = {w: round(wavelet["real_nc"][w]["ridge"]["S_full_mean"], 2)
             for w in WAVELETS}
    print(f"  [wavelets] Ridge mean S_full: {ridge}")
    print(f"  [wavelets] Ridge modal S_full: "
          f"{ {w: int(round(wavelet['real_nc'][w]['ridge']['S_full_mode'])) for w in WAVELETS} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())

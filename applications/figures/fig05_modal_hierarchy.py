#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig05_modal_hierarchy.py — main figure 4: modal interpretation of the hierarchy.

  (a) cumulative band coverage C_b(r) of the retained rank-128 POD basis, with
      the mode count needed for 90 % coverage marked for A4 and W1
  (b-d) per-modal NRMSE against lambda_j/lambda_1 at (M, sigma) = (20, 0) for
      MLP, Ridge and VCNN, with the Spearman correlation and its 95 % bootstrap interval

Data: artifacts/statistics/mode_scale_energy.json (a),
      artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz and
      artifacts/statistics/modal_coefficient_error.json (b-d).
Output: artifacts/figures/fig05_modal_hierarchy.pdf
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

MODELS = [("MLP", "mlp", "MLP"), ("Ridge", "ridge", "Ridge"),
          ("VCNN", "vcnn", "VCNN")]


def load_coverage() -> tuple[dict, dict]:
    d = json.loads((STATS / "mode_scale_energy.json").read_text(encoding="utf-8"))
    return d["cumulative_coverage"], d["threshold_modes"]


def load_mode_energy() -> np.ndarray:
    pod = np.load(str(_ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    coeffs = np.asarray(pod["coefficients"], dtype=np.float64)
    energy = np.var(coeffs, axis=0)
    return energy / energy[0]


def load_nrmse(key: str) -> tuple[np.ndarray, tuple[float, float]]:
    records = json.loads(
        (STATS / "modal_coefficient_error.json").read_text(encoding="utf-8"))["records"]
    record = next(r for r in records if r["model"] == key
                  and r["sensor_count"] == 20 and r["noise_sigma"] == 0.0)
    return (np.asarray(record["nrmse_per_mode"], dtype=np.float64),
            record["spearman_ci_95"])


def panel_coverage(ax, coverage: dict, thresholds: dict) -> None:
    """(a) cumulative band coverage, wide top panel."""
    for band in ps.BANDS:
        arr = np.asarray(coverage[band], dtype=np.float64)
        ax.plot(np.arange(1, len(arr) + 1), arr, color=ps.BAND_COLORS[band],
                lw=1.2, label=band)
    ax.axhline(0.9, color="0.45", ls=":", lw=0.8)
    # 90 % crossing of the coarsest and the finest band, labelled with the mode count reported in the text (12 for A4, 92 for W1). The labels are placed with positive data coordinates: the x axis is logarithmic.
    for band, factor, ha in [("A4", 1.28, "left"), ("W1", 0.78, "right")]:
        r90 = float(thresholds[band]["0.9"])
        ax.plot([r90], [0.9], marker="o", ms=4.0, color=ps.BAND_COLORS[band],
                mec="black", mew=0.5, zorder=6)
        ax.text(r90 * factor, 0.925, f"{int(round(r90))}",
                fontsize=6.5, color=ps.BAND_COLORS[band], ha=ha, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("$C_b(r)$")
    ax.legend(fontsize=6.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False, handlelength=1.1,
              columnspacing=0.7, borderaxespad=0.0)
    ps.panel_label(ax, "a", x=-0.075, y=1.02)


def panel_modal(ax, energy: np.ndarray, nrmse: np.ndarray,
                ci: tuple[float, float], name: str, letter: str,
                show_ylabel: bool) -> None:
    """(b-d) per-modal NRMSE versus modal energy."""
    from scipy.stats import spearmanr

    ax.scatter(energy, nrmse, c=ps.MODEL_COLORS[name], alpha=0.45, s=5,
               edgecolors="none", rasterized=True)
    rho, _ = spearmanr(energy, nrmse)
    ha, va = (0.05, 0.97) if name != "Ridge" else (0.05, 0.97)
    ax.text(ha, va, f"$\\rho$ = {rho:.2f}\n[{float(ci[0]):.2f}, {float(ci[1]):.2f}]",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.65",
                      boxstyle="round,pad=0.22", lw=0.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(nrmse)) * 0.4, float(np.max(nrmse)) * 5.0)
    ax.set_xlabel("$\\lambda_j/\\lambda_1$", fontsize=7.5)
    ax.set_ylabel("NRMSE $e_j$" if show_ylabel else "", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title(name, fontsize=7.5, pad=3)
    ps.panel_label(ax, letter, x=-0.26, y=1.02)


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    coverage, thresholds = load_coverage()
    energy = load_mode_energy()

    fig = plt.figure(figsize=(5.33, 2.60))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.85,
                          wspace=0.55, left=0.075, right=0.985, top=0.94,
                          bottom=0.155)

    ax = fig.add_subplot(gs[0, :])
    panel_coverage(ax, coverage, thresholds)

    for i, (name, key, color) in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, i])
        nrmse, ci = load_nrmse(key)
        panel_modal(ax, energy, nrmse, (float(ci[0]), float(ci[1])), color,
                    chr(ord("b") + i), show_ylabel=(i == 0))

    ps.save(fig, OUT_DIR, "fig05_modal_hierarchy")

    print("  [90% modes] " + ", ".join(
        f"{b}:{int(round(float(thresholds[b]['0.9'])))}" for b in ps.BANDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

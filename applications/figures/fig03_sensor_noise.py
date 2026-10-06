#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_sensor_noise.py — main figure 3: sensor count and measurement noise.

  (a) mean GER against M under clean measurements (NC MLP)
  (b) mean S_full against M for sigma = 0, 10^-3, 10^-2, 10^-1 (NC MLP)

Both panels use one representative MLP run for the scale index and the same records as the model comparison table. No reference line or fit is drawn; the supplementary material carries the phase diagrams and the numeric tables.

Data: artifacts/statistics/band_error_decomposition.json (a),
      artifacts/statistics/sensor_noise_phase.json (b).
Output: artifacts/figures/fig03_sensor_noise.pdf
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

M_LIST = [10, 15, 20, 30, 50]
SIGMA_LIST = [0.0, 0.001, 0.01, 0.1]
SIGMA_LABELS = ["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"]


def load_ger() -> list:
    """Mean clean-measurement GER of the MLP at each sensor count."""
    records = json.loads(
        (STATS / "band_error_decomposition.json").read_text(encoding="utf-8"))["records"]
    out = []
    for m in M_LIST:
        gers = [r["global_error"] for r in records
                if r["model"] == "mlp" and r["sensor_count"] == m
                and r["noise_sigma"] == 0.0]
        out.append(float(np.mean(gers)) if gers else 0.0)
    return out


def load_sfull() -> dict:
    """Mean S_full of the representative MLP run on the sensor-noise grid."""
    stats = json.loads(
        (STATS / "sensor_noise_phase.json").read_text(encoding="utf-8"))
    phase = stats["phase_summary"]["mlp"]
    out = {}
    for m in M_LIST:
        for s in SIGMA_LIST:
            entry = phase.get(str(m), {}).get(str(s), {})
            out[(m, s)] = float(entry.get("mean_S_full", 0.0)) if entry else 0.0
    return out


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    ger = load_ger()
    sfull = load_sfull()

    fig, axes = plt.subplots(1, 2, figsize=(5.33, 2.10),
                             gridspec_kw=dict(wspace=0.33, left=0.085,
                                              right=0.985, top=0.86,
                                              bottom=0.20))

    # (a) clean global error versus M
    ax = axes[0]
    ax.plot(M_LIST, ger, color=ps.MODEL_COLORS["MLP"],
            marker=ps.MODEL_MARKERS["MLP"], lw=1.3, ms=4.5)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 3e-2)
    ax.set_xticks(M_LIST)
    ax.set_xlabel("sensor count $M$")
    ax.set_ylabel("mean GER")
    ps.panel_label(ax, "a", x=-0.24, y=1.02)

    # (b) mean scale count on the sensor-noise grid
    ax = axes[1]
    for si, s in enumerate(SIGMA_LIST):
        ax.plot(M_LIST, [sfull[(m, s)] for m in M_LIST],
                color=ps.SIGMA_COLORS[s], marker=ps.SIGMA_MARKERS[s], lw=1.3,
                ms=4.5, label=f"$\\sigma$ = {SIGMA_LABELS[si]}")
    ax.set_xticks(M_LIST)
    ax.set_ylim(-0.25, 5.4)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("sensor count $M$")
    ax.set_ylabel("mean $S_{\\mathrm{full}}$")
    ax.legend(fontsize=6.8, ncol=2, loc="lower right", frameon=False,
              handlelength=1.3, columnspacing=0.8, borderaxespad=0.25)
    ps.panel_label(ax, "b", x=-0.24, y=1.02)

    ps.save(fig, OUT_DIR, "fig03_sensor_noise")

    print(f"  [GER(σ=0)] {[round(g, 4) for g in ger]}")
    print("  [S_full] M=20: " + ", ".join(
        f"σ={SIGMA_LABELS[i]}:{sfull[(20, s)]:.2f}"
        for i, s in enumerate(SIGMA_LIST)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS06_tau_sensitivity.py — Fig. S6 threshold (tau) sensitivity

Purpose
    Mean S_full against the sensor count M for three threshold values, at a
    clean and at a transition noise level, to show that the ranking of the
    configurations does not depend on the choice of tau.
Data source
    artifacts/statistics/threshold_sensitivity.json
Output
    artifacts/figures/figS06_tau_sensitivity.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
SUPP = _ROOT / "artifacts" / "statistics"


def main() -> int:
    ps.apply()

    d = json.loads((SUPP / "threshold_sensitivity.json").read_text())
    results = d["results"]
    fig, axes = ps.figure(6.6, 3.0, 1, 2)
    for axi, (ax, sigma_val) in enumerate([
        (axes[0], 0.0),
        (axes[1], 0.01),
    ]):
        for ti, tau in enumerate([0.03, 0.05, 0.08]):
            means = []
            for m in [10, 15, 20, 30, 50]:
                row = next((r for r in results if r["model"] == "mlp"
                            and r["mask_num"] == m
                            and abs(r["sigma"] - sigma_val) < 1e-10
                            and abs(r["tau"] - tau) < 1e-10), None)
                means.append(row["mean_S_full"] if row else 0)
            ax.plot([10, 15, 20, 30, 50], means, "o-",
                    color=[ps.SIGMA_COLORS[0.0], ps.SIGMA_COLORS[0.01],
                           ps.SIGMA_COLORS[0.1]][ti],
                    lw=1.3, ms=4.5, label=f"$\\tau$={tau:.2f}")
        ax.set_xlabel("Sensor count $M$")
        ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
        ax.set_xticks([10, 15, 20, 30, 50])
        ax.set_ylim(-0.2, 5.2)
        ax.legend(fontsize=6.5)
        ps.panel_label(ax, chr(ord("a") + axi), x=-0.2, y=1.05)

    fig.tight_layout()
    ps.save(fig, OUT_DIR, "figS06_tau_sensitivity")
    print(f"  [S6] tau sensitivity saved")

    return 0


if __name__ == "__main__":
    sys.exit(main())

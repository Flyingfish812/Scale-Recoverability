#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS07_sensor_family_ger.py — supplementary figure S13: sensor-family GER

Purpose
    Per-sequence mean GER of the MLP, ridge and greedy-POD estimators against the sensor count M at two noise levels, with the spread over the sensor families shown as error bars.
Data source
    artifacts/statistics/sensor_family/sensor_count_effect.csv
Output
    artifacts/figures/figS07_sensor_family_ger.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
SUPP = _ROOT / "artifacts" / "statistics"


def main() -> int:
    ps.apply()

    rows = list(csv.DictReader(open(SUPP / "sensor_family" / "sensor_count_effect.csv",
                                    encoding="utf-8")))
    data = {}
    for r in rows:
        data.setdefault(r["model"], {}).setdefault(float(r["sigma"]), {})[
            int(r["sensor_count"])] = {
            "median": float(r["ger_mean_median"]),
            "min": float(r["ger_mean_min"]),
            "max": float(r["ger_mean_max"]),
        }
    fig, axes = ps.figure(6.6, 2.1, 1, 3)
    Ms = [10, 15, 20, 30, 50]
    for ax, model in zip(axes, ["mlp", "ridge", "gappy"]):
        for sigma, ls, color in [(0.0, "-", ps.MODEL_COLORS["MLP"]),
                                 (0.01, "--", ps.MODEL_COLORS["Ridge"])]:
            if sigma not in data.get(model, {}):
                continue
            vals = [data[model][sigma][M] for M in Ms]
            med = np.array([v["median"] for v in vals])
            lo = np.array([v["min"] for v in vals])
            hi = np.array([v["max"] for v in vals])
            ax.errorbar(Ms, med, yerr=[med - lo, hi - med], ls=ls, color=color,
                        fmt="o", ms=3.5, lw=1.2, capsize=2.5,
                        label=f"$\\sigma$={sigma:g}")
        ax.set_xlabel("$M$")
        ax.set_xticks(Ms)
        ax.set_title({"mlp": "MLP", "ridge": "Ridge", "gappy": "Gappy POD"}[model],
                     fontsize=ps.TITLE_FONT)
        if model == "mlp":
            ax.set_ylabel("Per-sequence mean GER")
        ax.legend(fontsize=6.5)
    axes[0].set_yscale("log")
    ps.save(fig, OUT_DIR, "figS07_sensor_family_ger")
    print(f"  [S7] saved")

    return 0


if __name__ == "__main__":
    sys.exit(main())

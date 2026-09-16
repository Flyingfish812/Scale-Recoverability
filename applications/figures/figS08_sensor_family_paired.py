#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS08_sensor_family_paired.py — Fig. S8 paired MLP-VCNN comparison

Purpose
    Paired difference (MLP minus VCNN) of every reported metric with its
    confidence interval, so that the sign of the difference can be read off
    one axis.
Data source
    artifacts/statistics/paired_model_comparison.json
Output
    artifacts/figures/figS08_sensor_family_paired.pdf
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
SUPP = _ROOT / "artifacts" / "statistics"


def main() -> int:
    ps.apply()

    d = json.loads((SUPP / "paired_model_comparison.json").read_text())
    metrics = ["GER", "S_full", "S_coh", "W1", "vorticity_RMSE", "gradient_RMSE"]
    labels = {"GER": "GER", "S_full": "$S_{\\mathrm{full}}$",
              "S_coh": "$S_{\\mathrm{coh}}$", "W1": "W1 error",
              "vorticity_RMSE": "Laplacian RMSE",
              "gradient_RMSE": "Gradient RMSE"}
    fig, ax = ps.figure(4.2, 3.2, 1, 1)
    ypos = np.arange(len(metrics))
    for i, m in enumerate(metrics):
        a = d["metrics"][m]
        mean, lo, hi = a["mean_diff"], a["ci_low"], a["ci_high"]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=ps.MODEL_COLORS["MLP"], ms=4, capsize=3, lw=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[m] for m in metrics], fontsize=7.5)
    ax.axvline(0, color="0.2", lw=0.7, ls="--")
    ax.set_xlabel("Paired difference (MLP $-$ VCNN)")
    ps.save(fig, OUT_DIR, "figS08_sensor_family_paired")
    print(f"  [S8] saved")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS02_phase.py — supplementary figures S7 and S8: phase diagrams (Ridge / VCNN)

Purpose
    Mean S_full against the sensor count M, one curve per sensor-noise level, for the closed-form ridge estimator and for VCNN, so that the two phase diagrams can be compared side by side.
Data source
    artifacts/statistics/band_error_records.json (ridge subset, via records)
    artifacts/statistics/sensor_noise_phase.json (VCNN phase summary)
Output
    artifacts/figures/figS02_ridge_phase.pdf
    artifacts/figures/figS02_vcnn_phase.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import records
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
SUPP = _ROOT / "artifacts" / "statistics"


def main() -> int:
    ps.apply()

    M = [10, 15, 20, 30, 50]
    sigmas = [0.0, 0.001, 0.01, 0.1]
    # Ridge (closed form), read from the band-error records
    ridge_series = {m: {s: records.config_summary("ridge", m, s)["S_full_mean"]
                        for s in sigmas}
                    for m in M}

    # VCNN (sensor-noise phase, physical units)
    phase = json.loads((SUPP / "sensor_noise_phase.json").read_text())
    vcnn = phase["phase_summary"]["vcnn"]
    vcnn_series = {}
    for m in M:
        vcnn_series[m] = {
            s: float(vcnn.get(str(m), {}).get(str(s), {}).get("mean_S_full", 0.0))
            for s in sigmas
        }

    for series, stem in [
        (ridge_series, "figS02_ridge_phase"),
        (vcnn_series, "figS02_vcnn_phase"),
    ]:
        fig, ax = ps.figure(6.6, 3.0, 1, 1)
        sig_lab = ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
        for si, s in enumerate(sigmas):
            vals = [series.get(m, {}).get(s) for m in M]
            if any(v is None for v in vals):
                continue
            ax.plot(M, vals, color=ps.SIGMA_COLORS[s], marker=ps.SIGMA_MARKERS[s],
                    lw=1.4, ms=5, label=f"$\\sigma$={sig_lab[si]}")
        ax.set_xlabel("Sensor count $M$")
        ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
        ax.set_xticks(M)
        ax.set_ylim(-0.2, 5.2)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ps.save(fig, OUT_DIR, stem)
        print(f"  [S2] {stem} done")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS05_mode_scale_energy.py — supplementary figure S9: mode-to-scale energy

Purpose
    (a) cumulative band energy C_b(r) of the POD modes;
    (b) share of the mode energy carried by each band, Gaussian-smoothed;
    (c) the log POD spectrum lambda_r / lambda_1.
Data source
    artifacts/statistics/mode_scale_energy.json
Output
    artifacts/figures/figS05_mode_scale_energy.pdf
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

BANDS = ps.BANDS


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    d = json.loads((SUPP / "mode_scale_energy.json").read_text())
    cc = d["cumulative_coverage"]              # {band: [C_b(r) for r in 1..128]}
    bw = d["band_energy_per_mode"]             # {band: [per-mode band energy]}
    mer = d["mode_energy_ratio"]               # [lambda_j/lambda_1]
    tm = d["threshold_modes"]                  # {band: {"0.9": n_modes}}

    fig = plt.figure(figsize=(6.6, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.58,
                          wspace=0.44, left=0.07, right=0.985,
                          top=0.94, bottom=0.08)

    # (a) cumulative coverage (full width, top)
    ax = fig.add_subplot(gs[0, :])
    for b in BANDS:
        arr = np.asarray(cc[b], dtype=float)
        ax.plot(np.arange(1, len(arr) + 1), arr, color=ps.BAND_COLORS[b],
                lw=1.3, label=b)
    ax.axhline(0.9, color="0.45", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("Cumulative band energy $C_b(r)$")
    ax.legend(fontsize=7, ncol=3, loc="lower right", frameon=True)
    ps.panel_label(ax, "a", x=-0.12, y=1.04)

    # (b) per-mode band energy fractions (bottom left) — multi-line (smoothed)
    # band_energy_weighted[b][j] = λ_j ‖W_b(φ_j)‖² is a weighted energy with
    # arbitrary units, so it is not plotted directly. The plotted quantity is the share of the mode φ_r energy carried by band b:
    # frac_b(r) = E_b(φ_r) / Σ_b' E_b'(φ_r) ∈ [0,1], summing to 1 per mode.
    # Those per-mode fractions oscillate strongly in r, so the plotted curves are Gaussian-smoothed trends, which
    # make the band shares readable.
    from scipy.ndimage import gaussian_filter1d
    ax = fig.add_subplot(gs[1, 0])
    Eb = np.stack([np.asarray(bw[b], dtype=float) for b in BANDS])   # (5, 128)
    frac = Eb / Eb.sum(axis=0, keepdims=True)                        # (5, 128)
    rr = np.arange(1, frac.shape[1] + 1)
    for i, b in enumerate(BANDS):
        ax.plot(rr, gaussian_filter1d(frac[i], sigma=4.0, mode="nearest"),
                color=ps.BAND_COLORS[b], lw=1.6, label=b)
    ax.set_xscale("log")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("Fraction of mode energy in band\n"
                  r"$E_b(\phi_r^{(u)})/E(\phi_r^{(u)})$  (0$-$1)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ps.panel_label(ax, "b", x=-0.18, y=1.04)

    # (c) log POD spectrum (bottom right)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(1, len(mer) + 1), mer, color="0.25", lw=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 128)
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("$\\lambda_r/\\lambda_1$ (log)")
    ps.panel_label(ax, "c", x=-0.18, y=1.04)

    ps.save(fig, OUT_DIR, "figS05_mode_scale_energy")
    print(f"  [S5] threshold_modes 90%: "
          f"{[ (b, tm[b]['0.9']) for b in BANDS]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

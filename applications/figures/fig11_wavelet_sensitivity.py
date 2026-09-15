#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig11_wavelet_sensitivity.py — Fig. 11 wavelet-family sensitivity (v3-3, 2026-08-31)

Three panels (same convention as the analytical benchmark, data verified):
  (a) real NC data (M=30, σ=0): mean S_full per model x wavelet
  (b) Ridge per-band E_direct across wavelets (hierarchy, log)
  (c) analytical benchmark: correct fraction of S_full per case
Data source: artifacts/statistics/wavelet_sensitivity.json
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
STATS = _ROOT / "artifacts" / "statistics" / "wavelet_sensitivity.json"

WAVELETS = ["haar", "db2", "db4", "sym4", "coif1"]
BANDS = ps.BANDS
TAU = ps.TAU


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    d = json.loads(STATS.read_text(encoding="utf-8"))
    nc = d["real_nc"]
    ana = d["analytical_benchmark"]
    models = ["ridge", "mlp", "vcnn"]
    labels = {"ridge": "Ridge", "mlp": "MLP", "vcnn": "VCNN"}

    # layout: (a) full width on top, (b)(c) half width below
    fig = plt.figure(figsize=(6.6, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42,
                          wspace=0.30, left=0.06, right=0.985,
                          top=0.95, bottom=0.09)

    # ── (a) S_full mean by model x wavelet (full width) ─────────
    ax = fig.add_subplot(gs[0, :])
    xpos = np.arange(len(WAVELETS))
    width = 0.26
    for mi, m in enumerate(models):
        means = [nc[w][m]["S_full_mean"] for w in WAVELETS]
        stds = [nc[w][m]["S_full_std"] for w in WAVELETS]
        ax.bar(xpos + (mi - 1) * width, means, width, yerr=stds, capsize=2.5,
               color=ps.MODEL_COLORS[labels[m]], alpha=0.9, label=labels[m])
    ax.set_xticks(xpos)
    ax.set_xticklabels(WAVELETS, fontsize=8)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("$\\bar{S}_{\\mathrm{full}}$")
    ax.legend(fontsize=7.5, loc="upper right", ncol=3, frameon=True)
    ps.panel_label(ax, "a", x=-0.12, y=1.04)

    # ── (b) Ridge per-band hierarchy (bottom left) ───────────────
    ax = fig.add_subplot(gs[1, 0])
    xpos = np.arange(len(BANDS))
    for wi, w in enumerate(WAVELETS):
        means = [nc[w]["ridge"]["E_direct_mean"][b] for b in BANDS]
        ax.plot(xpos + wi * 0.13, means, "o-", ms=4.5, lw=1.3, label=w,
                alpha=0.9, color=ps.SIGMA_COLORS[list(ps.SIGMA_COLORS)[wi % 4]])
    ax.axhline(TAU, color=ps.TAU_COLOR, ls="--", lw=1.0)
    ax.text(0.04, TAU * 1.5, f"$\\tau$={TAU}", color=ps.TAU_COLOR, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(BANDS, fontsize=8)
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$ (log)")
    ax.legend(fontsize=7, loc="upper left", ncol=1, frameon=True)
    ps.panel_label(ax, "b", x=-0.18, y=1.04)

    # ── (c) analytical detection correct fraction (bottom right) ──
    ax = fig.add_subplot(gs[1, 1])
    cases = ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3",
             "E1_del_W1_only", "E2_partial_W3"]
    case_short = ["A", "B", "C", "D", "E1", "E2"]
    xw = np.arange(len(WAVELETS))
    for ci, c in enumerate(cases):
        fracs = [ana[w][c]["correct_frac"] for w in WAVELETS]
        ax.plot(xw, fracs, "o-", ms=4.5, lw=1.3, label=case_short[ci],
                color=ps.MODEL_COLORS[list(ps.MODEL_COLORS)[ci % 4]])
    ax.set_xticks(xw)
    ax.set_xticklabels(WAVELETS, fontsize=8)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("correct fraction")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=7, ncol=3, loc="lower left", frameon=True)
    ps.panel_label(ax, "c", x=-0.18, y=1.04)

    ps.save(fig, OUT_DIR, "fig11_wavelet_sensitivity")
    print(f"  [Fig11] S_full_mode: "
          f"{[(w, {m: nc[w][m]['S_full_mode'] for m in models}) for w in WAVELETS]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

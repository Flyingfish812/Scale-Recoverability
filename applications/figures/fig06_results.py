#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig06_results.py — Fig. 6 MLP results, three panels + Fig. 7 phase diagram (v3-3, 2026-08-31)

Fig. 6 (sixth figure in manuscript citation order):
  fig06_ger_vs_M.pdf      (a) Mean GER vs M @ σ=0
  fig06_sfull_vs_M.pdf    (b) Mean S_full vs M @ σ=0
  fig06_noise_sfull.pdf   (c) Mean S_full vs σ @ M=20,30
Fig. 7: fig07_phase_diagram.pdf (5M x 4σ multi-line)
Data sources: three_layer_fixed.json (GER) + s26_pass_probability.json (S_full, authoritative)
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
SIGMA_LABELS = ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]


def load_ger() -> list:
    records = json.loads((STATS / "band_error_decomposition.json").read_text())["records"]
    out = []
    for m in M_LIST:
        gers = [r["global_error"] for r in records
                if r["model"] == "mlp" and r["sensor_count"] == m and r["noise_sigma"] == 0.0]
        out.append(float(np.mean(gers)) if gers else 0.0)
    return out


def load_sfull() -> dict:
    stats = json.loads((STATS / "sensor_noise_phase.json").read_text())
    p = stats["phase_summary"]["mlp"]
    out = {}
    for m in M_LIST:
        for s in SIGMA_LIST:
            e = p.get(str(m), {}).get(str(s), {})
            out[(m, s)] = float(e.get("mean_S_full", 0.0)) if e else 0.0
    return out


def main() -> int:
    ps.apply()

    ger = load_ger()
    sfull = load_sfull()

    # (a) GER vs M
    fig, ax = ps.figure(3.25, 2.7, 1, 1)
    ax.plot(M_LIST, ger, color=ps.MODEL_COLORS["MLP"],
            marker=ps.MODEL_MARKERS["MLP"], lw=1.4, ms=5)
    ax.set_xlabel("Sensor count $M$")
    ax.set_ylabel("Mean GER")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 3e-2)
    ax.set_xticks(M_LIST)
    ps.save(fig, OUT_DIR, "fig06_ger_vs_M")

    # (b) S_full vs M
    fig, ax = ps.figure(3.25, 2.7, 1, 1)
    vals = [sfull[(m, 0.0)] for m in M_LIST]
    ax.plot(M_LIST, vals, color=ps.MODEL_COLORS["VCNN"],
            marker=ps.MODEL_MARKERS["VCNN"], lw=1.4, ms=5)
    ax.axhline(5, color="0.5", ls="--", lw=0.8)
    ax.axhline(3, color=ps.TAU_COLOR, ls=":", lw=0.8)
    ax.set_xlabel("Sensor count $M$")
    ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax.set_ylim(2.0, 5.2)
    ax.set_xticks(M_LIST)
    ps.save(fig, OUT_DIR, "fig06_sfull_vs_M")

    # (c) S_full vs σ
    fig, ax = ps.figure(3.25, 2.7, 1, 1)
    x = [1e-4, 1e-3, 1e-2, 1e-1]
    for m, color in [(20, ps.MODEL_COLORS["MLP"]), (30, ps.MODEL_COLORS["Ridge"])]:
        row = [sfull[(m, s)] for s in SIGMA_LIST]
        ax.plot(x, row, color=color, marker="o", lw=1.4, ms=5, label=f"$M$={m}")
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.set_xlim(6e-5, 0.28)
    ax.set_ylim(-0.2, 5.2)
    ax.set_xlabel("Noise level $\\sigma$")
    ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax.legend(fontsize=7.5, loc="center left")
    ps.save(fig, OUT_DIR, "fig06_noise_sfull")

    # Fig. 7 phase diagram
    fig, ax = ps.figure(6.6, 2.9, 1, 1)
    for si, s in enumerate(SIGMA_LIST):
        ax.plot(M_LIST, [sfull[(m, s)] for m in M_LIST],
                color=ps.SIGMA_COLORS[s], marker=ps.SIGMA_MARKERS[s],
                lw=1.4, ms=5, label=f"$\\sigma$={SIGMA_LABELS[si]}")
    ax.axhline(3, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("Sensor count $M$")
    ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax.set_ylim(-0.2, 5.2)
    ax.set_xticks(M_LIST)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ps.save(fig, OUT_DIR, "fig07_phase_diagram")

    print(f"  [Fig6a] GER={[round(g,4) for g in ger]}")
    print(f"  [Fig6b] S_full(M,σ=0)={[round(sfull[(m,0.0)],2) for m in M_LIST]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

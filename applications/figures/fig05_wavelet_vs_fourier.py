#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig05_wavelet_vs_fourier.py — Fig. 5 scale diagnostics vs Fourier (v3-3, 2026-08-31)

Two panels:
  (a) S_full (wavelet) vs S_FFT (Fourier dyadic) scatter + count labels
  (b) controlled-truncation agreement: detection accuracy for wavelet- vs
      Fourier-truncated reconstructions (S_full / S_FFT)
Data sources (verified):
  - artifacts/statistics/fourier_band_baseline.{csv,json}
  - artifacts/statistics/transform_symmetry_check.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
DATA = _ROOT / "artifacts" / "statistics"


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    # ── (a) scatter ──────────────────────────────────────────────
    s_full, s_fft = [], []
    with open(DATA / "fourier_band_baseline.csv") as f:
        for row in csv.DictReader(f):
            s_full.append(int(float(row["S_full"])))
            s_fft.append(int(float(row["S_FFT"])))
    s_full = np.array(s_full)
    s_fft = np.array(s_fft)
    fb = json.loads((DATA / "fourier_band_baseline.json").read_text())
    corr = fb.get("correlations", {})

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1),
                             gridspec_kw=dict(wspace=0.3))

    ax = axes[0]
    jit = np.random.RandomState(42).uniform(-0.15, 0.15, size=len(s_full))
    ax.scatter(s_full + jit, s_fft + jit, alpha=0.55, s=16, c=ps.MODEL_COLORS["MLP"],
               edgecolors="white", linewidth=0.3)
    ax.plot([-0.5, 5.5], [-0.5, 5.5], ls="--", lw=0.9, color="0.45",
            label="perfect agreement")
    for sv in range(6):
        for fv in range(6):
            c = int(((s_full == sv) & (s_fft == fv)).sum())
            if c > 3:
                ax.text(sv + 0.2, fv + 0.2, str(c), fontsize=6.5,
                        ha="center", va="center", color=ps.ACCENT, alpha=0.7)
    ax.set_xlabel("$S_{\\mathrm{full}}$ (wavelet)")
    ax.set_ylabel("$S_{\\mathrm{FFT}}$ (Fourier dyadic)")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    # legend inside the axes, lower right (single entry, white background)
    ax.legend(fontsize=7, loc="lower right", frameon=True,
              bbox_to_anchor=(0.99, 0.01), borderaxespad=0.2)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ps.panel_label(ax, "a")

    # ── (b) controlled truncation accuracy ───────────────────────
    sc = json.loads((DATA / "transform_symmetry_check.json").read_text())
    wbl = [r for r in sc["results"] if r["experiment"] == "wavelet_band_limited"]
    fal = [r for r in sc["results"] if r["experiment"] == "fourier_annulus_limited"]

    def acc(results):
        fields = sorted({r["field_idx"] for r in results
                         if r["expected_recoverable"] is not None})
        out = {}
        for fid in fields:
            sub = [r for r in results if r["field_idx"] == fid
                   and r["expected_recoverable"] is not None]
            out[fid] = (sum(1 for r in sub if r["S_full_correct"]) / len(sub) * 100,
                        sum(1 for r in sub if r["S_FFT_correct"]) / len(sub) * 100,
                        len(sub))
        return out

    w = acc(wbl)
    f = acc(fal)
    ax = axes[1]
    # grouping: 2 bars per experiment (S_full / S_FFT)
    wbl_sf = float(np.mean([w[k][0] for k in w]))
    wbl_fft = float(np.mean([w[k][1] for k in w]))
    fal_sf = float(np.mean([f[k][0] for k in f]))
    fal_fft = float(np.mean([f[k][1] for k in f]))
    xpos = np.arange(2)
    width = 0.34
    ax.bar(xpos - width / 2, [wbl_sf, fal_sf], width,
           color=ps.MODEL_COLORS["Gappy"], alpha=0.9, label="$S_{\\mathrm{full}}$")
    ax.bar(xpos + width / 2, [wbl_fft, fal_fft], width,
           color=ps.MODEL_COLORS["MLP"], alpha=0.9, label="$S_{\\mathrm{FFT}}$")
    for xi, v in enumerate([wbl_sf, fal_sf]):
        ax.text(xi - width / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=7,
                color=ps.MODEL_COLORS["Gappy"])
    for xi, v in enumerate([wbl_fft, fal_fft]):
        ax.text(xi + width / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=7,
                color=ps.MODEL_COLORS["MLP"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Wavelet-truncated\nreconstructions",
                        "Fourier-annulus-\ntruncated reconstructions"], fontsize=7.5)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.6, 1.6)
    # legend inside the axes, lower right (white background, clear of labels)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(0.99, 0.01),
              frameon=True, borderaxespad=0.2)
    ps.panel_label(ax, "b")

    # panel (b): the two truncation experiments give the same accuracy
    print(f"  [ctrl] wavelet-trunc. ({len(w)} fields): S_full={wbl_sf:.0f}% "
          f"S_FFT={wbl_fft:.0f}%; fourier-trunc. ({len(f)} fields): "
          f"S_full={fal_sf:.0f}% S_FFT={fal_fft:.0f}%")
    print(f"  [corr] {corr}")

    ps.save(fig, OUT_DIR, "fig05_wavelet_vs_fourier")
    return 0


if __name__ == "__main__":
    sys.exit(main())

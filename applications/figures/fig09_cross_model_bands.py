#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig09_cross_model_bands.py — main figure 5: cross-model band errors.

At (M, sigma) = (20, 0): per-band direct error of MLP, closed-form Ridge and VCNN, with the tolerance line and the first-failed-band labels.

Data: artifacts/statistics/band_error_records.json (MLP and VCNN records,
      300 test snapshots x 3 seeds) and artifacts/statistics/truncation_reference_audit.json (Ridge summary), both through ``applications/figures/records.py``.
Output: artifacts/figures/fig09_cross_model_bands.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import records
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
STATS = _ROOT / "artifacts" / "statistics"
BANDS = ps.BANDS
TAU = ps.TAU


def _clipped_yerr(means, stds):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    return np.vstack([np.minimum(stds, means), stds])


def main() -> int:
    ps.apply()

    ridge_band = records.config_summary("ridge", 20, 0.0)["per_band_mean"]

    results = records.band_records()
    model_data = {}
    for mt in ["mlp", "vcnn"]:
        samples = [r for r in results if r["model_type"] == mt
                   and r["mask_num"] == 20 and r["noise_sigma"] == 0.0]
        if samples:
            model_data[mt] = {}
            for b in BANDS:
                vals = [r.get(f"E_total_{b}") for r in samples
                        if r.get(f"E_total_{b}") is not None and r[f"E_total_{b}"] > 0]
                model_data[mt][b] = (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    fig, ax = ps.figure(6.6, 3.2, 1, 1)
    x = np.arange(len(BANDS))
    width = 0.18
    YMIN, YMAX = 1.5e-3, 0.5

    series = [
        ("MLP", -width, model_data["mlp"], ps.MODEL_COLORS["MLP"]),
        ("Ridge", 0.0, None, ps.MODEL_COLORS["Ridge"]),
        ("VCNN", width, model_data["vcnn"], ps.MODEL_COLORS["VCNN"]),
    ]
    bars_data = {}
    for name, off, md, color in series:
        if md is None:
            means = [ridge_band[b] for b in BANDS]
            stds = None
        else:
            means = [md[b][0] for b in BANDS]
            stds = [md[b][1] for b in BANDS]
        ax.bar(x + off, means, width, bottom=YMIN, color=color, alpha=0.9,
               label=name)
        if stds is not None:
            lo = [min(s, m) for m, s in zip(means, stds)]
            ax.errorbar(x + off, means, yerr=lo, fmt="none", color="0.15",
                        capsize=2, capthick=0.8, lw=0.8)
        bars_data[name] = (off, means, color)

    # ── log y axis (error convention), failure region shaded to the top ──
    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlim(-0.55, 5.0)
    ax.axhline(TAU, color=ps.TAU_COLOR, ls="--", lw=1.1)
    ax.axhspan(TAU, YMAX, color=ps.TAU_COLOR, alpha=0.07, zorder=0)
    # fail label at the far left (x axis left end, below the τ line)
    ax.text(-0.50, YMIN * 2.2, f"fail $>\\tau$={TAU}", color=ps.TAU_COLOR,
            fontsize=7.5, va="bottom")

    # first-failed-band labels — vertical arrows point at the bar tops text sits in one row above the τ line, clear of the line and labels
    for name, (off, means, color) in bars_data.items():
        k = next((i for i, m in enumerate(means) if m > TAU), None)
        if k is not None:
            y_txt = max(means[k] * 2.6, 0.13)   # text above the bar top and τ line
            ax.annotate(f"{name} mean fails at {BANDS[k]}",
                        xy=(x[k] + off, means[k]),
                        xytext=(x[k] + off, y_txt),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                        fontsize=7, color=color, ha="center")
        else:
            # all pass: plain text (no arrow) in the free upper-right area; the
            # criterion is on the band errors averaged over the records first.
            label = f"{name}: all mean band errors below " + r"$\tau$"
            ax.text(3.05, 0.30, label, fontsize=7, color=color, ha="center")

    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("Mean $E_{\\mathrm{direct}}(b)$")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(-0.02, 1.0))
    ps.save(fig, OUT_DIR, "fig09_cross_model_bands")

    print(f"  [Fig9] MLP={[round(model_data['mlp'][b][0],4) for b in BANDS]}")
    print(f"  [Fig9] Ridge={[round(ridge_band[b],4) for b in BANDS]}")
    print(f"  [Fig9] VCNN={[round(model_data['vcnn'][b][0],4) for b in BANDS]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

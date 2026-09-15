#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig08_recoverability_chain.py — Fig. 8 error chain (v3-3, 2026-08-31)

Two panels at M=20, σ=0 for Oracle / MLP / VCNN / Ridge / Gappy:
  (a) S_full bar comparison
  (b) GER bar comparison (log)
Data sources (all verified): three_layer_errors_full.json (300 snapshots x 3 seeds)
+ s05_true_ridge.json + s23_gappy_pod_fixed.json
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


def load_json(name):
    return json.loads((STATS / name).read_text(encoding="utf-8"))


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    # ── data ────────────────────────────────────────────────────
    results = records.band_records()

    def avg(model, metric, mask=20, sigma=0.0):
        vals = [r[metric] for r in results if r["model_type"] == model
                and r["mask_num"] == mask and r["noise_sigma"] == sigma]
        return float(np.mean(vals)) if vals else 0.0

    mlp_ger, mlp_sf = avg("mlp", "GER"), avg("mlp", "S_full_total")
    vcnn_ger, vcnn_sf = avg("vcnn", "GER"), avg("vcnn", "S_full_total")

    ridge = records.config_summary("ridge", 20, 0.0)
    ridge_ger, ridge_sf = ridge["GER_mean"], ridge["S_full_mean"]

    s23 = load_json("gappy_pod_baseline.json")
    gappy_ger = next((r["test_ger_mean"] for r in s23["results"]
                      if r["mask_num"] == 20 and r["sigma"] == 0.0), 0.0333)
    gappy_sf = 0.53  # fixed value from paper_facts (original method S_full)

    oracle_ger, oracle_sf = 0.0006, 5.0

    models = [("POD trunc.", oracle_ger, oracle_sf, ps.MODEL_COLORS["Oracle"]),
              ("MLP", mlp_ger, mlp_sf, ps.MODEL_COLORS["MLP"]),
              ("VCNN", vcnn_ger, vcnn_sf, ps.MODEL_COLORS["VCNN"]),
              ("Ridge", ridge_ger, ridge_sf, ps.MODEL_COLORS["Ridge"]),
              ("Gappy POD", gappy_ger, gappy_sf, ps.MODEL_COLORS["Gappy"])]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0),
                             gridspec_kw=dict(wspace=0.3))

    # (a) S_full
    ax = axes[0]
    labels = [m[0] for m in models]
    sf_vals = [m[2] for m in models]
    colors = [m[3] for m in models]
    bars = ax.bar(labels, sf_vals, color=colors, alpha=0.9, width=0.62)
    ax.axhline(5, color="0.5", ls="--", lw=0.8)
    for bar, val in zip(bars, sf_vals):
        s = f"{int(val)}" if float(val).is_integer() else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08 if val < 5 else 0.12,
                s, ha="center", fontsize=7)
    ax.set_ylabel("$S_{\\mathrm{full}}$")
    ax.set_ylim(0, 5.8)
    ax.set_yticks(range(0, 6))
    ax.tick_params(axis="x", labelsize=7.5)
    ps.panel_label(ax, "a", x=-0.22, y=1.04)

    # (b) GER
    ax = axes[1]
    ger_vals = [m[1] for m in models]
    bars = ax.bar(labels, ger_vals, color=colors, alpha=0.9, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("GER (log)")
    for bar, val in zip(bars, ger_vals):
        s = f"{val:.4f}" if val < 0.01 else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                s, ha="center", fontsize=6.5, rotation=30)
    ax.set_ylim(top=6e-2)
    ax.tick_params(axis="x", labelsize=7.5)
    ps.panel_label(ax, "b", x=-0.22, y=1.04)

    ps.save(fig, OUT_DIR, "fig08_recoverability_chain")
    print(f"  [chain] POD-trunc S={oracle_sf} G={oracle_ger:.4f}; MLP S={mlp_sf:.2f} "
          f"G={mlp_ger:.4f}; VCNN S={vcnn_sf:.2f} G={vcnn_ger:.4f}; "
          f"Ridge S={ridge_sf:.2f} G={ridge_ger:.4f}; Gappy S={gappy_sf:.2f} G={gappy_ger:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

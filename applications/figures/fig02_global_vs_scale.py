#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_global_vs_scale.py — main figure 2: scale information beyond the global relative error.

The four panels share one text-width row layout. Every global error shown here is the streamwise-component relative error GER_u, consistent with the metric the pairs are matched on:

  (a) low-GER_u cross-model example: Ridge versus VCNN per-band E_direct at
      (M, sigma) = (30, 0), snapshot 49. The two global errors are NOT equal;
      the panel is not a matched pair.
  (b) representative within-configuration matched pair (MLP, M = 50,
      sigma = 10^-3, seed 0; snapshots 98 and 227): per-band E_direct of the two
      members, whose GER_u values agree to within the 1 % matching criterion.
  (c) ECDF of the relative GER_u difference over all matched pairs, grouped by
      Delta S_full, with the 1 % relative matching criterion shown as a dotted
      line.
  (d) pooled statistics of the same pairs: median W1 band error, Laplacian RMSE
      and gradient RMSE for the low- and high-S_full member, with the snapshot-clustered bootstrap p-values.

Data: artifacts/statistics/band_error_decomposition.json (a),
      artifacts/statistics/band_error_records.json (b),
      artifacts/statistics/equal_ger_pairs.json (c, d).
No pair, baseline or statistic is added here.

Output: artifacts/figures/fig02_global_vs_scale.pdf
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

# (a): closed-form Ridge and VCNN on the same snapshot, flagged in Fig. 3a
TYPE_A = [("ridge", 30, 0.0, 49, 0, "Ridge"),
          ("vcnn", 30, 0.0, 49, 202, "VCNN")]
# (b): the representative pair of the equal-GER_u search
PAIR = dict(model="mlp", sensors=20, sigma=0.001, seed=0, idx=(240, 194))
# published matching criterion of the pair search: 1 % relative GER difference
GER_TOL = 0.01
BANDS = ps.BANDS

TICK = 6.5


def panel_low_ger(ax, records: list) -> None:
    """(a) low-GER_u cross-model example, per-band error bars."""
    x = np.arange(len(BANDS))
    width = 0.36
    for i, (model, mask, sigma, snap, seed, name) in enumerate(TYPE_A):
        rec = next(r for r in records
                   if r["model"] == model and r["sensor_count"] == mask
                   and r["noise_sigma"] == sigma and r["snapshot_index"] == snap
                   and r["training_seed"] == seed)
        errs = [rec["band_errors"][b]["total"] for b in BANDS]
        off = -width / 2 if i == 0 else width / 2
        ax.bar(x + off, errs, width, color=ps.MODEL_COLORS[name], alpha=0.9,
               label=f"{name}: $\\mathrm{{GER}}_u$ "
                     f"{rec.get('global_error_u', rec['global_error']):.5f}, "
                     f"$S_{{\\mathrm{{full}}}}$ = {rec['s_full']}")
        fail = next((k for k, e in enumerate(errs) if e > ps.TAU), None)
        if fail is not None:
            ax.plot([x[fail] + off], [errs[fail]], marker="v", ms=5,
                    color=ps.ACCENT, ls="none", zorder=6)
    ax.axhline(ps.TAU, color=ps.TAU_COLOR, ls="--", lw=0.9)
    ax.text(0.995, ps.TAU * 1.15, f"$\\tau$ = {ps.TAU}",
            transform=ax.get_yaxis_transform(), fontsize=TICK,
            color=ps.TAU_COLOR, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 3e-1)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_xlabel("wavelet band")
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$")
    ax.legend(fontsize=6.8, loc="upper left", frameon=False, handlelength=1.2,
              handletextpad=0.5)
    ps.panel_label(ax, "a", x=-0.11, y=1.02)


def panel_pair(ax, records: list, pairs: dict) -> None:
    """(b) per-band curves of the representative matched pair."""
    sel = [r for r in records
           if r["model"] == PAIR["model"] and r["sensor_count"] == PAIR["sensors"]
           and r["noise_sigma"] == PAIR["sigma"]
           and r["training_seed"] == PAIR["seed"]
           and r["snapshot_index"] in PAIR["idx"]]
    x = np.arange(len(BANDS))
    styles = [(ps.MODEL_COLORS["Ridge"], "o"), (ps.MODEL_COLORS["Gappy"], "s")]
    for (color, marker), idx in zip(styles, PAIR["idx"]):
        rec = next(r for r in sel if r["snapshot_index"] == idx)
        errs = [rec["band_errors"][b]["total"] for b in BANDS]
        ax.plot(x, errs, color=color, marker=marker, ms=3.2, lw=1.2,
                label=f"$S_{{\\mathrm{{full}}}}$ = {rec['s_full']}")
        fail = next((k for k, e in enumerate(errs) if e > ps.TAU), None)
        if fail is not None:
            ax.plot([x[fail]], [errs[fail]], marker="v", ms=5.0,
                    color=ps.ACCENT, ls="none", zorder=6)
    ax.axhline(ps.TAU, color=ps.TAU_COLOR, ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 4e-1)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS, fontsize=TICK)
    ax.set_xlabel("wavelet band", fontsize=TICK + 0.5)
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$", fontsize=TICK + 0.5)
    ax.tick_params(labelsize=TICK)
    rep = pairs["representative_pair"]
    ax.set_title(f"matched pair, $\\mathrm{{GER}}_u$ {rep['GER_low']:.5f} in both",
                 fontsize=6.8, pad=3)
    ax.legend(fontsize=TICK, loc="lower right", frameon=False, handlelength=1.0,
              handletextpad=0.4, borderaxespad=0.2)
    ps.panel_label(ax, "b", x=-0.26, y=1.02)


def panel_ecdf(ax, pairs: dict) -> None:
    """(c) relative GER_u difference of all matched pairs."""
    ger_diff = np.array([p["GER_diff"] for p in pairs["all_pairs"]])
    sfull_diff = np.array([p["S_full_diff"] for p in pairs["all_pairs"]])
    for value, color in [(2, ps.MODEL_COLORS["MLP"]),
                         (3, ps.MODEL_COLORS["Ridge"])]:
        sel = np.sort(ger_diff[sfull_diff == value])
        ax.step(sel, np.arange(1, len(sel) + 1) / len(sel), where="post",
                lw=1.1, color=color,
                label=f"$\\Delta S$ = {value} (n = {len(sel)})")
    ax.plot([float(np.min(ger_diff))], [0.5], marker="*", ms=6.5,
            color=ps.ACCENT, ls="none", zorder=6)
    ax.axvline(GER_TOL, color="0.45", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(3e-7, 3e-2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("relative $\\Delta\\mathrm{GER}_u$", fontsize=TICK + 0.5)
    ax.set_ylabel("cumulative fraction", fontsize=TICK + 0.5)
    ax.tick_params(labelsize=TICK)
    ax.legend(fontsize=TICK - 0.5, loc="upper left", frameon=False,
              handlelength=1.0, handletextpad=0.35, borderaxespad=0.15)
    ps.panel_label(ax, "c", x=-0.26, y=1.02)


def panel_pooled(ax, pairs: dict) -> None:
    """(d) pooled medians of the matched pairs."""
    tests = {t["label"]: t for t in pairs["paired_statistics"]["tests"]}
    boot = {r["label"]: r for r in pairs["cluster_bootstrap_ci"]["results"]}
    metrics = [("W1", "W1_band_error", "W1_band_error_diff"),
               ("Laplacian", "vorticity_RMSE", "vorticity_RMSE_diff"),
               ("Gradient", "gradient_RMSE", "gradient_RMSE_diff")]
    x = np.arange(len(metrics))
    width = 0.34
    for i, (label, key, boot_key) in enumerate(metrics):
        low = tests[key]["median_low_S_full"]
        high = tests[key]["median_high_S_full"]
        ax.bar(i - width / 2, low, width, color=ps.MODEL_COLORS["Ridge"],
               alpha=0.9, label="low $S_{\\mathrm{full}}$" if i == 0 else "")
        ax.bar(i + width / 2, high, width, color=ps.MODEL_COLORS["Gappy"],
               alpha=0.9, label="high $S_{\\mathrm{full}}$" if i == 0 else "")
        p_val = boot[boot_key]["p_value_bootstrap"]
        p_str = "$p$ < 0.001" if p_val < 0.001 else f"$p$ = {p_val:.3f}"
        ax.text(i, max(low, high) * 1.30, p_str, ha="center", fontsize=TICK)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 4e-1)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=TICK)
    ax.set_ylabel("median (log)", fontsize=TICK + 0.5)
    ax.tick_params(labelsize=TICK)
    ax.legend(fontsize=TICK - 0.5, loc="upper right", frameon=False,
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.15)
    ax.set_title(f"{pairs['paired_statistics']['n_pairs']} matched pairs",
                 fontsize=6.8, pad=3)
    ps.panel_label(ax, "d", x=-0.26, y=1.02)


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    decomposition = json.loads(
        (STATS / "band_error_decomposition.json").read_text(encoding="utf-8"))["records"]
    records = json.loads(
        (STATS / "band_error_records.json").read_text(encoding="utf-8"))["records"]
    pairs = json.loads((STATS / "equal_ger_pairs.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(5.33, 2.50))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.90,
                          wspace=0.62, left=0.085, right=0.985, top=0.93,
                          bottom=0.135)

    ax = fig.add_subplot(gs[0, :])
    panel_low_ger(ax, decomposition)

    ax = fig.add_subplot(gs[1, 0])
    panel_pair(ax, records, pairs)
    ax = fig.add_subplot(gs[1, 1])
    panel_ecdf(ax, pairs)
    ax = fig.add_subplot(gs[1, 2])
    panel_pooled(ax, pairs)

    ps.save(fig, OUT_DIR, "fig02_global_vs_scale")

    rep = pairs["representative_pair"]
    print(f"  [pair] {rep['config_key']} idx {PAIR['idx']}: S_full "
          f"{rep['S_full_low']} vs {rep['S_full_high']}, GER "
          f"{rep['GER_low']:.9f} vs {rep['GER_high']:.9f}")
    print(f"  [pairs] n={pairs['paired_statistics']['n_pairs']}; "
          f"tolerance {pairs['matching_criteria']['ger_tolerance']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_known_scale.py — JFM v0-2 Fig. 1: known-scale validation (composite).

Combines the old manuscript Fig. 2 (analytical benchmark) with the scale-count
information of old Tables 1 and 12, at the JFM text width (384 pt = 5.33 in):

  (a) representative prescribed-scale analytical wake (streamwise velocity u)
      and the W1-associated carrier field that cases B and E1 remove
  (b) prescribed versus measured scale count for cases A-E2 (B-D = 4/3/2)
  (c) per-band E_direct of the equal-GER cases E1 and E2 (S_full = 4 versus 2)

Data: artifacts/statistics/analytical_benchmark.json for (b) and (c); the field
in (a) is the same deterministic construction (luna.benchmarks.analytical_wake,
default parameters). Construction parameters and the complete case table stay in
supplementary Table S1; no case or baseline is added here.

Output: artifacts/figures/fig01_known_scale.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from luna.benchmarks.analytical_wake import WakeParams, snapshot, scale_u_components
from luna.core.constants import BANDS_CF, TAU_DEFAULT

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
CONSOL = _ROOT / "artifacts" / "statistics" / "analytical_benchmark.json"

# case order and the labels used in the composite figure
CASES = [("A_full", "A"), ("B_del_W1", "B"), ("C_del_W1W2", "C"),
         ("D_del_W3", "D"), ("E1_del_W1_only", "E1"),
         ("E2_partial_W3", "E2")]


def load_cases() -> dict:
    return json.loads(CONSOL.read_text(encoding="utf-8"))["cases"]


def panel_field(ax, u: np.ndarray, title: str) -> None:
    """Field panel of the 2:1 wake grid, one symmetric linear colour scale."""
    vmax = float(np.abs(u).max())
    ax.imshow(u.T, origin="lower", cmap="RdBu_r", aspect="auto",
              vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, pad=1.8)


def panel_counts(ax, cases: dict) -> None:
    """(b) prescribed versus measured consecutive scale count."""
    x = np.arange(len(CASES))
    measured = [cases[c]["S_full_mean"] for c, _ in CASES]
    prescribed = [cases[c]["expected_S_full"] for c, _ in CASES]
    ax.bar(x, measured, width=0.62, color=ps.MODEL_COLORS["MLP"], alpha=0.85,
           label="measured $S_{\\mathrm{full}}$")
    ax.plot(x, prescribed, "k_", ms=11, mew=1.6, ls="none",
            label="prescribed")
    ax.set_ylim(0, 6.0)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CASES])
    ax.set_xlabel("case")
    ax.set_ylabel("scale count")
    ax.grid(axis="x", alpha=0.0)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.30),
              ncol=2, frameon=False, handlelength=1.2, columnspacing=0.9)
    # the equal-GER pair shares one annotation: GER is equal for E1 and E2
    # both are streamwise-component errors (GER_u), see the main text
    ger_e1 = cases["E1_del_W1_only"]["GER_mean"]
    ger_e2 = cases["E2_partial_W3"]["GER_mean"]
    ax.annotate("", xy=(4.30, 4.45), xytext=(5.70, 4.45),
                arrowprops=dict(arrowstyle="-", color=ps.ACCENT, lw=1.0))
    ax.text(5.0, 4.58, f"$\\mathrm{{GER}}_u$ {ger_e1:.4f} in both",
            ha="center", va="bottom", fontsize=6.5, color=ps.ACCENT)


def panel_e1_e2(ax, cases: dict) -> None:
    """(c) per-band error of the two equal-GER cases."""
    e1 = cases["E1_del_W1_only"]
    e2 = cases["E2_partial_W3"]
    x = np.arange(len(BANDS_CF))
    for case, color, label in [(e1, ps.MODEL_COLORS["MLP"], "E1"),
                               (e2, ps.MODEL_COLORS["Ridge"], "E2")]:
        mean = [case["E_direct_mean"][b] for b in BANDS_CF]
        std = [case["E_direct_std"][b] for b in BANDS_CF]
        ax.errorbar(x, mean, yerr=std, color=color, marker="o", ms=3.5,
                    lw=1.2, elinewidth=0.7, capsize=1.6,
                    label=f"{label}  $S_{{\\mathrm{{full}}}}$ = "
                          f"{int(round(case['S_full_mean']))}")
        fail = next((k for k, b in enumerate(BANDS_CF)
                     if case["E_direct_mean"][b] > TAU_DEFAULT), None)
        if fail is not None:
            ax.plot([x[fail]], [mean[fail]], marker="v", ms=5.5,
                    color=ps.ACCENT, ls="none", zorder=6)
    ax.axhline(TAU_DEFAULT, color=ps.TAU_COLOR, ls="--", lw=0.9)
    ax.text(0.02, TAU_DEFAULT * 1.12, f"$\\tau$ = {TAU_DEFAULT}",
            transform=ax.get_yaxis_transform(), fontsize=6.5,
            color=ps.TAU_COLOR, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(5e-6, 3.0)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_CF)
    ax.set_xlabel("wavelet band")
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.2)


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    cases = load_cases()
    params = WakeParams()
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    u = snapshot(x, y, params, 0)
    # the W1-associated carrier field: exactly what cases B and E1 remove
    w1_carriers = scale_u_components(x, y, params, 0)[5]

    fig = plt.figure(figsize=(5.33, 2.10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.10, 1.10], wspace=0.60,
                          left=0.065, right=0.985, top=0.80, bottom=0.185)

    sub = gs[0, 0].subgridspec(2, 1, hspace=0.32)
    ax = fig.add_subplot(sub[0, 0])
    panel_field(ax, u, "streamwise velocity $u$")
    ps.panel_label(ax, "a", x=-0.28, y=1.06)
    ax = fig.add_subplot(sub[1, 0])
    panel_field(ax, w1_carriers, "W$_1$-associated carriers")

    ax = fig.add_subplot(gs[0, 1])
    panel_counts(ax, cases)
    ps.panel_label(ax, "b", x=-0.30, y=1.02)

    ax = fig.add_subplot(gs[0, 2])
    panel_e1_e2(ax, cases)
    ps.panel_label(ax, "c", x=-0.30, y=1.02)

    ps.save(fig, OUT_DIR, "fig01_known_scale")

    print("  [case counts] " + ", ".join(
        f"{lab}:{cases[c]['S_full_mean']:.0f}/{cases[c]['expected_S_full']}"
        for c, lab in CASES))
    print(f"  [E1/E2] GER {cases['E1_del_W1_only']['GER_mean']:.5f} vs "
          f"{cases['E2_partial_W3']['GER_mean']:.5f}; S_full "
          f"{cases['E1_del_W1_only']['S_full_mean']:.0f} vs "
          f"{cases['E2_partial_W3']['S_full_mean']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

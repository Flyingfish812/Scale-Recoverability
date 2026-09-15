#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_analytical_benchmark.py — Fig. 2 known-scale validation (v3-3, 2026-09-01)

Field panels keep the 2:1 aspect ratio of the data, and a colorbar is added only
where the neighbouring panel leaves room. Figures show data only; conclusions
stay in the caption.

Manuscript version: (a) Target u | (b) Band A4 | (c) Band W3
             (d) E1 recon | (e) E2 recon | (f) per-band error E1 vs E2
Appendix full version: keeps all (a)-(m) band decomposition panels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from luna.benchmarks.analytical_wake import (
    WakeParams, snapshot, scale_u_components, velocity, case_metrics,
)
from luna.core.constants import BANDS_CF, TAU_DEFAULT
from luna.wavelet.transform import decompose_field_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
CONSOL = _ROOT / "artifacts" / "statistics" / "analytical_benchmark.json"


def _recon_image(u, u_j, cname):
    if cname == "A_full":
        return u.copy()
    if cname in ("B_del_W1", "E1_del_W1_only"):
        return u - u_j[5]
    if cname == "C_del_W1W2":
        return u - u_j[5] - u_j[4]
    if cname == "D_del_W3":
        return u - u_j[3]
    if cname == "E2_partial_W3":
        recs = case_metrics(u, u_j, tau=TAU_DEFAULT)
        return u - _alpha_for_ger(u, u_j, recs["E1_del_W1_only"]["GER"]) * u_j[3]
    raise KeyError(cname)


def _alpha_for_ger(u, u_j, target_ger, tol=1e-7):
    from luna.wavelet.metrics import rel_l2
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rel_l2(u - mid * u_j[3], u) < target_ger:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _field_panel(fig, ax, arr, label, vmax=None, colorbar=True,
                 show_label=True):
    """Field panel: automatic 2:1 aspect ratio, optional compact colorbar."""
    if vmax is None:
        vmax = float(np.abs(arr).max())
    im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_label:
        ps.panel_label(ax, label, x=-0.10, y=1.02)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015, aspect=16)
    return im


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    params = WakeParams()
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    u = snapshot(x, y, params, 0)
    u_j = scale_u_components(x, y, params, 0)
    v = velocity(x, y, params, 0)[1]
    recs = case_metrics(u, u_j, tau=TAU_DEFAULT)
    bd = decompose_field_2d(u)
    vm = float(np.abs(u).max())

    consol = json.loads(CONSOL.read_text(encoding="utf-8"))
    cE1, cE2 = consol["cases"]["E1_del_W1_only"], consol["cases"]["E2_partial_W3"]
    print(f"  E1: GER={cE1['GER_mean']:.4f} S_full={cE1['S_full_mean']:.2f} (expected 4)")
    print(f"  E2: GER={cE2['GER_mean']:.4f} S_full={cE2['S_full_mean']:.2f} (expected 2)")

    # ── manuscript version: 3 rows x 2 cols, fields kept at 2:1 ──
    # each column ~3.1in wide, field ~1.5in tall (2:1); tight row spacing
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15],
                          hspace=0.38, wspace=0.30,
                          left=0.05, right=0.985, top=0.93, bottom=0.08)

    # Row 1: (a) Target u | (b) Band A4
    _field_panel(fig, fig.add_subplot(gs[0, 0]), u, "a", vmax=vm)
    ax = fig.add_subplot(gs[0, 1])
    _field_panel(fig, ax, bd["A4"], "b", vmax=vm)
    ax.set_title("Band A4", fontsize=ps.TITLE_FONT)

    # Row 2: (c) Band W3 | (d) E1: W1 removed
    ax = fig.add_subplot(gs[1, 0])
    _field_panel(fig, ax, bd["W3"], "c", vmax=vm)
    ax.set_title("Band W3", fontsize=ps.TITLE_FONT)
    ax = fig.add_subplot(gs[1, 1])
    r = recs["E1_del_W1_only"]
    _field_panel(fig, ax, _recon_image(u, u_j, "E1_del_W1_only"), "d", vmax=vm,
                 colorbar=False)
    ax.set_title(f"E1: W1-associated carriers removed\nGER {cE1['GER_mean']:.4f}  $S_{{\\mathrm{{full}}}}$ = {r['S_full']}",
                 fontsize=ps.TITLE_FONT)

    # Row 3: (e) E2: partial W3 | (f) per-band error
    ax = fig.add_subplot(gs[2, 0])
    r = recs["E2_partial_W3"]
    _field_panel(fig, ax, _recon_image(u, u_j, "E2_partial_W3"), "e", vmax=vm,
                 colorbar=False)
    ax.set_title(f"E2: partial W3-associated carriers\nGER {cE2['GER_mean']:.4f}  $S_{{\\mathrm{{full}}}}$ = {r['S_full']}",
                 fontsize=ps.TITLE_FONT)

    # (f) per-band error: E1 vs E2
    ax = fig.add_subplot(gs[2, 1])
    xpos = np.arange(len(BANDS_CF))
    w = 0.36
    e1 = [max(recs["E1_del_W1_only"]["E_direct"][b], 1e-4) for b in BANDS_CF]
    e2 = [max(recs["E2_partial_W3"]["E_direct"][b], 1e-4) for b in BANDS_CF]
    ax.bar(xpos - w / 2, e1, w, color=ps.MODEL_COLORS["MLP"], alpha=0.92,
           label="E1 (W1-associated carriers removed)")
    ax.bar(xpos + w / 2, e2, w, color=ps.MODEL_COLORS["Ridge"], alpha=0.92,
           label="E2 (partial W3-associated carriers)")
    ax.axhline(ps.TAU, color=ps.TAU_COLOR, ls="--", lw=1.0)
    # τ label: outside the plotting area, top right
    ax.text(0.995, 1.045, f"$\\tau$ = {ps.TAU}", transform=ax.transAxes,
            fontsize=7.5, color=ps.TAU_COLOR, ha="right", va="bottom")
    # first-failed-band: red inverted triangle (no long annotation)
    for off, errs in [(-w / 2, e1), (w / 2, e2)]:
        for k in range(len(BANDS_CF)):
            if errs[k] > ps.TAU:
                ax.plot([xpos[k] + off], [errs[k] * 1.25], marker="v",
                        ms=8, color=ps.TAU_COLOR, zorder=6)
                break
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(BANDS_CF)
    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"])
    ax.legend(fontsize=7, loc="upper left", frameon=True)
    ps.panel_label(ax, "f", x=-0.16, y=1.02)

    ps.save(fig, OUT_DIR, "fig02_analytical_benchmark")

    # ── appendix full version: (a)-(m) ──────────────────────────
    fig2 = plt.figure(figsize=(6.6, 7.6))
    gs2 = fig2.add_gridspec(5, 1, height_ratios=[1.0, 0.95, 0.95, 1.15, 1.25],
                            hspace=0.62, left=0.05, right=0.985,
                            top=0.96, bottom=0.05)

    # (a,b) target u / v
    row = gs2[0].subgridspec(1, 2, wspace=0.22)
    _field_panel(fig2, fig2.add_subplot(row[0, 0]), u, "a")
    _field_panel(fig2, fig2.add_subplot(row[0, 1]), v, "b")

    # (c-e) A4, W4, W3 — colorbar on the first panel only
    row = gs2[1].subgridspec(1, 3, wspace=0.28)
    for i, b in enumerate(["A4", "W4", "W3"]):
        ax = fig2.add_subplot(row[0, i])
        _field_panel(fig2, ax, bd[b], chr(ord("c") + i),
                     colorbar=(i == 0))
        ax.set_title(f"Band {b}", fontsize=ps.TITLE_FONT)

    # (f-h) W2, W1, energy
    row = gs2[2].subgridspec(1, 3, wspace=0.28, width_ratios=[1, 1, 1.1])
    for i, b in enumerate(["W2", "W1"]):
        ax = fig2.add_subplot(row[0, i])
        _field_panel(fig2, ax, bd[b], chr(ord("f") + i),
                     colorbar=(i == 0))
        ax.set_title(f"Band {b}", fontsize=ps.TITLE_FONT)
    ax = fig2.add_subplot(row[0, 2])
    tot = float(np.sum(u ** 2))
    om = [float(np.sum(bd[b] ** 2) / tot) for b in BANDS_CF]
    ax.bar(np.arange(5), np.maximum(om, 1e-5),
           color=[ps.BAND_COLORS[b] for b in BANDS_CF],
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(BANDS_CF, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 8.0)
    ax.set_yticks([1e-4, 1e-2, 1e0])
    ax.set_yticklabels(["$10^{-4}$", "$10^{-2}$", "$10^{0}$"], fontsize=7)
    ax.tick_params(axis="y", pad=1)
    ax.set_title("Target band energy $\\omega_b$", fontsize=ps.TITLE_FONT,
                 pad=4)
    for k, (val, b) in enumerate(zip(om, BANDS_CF)):
        ytxt = val * 1.8 if b != "A4" else 2.2
        ax.text(k, ytxt, f"{val:.3f}", ha="center", fontsize=6.5)
    ps.panel_label(ax, "h", x=-0.18, y=1.02)

    # (i-l) representative reconstructions — 4 columns, one-line titles
    row = gs2[3].subgridspec(1, 4, wspace=0.26)
    recon_cases = [("A_full", "A: full"), ("D_del_W3", "D: W3 removed"),
                   ("E1_del_W1_only", "E1: W1-associated carriers removed"),
                   ("E2_partial_W3", "E2: partial W3-associated carriers")]
    for i, (cname, lab) in enumerate(recon_cases):
        ax = fig2.add_subplot(row[0, i])
        rr = recs[cname]
        _field_panel(fig2, ax, _recon_image(u, u_j, cname), None,
                     colorbar=(i == 0), show_label=False)
        ax.set_title(f"({chr(ord('i') + i)}) {lab}\n"
                     f"GER {rr['GER']:.3f}, $S_{{\\mathrm{{full}}}}$ {rr['S_full']}",
                     fontsize=ps.TITLE_FONT, pad=5)

    # (m) per-band error for all cases
    ax = fig2.add_subplot(gs2[4])
    case_panels = [("A_full", "A"), ("B_del_W1", "B"), ("C_del_W1W2", "C"),
                   ("D_del_W3", "D"), ("E1_del_W1_only", "E1"),
                   ("E2_partial_W3", "E2")]
    floor = 1e-4
    xpos = np.arange(len(case_panels)) * (len(BANDS_CF) + 1.5)
    for i, (cname, lab) in enumerate(case_panels):
        r = recs[cname]
        errs = [max(r["E_direct"][b], floor) for b in BANDS_CF]
        first_fail = None
        for k, b in enumerate(BANDS_CF):
            if r["E_direct"][b] > ps.TAU:
                first_fail = k
                break
        for k, e in enumerate(errs):
            ax.bar(xpos[i] + k, e, width=0.8,
                   color=ps.BAND_COLORS[BANDS_CF[k]],
                   edgecolor="black", linewidth=0.3)
        if first_fail is not None:
            ax.plot(xpos[i] + first_fail, errs[first_fail] * 1.25, "kv", ms=7,
                    zorder=5)
        ax.text(xpos[i] + len(BANDS_CF) / 2, 0.98, lab, ha="center",
                fontsize=8.5, fontweight="bold", color="#333333")
        ax.text(xpos[i] + len(BANDS_CF) / 2, 0.60,
                f"$S_{{\\mathrm{{full}}}}$={r['S_full']}",
                ha="center", fontsize=7.5)
    ax.axhline(ps.TAU, color=ps.TAU_COLOR, ls="--", lw=1.0)
    ax.text(0.995, 1.02, f"$\\tau$ = {ps.TAU}", transform=ax.transAxes,
            fontsize=8, color=ps.TAU_COLOR, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(floor, 2.0)
    ax.set_xticks([])
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$", fontsize=8)
    handles = [plt.matplotlib.patches.Patch(color=ps.BAND_COLORS[b], label=b)
               for b in BANDS_CF]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              fontsize=7, ncol=5, frameon=False, borderaxespad=0)
    ax.text(0.5, 1.19, "Per-band errors  (▼ = first failed band)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=ps.TITLE_FONT)
    # E1/E2 matched-GER highlight
    bx0 = xpos[4] - 0.9
    bx1 = xpos[5] + len(BANDS_CF) - 0.1
    ax.add_patch(plt.matplotlib.patches.Rectangle(
        (bx0, 2.5e-4), bx1 - bx0, 0.78 - 2.5e-4, fill=False,
        edgecolor=ps.ACCENT, lw=1.6, ls="--", zorder=4))

    ps.save(fig2, OUT_DIR, "fig02_analytical_benchmark_full")
    return 0


if __name__ == "__main__":
    sys.exit(main())

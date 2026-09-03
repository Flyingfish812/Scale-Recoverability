#!/usr/bin/env python3
"""
=============================================================================
P0-1: NC-inspired analytical multiscale benchmark — metric validation
=============================================================================

Purpose
-------
Validate the Scale-Recoverability diagnostics (S_full, S_coh, per-band
direct errors) on a fully analytical, NC-inspired multiscale wake field
where the ground-truth scale content is strictly known.

Controlled cases (each reconstruction removes a known scale):
    A  : full field (reference)
    B  : finest scale removed (W1)              -> expect S_full = 4
    C  : two finest scales removed (W1, W2)     -> expect S_full = 3
    D  : intermediate scale destroyed (W3)      -> expect S_full = 2
    E  : matched-GER pair (E1: W1 removed, E2: partial W3 removed)
         with equal GER but different S_full

Outputs
-------
    artifacts/derived/main/statistics/analytical_benchmark.json
    artifacts/derived/main/statistics/analytical_benchmark.csv
    applications/paper_main/build/figures/fig_analytical_benchmark.pdf/.png

Usage
-----
    conda run -n sana python applications/paper_main/analyses/_canonical/compute_p0_analytical.py
    conda run -n sana python .../compute_p0_analytical.py --no-figure
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.benchmarks.analytical_wake import (
    WakeParams,
    generate_ensemble,
    scale_u_components,
    snapshot,
    velocity,
    wake_envelope,
    base_streamfunction,
    case_metrics,
)
from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE
from luna.pod.band_pod import fit_band_pod
from luna.wavelet.transform import decompose_field_2d

STATS_DIR = _PROJECT_ROOT / "artifacts" / "derived" / "main" / "statistics"
FIG_DIR = _PROJECT_ROOT / "applications" / "paper_main" / "build" / "figures"


# ══════════════════════════════════════════════════════════════════════
# Ensemble statistics
# ══════════════════════════════════════════════════════════════════════
def ensemble_case_stats(
    params: WakeParams,
    n_snapshots: int,
    seed_offset: int,
    band_pod: dict | None = None,
    tau: float = TAU_DEFAULT,
) -> dict:
    """Compute mean/std of all metrics over an ensemble of analytical snapshots."""
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)

    case_names = ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3", "E1_del_W1_only", "E2_partial_W3"]
    acc = {c: {"GER": [], "S_full": [], "S_coh": [], "E_direct": {b: [] for b in BANDS_CF}}
           for c in case_names}

    for i in range(n_snapshots):
        seed = seed_offset + i
        u = snapshot(x, y, params, seed)
        u_j = scale_u_components(x, y, params, seed)
        recs = case_metrics(u, u_j, band_pod=band_pod, tau=tau)
        for c in case_names:
            acc[c]["GER"].append(recs[c]["GER"])
            acc[c]["S_full"].append(recs[c]["S_full"])
            if recs[c]["S_coh"] is not None:
                acc[c]["S_coh"].append(recs[c]["S_coh"])
            for b in BANDS_CF:
                acc[c]["E_direct"][b].append(recs[c]["E_direct"][b])

    out: dict = {}
    for c in case_names:
        d = {
            "GER_mean": float(np.mean(acc[c]["GER"])),
            "GER_std": float(np.std(acc[c]["GER"])),
            "S_full_mean": float(np.mean(acc[c]["S_full"])),
            "S_full_std": float(np.std(acc[c]["S_full"])),
            "S_full_all": [int(v) for v in acc[c]["S_full"]],
            "S_full_correct_frac": float(
                np.mean([v == expected(c) for v in acc[c]["S_full"]])
            ),
            "E_direct_mean": {b: float(np.mean(acc[c]["E_direct"][b])) for b in BANDS_CF},
            "E_direct_std": {b: float(np.std(acc[c]["E_direct"][b])) for b in BANDS_CF},
        }
        if acc[c]["S_coh"]:
            d["S_coh_mean"] = float(np.mean(acc[c]["S_coh"]))
            d["S_coh_std"] = float(np.std(acc[c]["S_coh"]))
        out[c] = d
    return out


def expected(case: str) -> int:
    return {"A_full": 5, "B_del_W1": 4, "C_del_W1W2": 3, "D_del_W3": 2,
            "E1_del_W1_only": 4, "E2_partial_W3": 2}[case]


# ══════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════
def make_figure(params: WakeParams, tau: float, seed: int = 0) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    u = snapshot(x, y, params, seed)
    u_j = scale_u_components(x, y, params, seed)
    v = velocity(x, y, params, seed)[1]
    recs = case_metrics(u, u_j, tau=tau)
    bd = decompose_field_2d(u)
    tot = np.sum(u ** 2)
    omega = {b: float(np.sum(bd[b] ** 2) / tot) for b in BANDS_CF}

    case_panels = [
        ("A_full", "A"), ("B_del_W1", "B"), ("C_del_W1W2", "C"),
        ("D_del_W3", "D"), ("E1_del_W1_only", "E1"), ("E2_partial_W3", "E2"),
    ]

    def recon_image(cname: str) -> np.ndarray:
        if cname == "A_full":
            return u.copy()
        if cname in ("B_del_W1", "E1_del_W1_only"):
            return u - u_j[5]
        if cname == "C_del_W1W2":
            return u - u_j[5] - u_j[4]
        if cname == "D_del_W3":
            return u - u_j[3]
        if cname == "E2_partial_W3":
            alpha = _alpha_for_ger(u, u_j, recs["E1_del_W1_only"]["GER"])
            return u - alpha * u_j[3]
        raise KeyError(cname)

    def field_panel(ax, arr, ttl, vmax=None):
        if vmax is None:
            vmax = np.abs(arr).max()
        im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    fig = plt.figure(figsize=(15, 13.2))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.05, 0.85, 0.85, 0.85, 1.15],
                          hspace=0.55)

    # ── Part 1a: target fields ───────────────────────────────────
    row = gs[0].subgridspec(1, 2, wspace=0.18)
    vm = np.abs(u).max()
    im = field_panel(fig.add_subplot(row[0, 0]), u, "(a) Target $u$", vmax=vm)
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046)
    im = field_panel(fig.add_subplot(row[0, 1]), v, "(b) $v=-\\partial_x\\psi$", vmax=np.abs(v).max())
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046)

    # ── Part 1b: scale components (coarse: A4, W4, W3) ───────────
    row = gs[1].subgridspec(1, 3, wspace=0.18)
    for i, b in enumerate(["A4", "W4", "W3"]):
        comp = bd[b]
        ax = fig.add_subplot(row[0, i])
        im = field_panel(ax, comp, f"({chr(ord('c') + i)}) Band {b}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Part 1c: scale components (fine: W2, W1) + energy ω_b ────
    row = gs[2].subgridspec(1, 3, wspace=0.22, width_ratios=[1, 1, 1.05])
    for i, b in enumerate(["W2", "W1"]):
        comp = bd[b]
        ax = fig.add_subplot(row[0, i])
        im = field_panel(ax, comp, f"({chr(ord('f') + i)}) Band {b}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(row[0, 2])
    vals = [omega[b] for b in BANDS_CF]
    ax.bar(np.arange(5), np.maximum(vals, 1e-5), color="#4C72B0",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(BANDS_CF, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2)
    ax.set_ylabel(r"$\omega_b$", fontsize=10)
    ax.set_title("(h) Target band energy $\\omega_b$ (log)", fontsize=10)
    for k, val in enumerate(vals):
        ax.text(k, val * 1.6, f"{val:.3f}", ha="center", fontsize=7)

    # ── Part 2: representative reconstructions (≤4 per row) ──────
    row = gs[3].subgridspec(1, 4, wspace=0.18)
    recon_cases = [
        ("A_full", "A: full"),
        ("D_del_W3", "D: W3 removed"),
        ("E1_del_W1_only", "E1: W1 removed"),
        ("E2_partial_W3", "E2: W3 part."),
    ]
    for i, (cname, label) in enumerate(recon_cases):
        ax = fig.add_subplot(row[0, i])
        rec = recs[cname]
        im = field_panel(ax, recon_image(cname), "", vmax=vm)
        ax.set_title(f"({chr(ord('i') + i)}) {label}\nGER={rec['GER']:.3f},  "
                     f"$S_{{\\mathrm{{full}}}}$={rec['S_full']}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Part 3: per-band error chart (log y) ─────────────────────
    ax = fig.add_subplot(gs[4])
    colors = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52", 3: "#8172B3", 4: "#CCB974"}
    floor = 1e-4
    xpos = np.arange(len(case_panels)) * (len(BANDS_CF) + 1.5)
    for i, (cname, label) in enumerate(case_panels):
        rec = recs[cname]
        errs = [max(rec["E_direct"][b], floor) for b in BANDS_CF]
        first_fail = None
        for k, b in enumerate(BANDS_CF):
            if rec["E_direct"][b] > tau:
                first_fail = k
                break
        for k, e in enumerate(errs):
            ax.bar(xpos[i] + k, e, width=0.8, color=colors[k], alpha=0.9,
                   edgecolor="black", linewidth=0.4)
        if first_fail is not None:
            ax.plot(xpos[i] + first_fail, errs[first_fail], "kv", markersize=8, zorder=5)
        ax.text(xpos[i] + len(BANDS_CF) / 2, 4e-4, f"$S_{{full}}$={rec['S_full']}",
                ha="center", fontsize=9)
        ax.text(xpos[i] + len(BANDS_CF) / 2, 1.5, label, ha="center",
                fontsize=10, fontweight="bold")
    ax.axhline(tau, color="red", linestyle="--", linewidth=1.2)
    ax.text(0.3, tau, "  $\\tau$", color="red", fontsize=10, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(floor, 2.0)
    ax.set_xticks([])
    ax.set_ylabel(r"Per-band direct error $E_{\mathrm{direct}}(b)$", fontsize=11)
    ax.set_title("(m) Per-band errors (log): pass $<\\tau$, fail $>\\tau$; "
                 "$\\blacktriangledown$ = first failed band", fontsize=10)
    handles = [mpatches.Patch(color=colors[k], label=b) for k, b in enumerate(BANDS_CF)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=5, framealpha=0.9)
    # E1/E2 matched-GER annotation (the headline)
    ax.annotate("", xy=(xpos[4] + 2.5, 1.05), xytext=(xpos[5] + 2.5, 1.05),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0))
    ax.text((xpos[4] + xpos[5]) / 2 + 2.5, 1.12,
            "E1/E2: equal GER\nbut different $S_{full}$", ha="center",
            fontsize=8, color="0.25")
    ax.text(0.3, 1.8, "bars at $10^{-4}$ = exactly zero (log floor)",
            fontsize=7, color="0.4")

    out = FIG_DIR / "fig_analytical_benchmark"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(description="P0-1 analytical multiscale benchmark")
    parser.add_argument("--n-train", type=int, default=200, help="snapshots for band-POD fit")
    parser.add_argument("--n-test", type=int, default=100, help="snapshots for stats")
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    params = WakeParams()

    t0 = time.time()
    # 1) band-POD for S_coh
    print("[1/4] fitting band-POD on analytical training ensemble ...")
    train_fields, (x, y) = generate_ensemble(args.n_train, params, seed_offset=0)
    band_pod = fit_band_pod(train_fields, pod_energy_threshold=0.99)

    # 2) ensemble statistics
    print(f"[2/4] computing case metrics over {args.n_test} snapshots ...")
    stats = ensemble_case_stats(params, args.n_test, args.seed_offset, band_pod, args.tau)

    # 3) representative snapshot (for figure + detailed record)
    print("[3/4] building representative snapshot (seed=0) ...")
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    u = snapshot(x, y, params, 0)
    u_j = scale_u_components(x, y, params, 0)
    rec_rep = case_metrics(u, u_j, band_pod=band_pod, tau=args.tau)

    # band energy fractions of the representative target
    bd = decompose_field_2d(u)
    tot = np.sum(u ** 2)
    band_fracs = {b: float(np.sum(bd[b] ** 2) / tot) for b in BANDS_CF}

    # 4) save
    print("[4/4] saving results ...")
    payload = {
        "benchmark": "P0-1 NC-inspired analytical multiscale wake",
        "grid": {"H": params.H, "W": params.W, "x0": params.x0, "y0": params.y0},
        "wavelet": {"family": DEFAULT_WAVELET, "level": DEFAULT_LEVEL, "mode": DEFAULT_MODE},
        "tau": args.tau,
        "params": params.__dict__,
        "wavenum": {str(k): v for k, v in {1: [(1, 1), (2, 1), (3, 2)], 2: [(6, 4), (7, 4)],
                                            3: [(14, 7)], 4: [(28, 14), (24, 12)],
                                            5: [(48, 32), (56, 28)]}.items()},
        "n_train_pod": args.n_train,
        "n_test": args.n_test,
        "target_band_energy_fractions": band_fracs,
        "representative_seed0": rec_rep,
        "ensemble_stats": stats,
        "elapsed_s": time.time() - t0,
    }
    json_path = STATS_DIR / "analytical_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # CSV (flat summary)
    import csv
    csv_path = STATS_DIR / "analytical_benchmark.csv"
    rows = []
    for cname, d in stats.items():
        row = {"case": cname, "expected_S_full": expected(cname)}
        row.update({f"GER_mean": d["GER_mean"], "GER_std": d["GER_std"],
                    "S_full_mean": d["S_full_mean"], "S_full_std": d["S_full_std"],
                    "S_full_correct_frac": d["S_full_correct_frac"]})
        for b in BANDS_CF:
            row[f"E_{b}_mean"] = d["E_direct_mean"][b]
            row[f"E_{b}_std"] = d["E_direct_std"][b]
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # figure
    fig_path = None
    if not args.no_figure:
        print("[+] rendering main figure ...")
        fig_path = make_figure(params, args.tau)

    # ── console summary ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  P0-1 ANALYTICAL MULTISCALE BENCHMARK — SUMMARY")
    print("=" * 78)
    print(f"  target band energy fractions: "
          + "  ".join(f"{b}={band_fracs[b]:.4f}" for b in BANDS_CF))
    print(f"  (real NC reference        :   A4=0.966  W4=0.029  W3=0.004  "
          f"W2=0.001  W1=0.000)")
    print("-" * 78)
    hdr = f"  {'case':<16}{'expected':>9}{'GER(mean)':>11}{'S_full':>9}{'correct':>9}"
    print(hdr)
    print("  " + "-" * 56)
    for cname in ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3", "E1_del_W1_only", "E2_partial_W3"]:
        d = stats[cname]
        print(f"  {cname:<16}{expected(cname):>9}{d['GER_mean']:>11.4f}"
              f"{d['S_full_mean']:>9.2f}{d['S_full_correct_frac']:>9.2f}")
    print("=" * 78)
    print(f"  results : {json_path}")
    print(f"           {csv_path}")
    if fig_path:
        print(f"  figure  : {fig_path}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

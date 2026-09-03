#!/usr/bin/env python3
"""
=============================================================================
P0-2: Wavelet-family sensitivity audit
=============================================================================

Purpose
-------
Answer Jean-Philippe's question: if we replace the default db2 wavelet with
another reasonable family, do the scale-recoverability conclusions change
fundamentally?

Strictly limited scope (per work-list P0-2):
  * ONE representative real-NC configuration: M=30, sigma=0
    (the paper's canonical Type-A counterexample).
  * Models compared at this configuration: Ridge, MLP (seed 0), VCNN (seed 0).
  * Wavelets compared: haar, db2, db4, sym4, coif1.
  * Fixed: decomposition level 4, periodization, tau=0.05, GT + reconstructions
    (no retraining, no M/sigma/model grid).
  * Plus: re-runs the P0-1 analytical benchmark controlled cases under each
    wavelet (checks whether the analytical validation itself is db2-specific).

Outputs
-------
    artifacts/derived/main/statistics/wavelet_sensitivity.json
    artifacts/derived/main/statistics/wavelet_sensitivity.csv
    applications/paper_main/build/figures/fig_wavelet_sensitivity.{pdf,png}

Usage
-----
    conda run -n sana python applications/paper_main/analyses/_canonical/compute_p0_2_wavelet_sensitivity.py
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.benchmarks.analytical_wake import (
    WakeParams, snapshot, scale_u_components, controlled_reconstructions,
    case_metrics,
)
from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_LEVEL, DEFAULT_MODE
from luna.pod.band_pod import fit_band_pod
from luna.wavelet.metrics import band_errors_all, compute_S_full, compute_S_coh, rel_l2

STATS_DIR = _PROJECT_ROOT / "artifacts" / "derived" / "main" / "statistics"
FIG_DIR = _PROJECT_ROOT / "applications" / "paper_main" / "build" / "figures"

WAVELETS = ["haar", "db2", "db4", "sym4", "coif1"]

NC_PATHS = {
    "ridge": _PROJECT_ROOT / "artifacts/ridge_closed_form_sweep_nc/ridge_n0030/seed000/tests/s0000/test_raw.npz",
    "mlp": _PROJECT_ROOT / "artifacts/pod_model_sweep_nc/mlp_n0030/seed000/tests/s0000/test_raw.npz",
    "vcnn": _PROJECT_ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n0030_seed000_custom/tests/s0000/test_raw.npz",
}
NC_DATA = _PROJECT_ROOT / "data" / "cylinder2d_q1.npy"
N_TRAIN_POD = 400


# ══════════════════════════════════════════════════════════════════════
# Real-NC audit
# ══════════════════════════════════════════════════════════════════════
def audit_real_nc(wavelet: str, level: int, mode: str, tau: float) -> dict:
    """Compute per-model metrics for one wavelet at M=30, sigma=0."""
    # band-POD on a training subset (excludes test indices), per wavelet
    ref = np.load(NC_PATHS["mlp"])
    test_indices = sorted(set(ref["test_indices"].tolist()))
    all_fields = np.load(NC_DATA)
    train_idx = sorted(set(range(all_fields.shape[0])) - set(test_indices))
    rng = np.random.RandomState(7)
    sub = sorted(rng.choice(train_idx, min(N_TRAIN_POD, len(train_idx)), replace=False))
    train_u = all_fields[sub][:, :, :, 0].astype(np.float64)
    band_pod = fit_band_pod(train_u, pod_energy_threshold=0.99,
                            wavelet=wavelet, level=level, mode=mode)

    out: dict[str, dict] = {}
    for model, path in NC_PATHS.items():
        d = np.load(path)
        u = d["target_nchw"][:, 0, :, :].astype(np.float64)   # (300, H, W)
        uh = d["output_nchw"][:, 0, :, :].astype(np.float64)
        n = u.shape[0]
        gers, sfs, scohs = [], [], []
        errs = {b: [] for b in BANDS_CF}
        for i in range(n):
            gers.append(rel_l2(uh[i], u[i]))
            e = band_errors_all(u[i], uh[i], wavelet, level, mode)
            for b in BANDS_CF:
                errs[b].append(e[b])
            sfs.append(compute_S_full(u[i], uh[i], tau, wavelet, level, mode))
            scohs.append(compute_S_coh(u[i], uh[i], band_pod, tau, wavelet, level, mode))
        out[model] = {
            "GER_mean": float(np.mean(gers)),
            "GER_std": float(np.std(gers)),
            "S_full_mean": float(np.mean(sfs)),
            "S_full_std": float(np.std(sfs)),
            "S_full_mode": int(np.bincount(sfs).argmax()),
            "S_full_dist": {str(k): int(v) for k, v in zip(*np.unique(sfs, return_counts=True))},
            "S_coh_mean": float(np.mean(scohs)),
            "S_coh_mode": int(np.bincount(scohs).argmax()),
            "E_direct_mean": {b: float(np.mean(errs[b])) for b in BANDS_CF},
            "E_direct_std": {b: float(np.std(errs[b])) for b in BANDS_CF},
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# Analytical-benchmark audit (is P0-1 validation db2-specific?)
# ══════════════════════════════════════════════════════════════════════
def audit_analytical(wavelet: str, level: int, mode: str, tau: float,
                     n_snapshots: int = 30) -> dict:
    params = WakeParams()
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    case_names = ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3", "E1_del_W1_only", "E2_partial_W3"]
    expected = {"A_full": 5, "B_del_W1": 4, "C_del_W1W2": 3, "D_del_W3": 2,
                "E1_del_W1_only": 4, "E2_partial_W3": 2}
    acc = {c: [] for c in case_names}
    for seed in range(n_snapshots):
        u = snapshot(x, y, params, seed)
        u_j = scale_u_components(x, y, params, seed)
        recs = case_metrics(u, u_j, tau=tau, wavelet=wavelet, level=level, mode=mode)
        for c in case_names:
            acc[c].append(recs[c]["S_full"])
    out = {}
    for c in case_names:
        arr = np.array(acc[c], dtype=int)
        out[c] = {
            "expected": expected[c],
            "S_full_mode": int(np.bincount(arr).argmax()),
            "S_full_mean": float(arr.mean()),
            "correct_frac": float(np.mean(arr == expected[c])),
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════
def make_figure(nc_results: dict, ana_results: dict, tau: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    models = ["ridge", "mlp", "vcnn"]
    model_labels = {"ridge": "Ridge", "mlp": "MLP", "vcnn": "VCNN"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ── (a) S_full by model x wavelet ─────────────────────────────
    ax = axes[0]
    xpos = np.arange(len(WAVELETS))
    width = 0.26
    colors = {"ridge": "#C44E52", "mlp": "#55A868", "vcnn": "#4C72B0"}
    for mi, m in enumerate(models):
        means = [nc_results[w][m]["S_full_mean"] for w in WAVELETS]
        stds = [nc_results[w][m]["S_full_std"] for w in WAVELETS]
        ax.bar(xpos + (mi - 1) * width, means, width, yerr=stds, capsize=3,
               color=colors[m], label=model_labels[m], alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(WAVELETS)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel(r"$\bar{S}_{\mathrm{full}}$")
    ax.set_title("(a) Real NC, $M=30$, $\\sigma=0$:\nmodel ranking by $S_{\\mathrm{full}}$")
    ax.legend(fontsize=8, loc="lower left")
    ax.axhline(4, color="gray", linestyle=":", linewidth=0.8)

    # ── (b) per-band hierarchy for Ridge across wavelets (log y) ─
    ax = axes[1]
    xpos = np.arange(len(BANDS_CF))
    for wi, w in enumerate(WAVELETS):
        means = [nc_results[w]["ridge"]["E_direct_mean"][b] for b in BANDS_CF]
        ax.plot(xpos + wi * 0.13, means, "o-", markersize=4, label=w, alpha=0.85)
    ax.axhline(tau, color="red", linestyle="--", linewidth=1.2)
    ax.text(0.05, tau * 1.35, f"$\\tau={tau}$", color="red", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(BANDS_CF)
    ax.set_ylabel(r"$E_{\mathrm{direct}}(b)$ (log)")
    ax.set_title("(b) Ridge per-band errors (hierarchy)\nacross wavelets (log)")
    ax.legend(fontsize=7, loc="lower left")

    # ── (c) analytical benchmark detection: multi-line trends ────
    ax = axes[2]
    cases = ["A_full", "B_del_W1", "C_del_W1W2", "D_del_W3", "E1_del_W1_only", "E2_partial_W3"]
    case_short = ["A", "B", "C", "D", "E1", "E2"]
    line_styles = [("-o", "#4C72B0"), ("-s", "#55A868"), ("-^", "#C44E52"),
                   ("-D", "#8172B3"), ("-v", "#CCB974"), ("-P", "#64B5CD")]
    xw = np.arange(len(WAVELETS))
    for ci, c in enumerate(cases):
        fracs = [ana_results[w][c]["correct_frac"] for w in WAVELETS]
        ax.plot(xw, fracs, line_styles[ci][0], color=line_styles[ci][1],
                markersize=4, label=f"{case_short[ci]}", alpha=0.9)
    ax.set_xticks(xw)
    ax.set_xticklabels(WAVELETS)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("correct fraction")
    ax.set_title("(c) Analytical benchmark:\ncorrect $S_{\\mathrm{full}}$ fraction per case")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    plt.tight_layout()
    out = FIG_DIR / "fig_wavelet_sensitivity"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
MODELS = ["ridge", "mlp", "vcnn"]


def _print_summary(nc_results: dict, ana_results: dict) -> None:
    hdr = f"  {'wavelet':<8}" + "".join(f"{m:>12}" for m in MODELS)
    print("  S_full (mode) " + hdr)
    for wt in WAVELETS:
        row = f"  {wt:<14}"
        for m in MODELS:
            row += f"{nc_results[wt][m]['S_full_mode']:>12}"
        print(row)
    print("-" * 92)
    print("  S_full (mean) " + hdr)
    for wt in WAVELETS:
        row = f"  {wt:<14}"
        for m in MODELS:
            row += f"{nc_results[wt][m]['S_full_mean']:>12.2f}"
        print(row)
    print("-" * 92)
    print("  S_coh (mode)  " + hdr)
    for wt in WAVELETS:
        row = f"  {wt:<14}"
        for m in MODELS:
            row += f"{nc_results[wt][m]['S_coh_mode']:>12}"
        print(row)
    print("-" * 92)
    print("  Analytical benchmark correct-fraction per case (A,B,C,D,E1,E2):")
    for wt in WAVELETS:
        row = f"  {wt:<8} " + "  ".join(f"{ana_results[wt][c]['correct_frac']:.2f}"
                                         for c in ["A_full", "B_del_W1", "C_del_W1W2",
                                                   "D_del_W3", "E1_del_W1_only", "E2_partial_W3"])
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="P0-2 wavelet-family sensitivity audit")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="re-print summary from saved JSON without recomputing")
    parser.add_argument("--figure-only", action="store_true",
                        help="re-render figure from saved JSON without recomputing")
    args = parser.parse_args()

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    json_path = STATS_DIR / "wavelet_sensitivity.json"
    if args.figure_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        fig_path = make_figure(payload["real_nc"], payload["analytical_benchmark"], args.tau)
        print(f"[figure-only] rendered {fig_path}.pdf (+ .png)")
        return 0
    if args.summary_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        print("\n" + "=" * 92)
        print("  P0-2 WAVELET-FAMILY SENSITIVITY — SUMMARY (M=30, sigma=0)")
        print("=" * 92)
        _print_summary(payload["real_nc"], payload["analytical_benchmark"])
        print("=" * 92)
        print(f"  results : {json_path}")
        return 0

    t0 = time.time()
    nc_results = {}
    ana_results = {}
    for w in WAVELETS:
        print(f"[wavelet {w}] real-NC audit ...", flush=True)
        nc_results[w] = audit_real_nc(w, args.level, args.mode, args.tau)
        print(f"[wavelet {w}] analytical-benchmark audit ...", flush=True)
        ana_results[w] = audit_analytical(w, args.level, args.mode, args.tau)

    # ── save ────────────────────────────────────────────────────
    payload = {
        "audit": "P0-2 wavelet-family sensitivity",
        "representative_case": {"dataset": "nc", "M": 30, "sigma": 0.0,
                                "models": MODELS},
        "wavelets": WAVELETS,
        "fixed": {"level": args.level, "mode": args.mode, "tau": args.tau},
        "real_nc": nc_results,
        "analytical_benchmark": ana_results,
        "elapsed_s": time.time() - t0,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    csv_path = STATS_DIR / "wavelet_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "wavelet", "model_or_case", "S_full_mean", "S_full_mode",
                    "S_coh_mean", "GER_mean"] + [f"E_{b}" for b in BANDS_CF])
        for wt, res in nc_results.items():
            for m, d in res.items():
                w.writerow(["nc", wt, m, round(d["S_full_mean"], 3), d["S_full_mode"],
                            round(d["S_coh_mean"], 3), round(d["GER_mean"], 4)]
                           + [round(d["E_direct_mean"][b], 4) for b in BANDS_CF])
        for wt, res in ana_results.items():
            for c, d in res.items():
                w.writerow(["analytical", wt, c, round(d["S_full_mean"], 3), d["S_full_mode"],
                            "NA", "NA"] + ["NA"] * len(BANDS_CF))

    fig_path = None
    if not args.no_figure:
        print("[+] rendering figure ...")
        fig_path = make_figure(nc_results, ana_results, args.tau)

    # ── summary ──────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("  P0-2 WAVELET-FAMILY SENSITIVITY — SUMMARY (M=30, sigma=0)")
    print("=" * 92)
    _print_summary(nc_results, ana_results)
    print("=" * 92)
    print(f"  results : {json_path}")
    print(f"           {csv_path}")
    if fig_path:
        print(f"  figure  : {fig_path}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

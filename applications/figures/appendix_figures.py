#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
appendix_figures.py — S-Fig 1-8 appendix figures, redrawn (v3-3, 2026-08-31)

All output is vector PDF, with consistent naming:
  figS01_oracle_rdb.pdf / figS01_oracle_sst.pdf   (S1: oracle audit)
  figS02_ridge_phase.pdf / figS02_vcnn_phase.pdf  (S2: phase diagrams)
  figS03_noise_propagation.pdf / figS03_level_sensitivity.pdf /
  figS03_coherent_only_sample.pdf                 (S3: supporting diagnostics)
  figS04_three_layer.pdf                          (S4: three-layer error)
  figS05_mode_scale_energy.pdf                    (S5: mode-scale energy)
  figS06_tau_sensitivity.pdf                      (S6: threshold sensitivity)
  figS07_sensor_family_ger.pdf                    (S7: sensor family GER)
  figS08_sensor_family_paired.pdf                 (S8: MLP-VCNN pairing)

Data sources are identical to the canonical versions (verified).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import records
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
SUPP = _ROOT / "artifacts" / "statistics"

BANDS = ps.BANDS
TAU = ps.TAU


# ══════════════════════════════════════════════════════════════════
# S1: Oracle audits (RDB / SST)
# ══════════════════════════════════════════════════════════════════
def s1_oracle():
    d = json.loads((SUPP / "truncation_reference_audit.json").read_text())
    cfg = {
        "rdb_h5": ("RDB (radial dam-break)", [16, 32, 64, 128], "figS01_oracle_rdb"),
        "sst_weekly": ("SST (sea surface temperature)", [32, 64, 128, 256, 512, 1024],
                       "figS01_oracle_sst"),
    }
    summaries = {s["dataset"]: s for s in d["summaries"]}
    for ds, (label, ranks, stem) in cfg.items():
        sm = summaries.get(ds)
        if sm is None:
            continue
        table = sm.get("table", {})
        fig, ax = ps.figure(6.6, 3.4, 1, 1)
        for b in BANDS:
            xs, ys = [], []
            for r in ranks:
                e = table.get(str(r), {}).get("bands", {}).get(b, {}).get("mean")
                if e is not None:
                    xs.append(r)
                    ys.append(float(e))
            if xs:
                ax.plot(xs, ys, marker=ps.BAND_MARKERS[b], color=ps.BAND_COLORS[b],
                        label=b, lw=1.3, ms=4.5)
        ax.axhline(TAU, color="0.45", ls="--", lw=0.9,
                   label=f"$\\tau$={TAU} (mean criterion)")
        ax.set_xlabel("POD rank $r$")
        ax.set_ylabel("Mean band truncation error")
        ax.set_yscale("log")
        ax.legend(ncol=3, fontsize=7)
        ps.save(fig, OUT_DIR, stem)
        print(f"  [S1] {stem} done")


# ══════════════════════════════════════════════════════════════════
# S2: Ridge / VCNN phase diagrams
# ══════════════════════════════════════════════════════════════════
def s2_phase():
    M = [10, 15, 20, 30, 50]
    sigmas = [0.0, 0.001, 0.01, 0.1]
    # Ridge (closed form), read from the band-error records
    ridge_series = {m: {s: records.config_summary("ridge", m, s)["S_full_mean"]
                        for s in sigmas}
                    for m in M}

    # VCNN (sensor-noise phase, physical units)
    phase = json.loads((_ROOT / "artifacts" / "statistics" / "sensor_noise_phase.json").read_text())
    vcnn = phase["phase_summary"]["vcnn"]
    vcnn_series = {}
    for m in M:
        vcnn_series[m] = {
            s: float(vcnn.get(str(m), {}).get(str(s), {}).get("mean_S_full", 0.0))
            for s in sigmas
        }

    for series, stem, name in [
        (ridge_series, "figS02_ridge_phase", "Ridge (closed-form)"),
        (vcnn_series, "figS02_vcnn_phase", "VCNN"),
    ]:
        fig, ax = ps.figure(6.6, 3.0, 1, 1)
        sig_lab = ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
        for si, s in enumerate(sigmas):
            vals = [series.get(m, {}).get(s) for m in M]
            if any(v is None for v in vals):
                continue
            ax.plot(M, vals, color=ps.SIGMA_COLORS[s], marker=ps.SIGMA_MARKERS[s],
                    lw=1.4, ms=5, label=f"$\\sigma$={sig_lab[si]}")
        ax.set_xlabel("Sensor count $M$")
        ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
        ax.set_xticks(M)
        ax.set_ylim(-0.2, 5.2)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ps.save(fig, OUT_DIR, stem)
        print(f"  [S2] {stem} done")
    return 0


# ══════════════════════════════════════════════════════════════════
# S3: noise propagation / level sensitivity / coherent-only
# ══════════════════════════════════════════════════════════════════
def s3_diagnostics():
    # (a) noise propagation
    d = json.loads((_ROOT / "artifacts" / "statistics" / "noise_propagation.json").read_text())
    per_config = d["per_configuration"]
    model_order = ["mlp", "vcnn", "ridge"]
    fig, ax = ps.figure(3.4, 3.0, 1, 1)  # standalone small figure
    x = np.arange(len(BANDS))
    w = 0.25
    for i, mt in enumerate(model_order):
        ratios = [v["degradation_ratio"] for k, v in per_config.items()
                  if k.split("_")[0] == mt]
        if not ratios:
            continue
        means = [float(np.mean([r[b] for r in ratios])) for b in BANDS]
        ax.bar(x + (i - 1) * w, means, w,
               color=[ps.MODEL_COLORS["MLP"], ps.MODEL_COLORS["VCNN"],
                      ps.MODEL_COLORS["Ridge"]][i], alpha=0.9,
               label=mt.upper() if mt != "mlp" else "MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS, fontsize=7.5)
    ax.set_ylabel("Degradation ratio (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5)
    ps.save(fig, OUT_DIR, "figS03_noise_propagation")

    # (b) level sensitivity — S_full/(L+1) and per-band error (spatial, as in the paper)
    level_data = json.loads((_ROOT / "artifacts" / "statistics" / "level_sensitivity.json").read_text())["levels"]
    level_keys = sorted(level_data.keys(), key=lambda k: int(k.split("_")[-1]))
    if not level_keys:
        print(f"  [warn] level_sensitivity structure: {list(level_data.keys())[:8]}")
        return None
    Ls = [int(k.split("_")[-1]) for k in level_keys]

    def level_view(k):
        entry = level_data[k]
        return {
            "s_full_mean": entry["spatial"]["s_full_mean"],
            "per_band_mean_error": entry["spatial"]["per_band_mean_error"],
            "bands": entry["bands"],
        }
    d = {k: level_view(k) for k in level_keys}

    # main figure: normalised S_full/(L+1) + per-band mean error (two panels)
    fig, axes = ps.figure(6.6, 3.4, 1, 2)
    ax = axes[0]
    vals = []
    for k in level_keys:
        v = d[k]
        if isinstance(v, dict):
            sf = v.get("s_full_mean", v.get("S_full_mean", v.get("mean_S_full")))
            vals.append(float(sf) if sf is not None else np.nan)
        else:
            vals.append(float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v))
    ax.plot(Ls, [v / (L + 1) for v, L in zip(vals, Ls)],
            "o-", color=ps.MODEL_COLORS["MLP"], lw=1.4, ms=5)
    ax.set_xticks(Ls)
    ax.set_xlabel("Decomposition level $L$")
    ax.set_ylabel("$S_{\\mathrm{full}}/(L+1)$")
    ax.set_ylim(0.85, 1.0)

    # per-band mean error of each level (level 3: 4 bands, level 4: 5, level 5: 6)
    ax = axes[1]
    for k, L, color in zip(level_keys, Ls,
                           [ps.MODEL_COLORS["MLP"], ps.MODEL_COLORS["VCNN"],
                            ps.MODEL_COLORS["Ridge"]]):
        v = d[k]
        err = v.get("per_band_mean_error", {})
        bands = v.get("bands", [])
        if isinstance(err, dict) and bands:
            xs = np.arange(len(bands))
            ax.plot(xs, [err.get(b, np.nan) for b in bands], "o-",
                    lw=1.3, ms=4.5, color=color, label=f"L={L}")
    ax.axhline(ps.TAU, color=ps.TAU_COLOR, ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(d[level_keys[-1]]["bands"])))
    ax.set_xticklabels(d[level_keys[-1]]["bands"], fontsize=7.5)
    ax.set_xlim(-0.3, len(d[level_keys[-1]]["bands"]) - 0.7)
    ax.set_xlabel("wavelet bands (coarse → fine)")
    ax.set_ylabel("Mean band error $E_{\\mathrm{direct}}(b)$")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    ps.save(fig, OUT_DIR, "figS03_level_sensitivity")
    print(f"  [S3b] levels={Ls} S_norm={[round(v/(L+1),3) for v,L in zip(vals,Ls)]}")

    # (c) coherent-only sample is drawn by figS03c_coherent_only.py
    ps.close(fig) if fig else None
    return 0


# ══════════════════════════════════════════════════════════════════
# S4: three-layer error
# ══════════════════════════════════════════════════════════════════
def s4_three_layer():
    d = records.band_records()
    if isinstance(d, dict) and "compensation_effect" in d:
        comp = d["compensation_effect"]
    else:
        from collections import defaultdict
        groups = defaultdict(lambda: defaultdict(list))
        for r in d:
            if r["model_type"] == "vcnn" and r["mask_num"] == 10 and r["noise_sigma"] == 0.0:
                for b in BANDS:
                    groups[b]["total"].append(r.get(f"E_total_{b}", 0))
                    groups[b]["trunc"].append(r.get(f"E_trunc_{b}", 0))
                    groups[b]["pred"].append(r.get(f"E_pred_{b}", 0))
        comp = {"n_samples": len(next(iter(groups.values()))["total"]) if groups else 0}
        for b in BANDS:
            comp[b] = {f"E_{k}_mean": float(np.mean(groups[b][k])) for k in
                       ["total", "trunc", "pred"]}
    fig, ax = ps.figure(6.6, 3.2, 1, 1)
    x = np.arange(len(BANDS))
    w = 0.25
    keys = [("E_total_mean", "$E_{\\mathrm{total}}$", ps.MODEL_COLORS["MLP"]),
            ("E_trunc_mean", "$E_{\\mathrm{trunc}}$ (POD trunc.)", ps.MODEL_COLORS["Oracle"]),
            ("E_pred_mean", "$E_{\\mathrm{pred}}$ (model)", ps.MODEL_COLORS["Ridge"])]
    for off, (k, lab, c) in enumerate(keys):
        vals = [max(comp[b].get(k, 1e-6), 1e-6) for b in BANDS]
        ax.bar(x + (off - 1) * w, vals, w, color=c, alpha=0.9, label=lab)
    ax.axhline(TAU, color="0.45", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("Relative L2 error (log)")
    ax.set_xlabel("Wavelet band")
    ax.legend(fontsize=7)
    ps.save(fig, OUT_DIR, "figS04_three_layer")
    print(f"  [S4] three-layer saved")


# ══════════════════════════════════════════════════════════════════
# S5: mode-to-scale energy
# ══════════════════════════════════════════════════════════════════
def s5_mode_scale():
    import matplotlib.pyplot as plt  # noqa: F401
    d = json.loads((_ROOT / "artifacts" / "statistics" / "mode_scale_energy.json").read_text())
    cc = d["cumulative_coverage"]              # {band: [C_b(r) for r in 1..128]}
    bw = d["band_energy_per_mode"]             # {band: [per-mode band energy]}
    mer = d["mode_energy_ratio"]               # [lambda_j/lambda_1]
    tm = d["threshold_modes"]                  # {band: {"0.9": n_modes}}

    fig = plt.figure(figsize=(6.6, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.58,
                          wspace=0.44, left=0.07, right=0.985,
                          top=0.94, bottom=0.08)

    # (a) cumulative coverage (full width, top)
    ax = fig.add_subplot(gs[0, :])
    for b in BANDS:
        arr = np.asarray(cc[b], dtype=float)
        ax.plot(np.arange(1, len(arr) + 1), arr, color=ps.BAND_COLORS[b],
                lw=1.3, label=b)
    ax.axhline(0.9, color="0.45", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("Cumulative band energy $C_b(r)$")
    ax.legend(fontsize=7, ncol=3, loc="lower right", frameon=True)
    ps.panel_label(ax, "a", x=-0.12, y=1.04)

    # (b) per-mode band energy fractions (bottom left) — multi-line (smoothed)
    # band_energy_weighted[b][j] = λ_j ‖W_b(φ_j)‖² is a weighted energy with
    # arbitrary units, so it is not plotted directly. The plotted quantity is
    # the share of the mode φ_r energy carried by band b:
    #   frac_b(r) = E_b(φ_r) / Σ_b' E_b'(φ_r) ∈ [0,1], summing to 1 per mode.
    # Those per-mode fractions oscillate strongly in r, so the plotted curves
    # are Gaussian-smoothed trends, which make the band shares readable.
    from scipy.ndimage import gaussian_filter1d
    ax = fig.add_subplot(gs[1, 0])
    Eb = np.stack([np.asarray(bw[b], dtype=float) for b in BANDS])   # (5, 128)
    frac = Eb / Eb.sum(axis=0, keepdims=True)                        # (5, 128)
    rr = np.arange(1, frac.shape[1] + 1)
    for i, b in enumerate(BANDS):
        ax.plot(rr, gaussian_filter1d(frac[i], sigma=4.0, mode="nearest"),
                color=ps.BAND_COLORS[b], lw=1.6, label=b)
    ax.set_xscale("log")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("Fraction of mode energy in band\n"
                  r"$E_b(\phi_r)/E(\phi_r)$  (0$-$1)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ps.panel_label(ax, "b", x=-0.18, y=1.04)

    # (c) log POD spectrum (bottom right)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(1, len(mer) + 1), mer, color="0.25", lw=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 128)
    ax.set_xlabel("POD mode $r$ (log)")
    ax.set_ylabel("$\\lambda_r/\\lambda_1$ (log)")
    ps.panel_label(ax, "c", x=-0.18, y=1.04)

    ps.save(fig, OUT_DIR, "figS05_mode_scale_energy")
    print(f"  [S5] threshold_modes 90%: "
          f"{[ (b, tm[b]['0.9']) for b in BANDS]}")
    return 0


# ══════════════════════════════════════════════════════════════════
# S6: tau sensitivity
# ══════════════════════════════════════════════════════════════════
def s6_tau():
    d = json.loads((SUPP / "threshold_sensitivity.json").read_text())
    results = d["results"]
    fig, axes = ps.figure(6.6, 3.0, 1, 2)
    for axi, (ax, sigma_val, ttl) in enumerate([
        (axes[0], 0.0, "$\\sigma$=0 (clean)"),
        (axes[1], 0.01, "$\\sigma$=0.01 (transition)"),
    ]):
        for ti, tau in enumerate([0.03, 0.05, 0.08]):
            means = []
            for m in [10, 15, 20, 30, 50]:
                row = next((r for r in results if r["model"] == "mlp"
                            and r["mask_num"] == m
                            and abs(r["sigma"] - sigma_val) < 1e-10
                            and abs(r["tau"] - tau) < 1e-10), None)
                means.append(row["mean_S_full"] if row else 0)
            ax.plot([10, 15, 20, 30, 50], means, "o-",
                    color=[ps.SIGMA_COLORS[0.0], ps.SIGMA_COLORS[0.01],
                           ps.SIGMA_COLORS[0.1]][ti],
                    lw=1.3, ms=4.5, label=f"$\\tau$={tau:.2f}")
        ax.set_xlabel("Sensor count $M$")
        ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
        ax.set_xticks([10, 15, 20, 30, 50])
        ax.set_ylim(-0.2, 5.2)
        ax.legend(fontsize=6.5)
        ps.panel_label(ax, chr(ord("a") + axi), x=-0.2, y=1.05)
        _ = ttl
    fig.tight_layout()
    ps.save(fig, OUT_DIR, "figS06_tau_sensitivity")
    print(f"  [S6] tau sensitivity saved")


# ══════════════════════════════════════════════════════════════════
# S7 / S8: sensor family
# ══════════════════════════════════════════════════════════════════
def s7_s8():
    # S7: multi-mask GER vs M
    rows = list(csv.DictReader(open(SUPP / "sensor_family" / "sensor_count_effect.csv",
                                    encoding="utf-8")))
    data = {}
    for r in rows:
        data.setdefault(r["model"], {}).setdefault(float(r["sigma"]), {})[
            int(r["sensor_count"])] = {
            "median": float(r["ger_mean_median"]),
            "min": float(r["ger_mean_min"]),
            "max": float(r["ger_mean_max"]),
        }
    fig, axes = ps.figure(6.6, 2.1, 1, 3)
    Ms = [10, 15, 20, 30, 50]
    for ax, model in zip(axes, ["mlp", "ridge", "gappy"]):
        for sigma, ls, color in [(0.0, "-", ps.MODEL_COLORS["MLP"]),
                                 (0.01, "--", ps.MODEL_COLORS["Ridge"])]:
            if sigma not in data.get(model, {}):
                continue
            vals = [data[model][sigma][M] for M in Ms]
            med = np.array([v["median"] for v in vals])
            lo = np.array([v["min"] for v in vals])
            hi = np.array([v["max"] for v in vals])
            ax.errorbar(Ms, med, yerr=[med - lo, hi - med], ls=ls, color=color,
                        fmt="o", ms=3.5, lw=1.2, capsize=2.5,
                        label=f"$\\sigma$={sigma:g}")
        ax.set_xlabel("$M$")
        ax.set_xticks(Ms)
        ax.set_title({"mlp": "MLP", "ridge": "Ridge", "gappy": "Gappy POD"}[model],
                     fontsize=ps.TITLE_FONT)
        if model == "mlp":
            ax.set_ylabel("Per-sequence mean GER")
        ax.legend(fontsize=6.5)
    axes[0].set_yscale("log")
    ps.save(fig, OUT_DIR, "figS07_sensor_family_ger")

    # S8: MLP-VCNN paired forest
    d = json.loads((SUPP / "paired_model_comparison.json").read_text())
    metrics = ["GER", "S_full", "S_coh", "W1", "vorticity_RMSE", "gradient_RMSE"]
    labels = {"GER": "GER", "S_full": "$S_{\\mathrm{full}}$",
              "S_coh": "$S_{\\mathrm{coh}}$", "W1": "W1 error",
              "vorticity_RMSE": "Laplacian RMSE",
              "gradient_RMSE": "Gradient RMSE"}
    fig, ax = ps.figure(4.2, 3.2, 1, 1)
    ypos = np.arange(len(metrics))
    for i, m in enumerate(metrics):
        a = d["metrics"][m]
        mean, lo, hi = a["mean_diff"], a["ci_low"], a["ci_high"]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=ps.MODEL_COLORS["MLP"], ms=4, capsize=3, lw=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[m] for m in metrics], fontsize=7.5)
    ax.axvline(0, color="0.2", lw=0.7, ls="--")
    ax.set_xlabel("Paired difference (MLP $-$ VCNN)")
    ps.save(fig, OUT_DIR, "figS08_sensor_family_paired")
    print(f"  [S7/S8] saved")
    return 0


def main() -> int:
    ps.apply()
    print("== Appendix figures ==")
    s1_oracle()
    s2_phase()
    s3_diagnostics()
    s4_three_layer()
    s5_mode_scale()
    s6_tau()
    s7_s8()
    return 0


if __name__ == "__main__":
    sys.exit(main())

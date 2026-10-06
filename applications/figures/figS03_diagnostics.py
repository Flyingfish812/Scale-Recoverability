#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS03_diagnostics.py — supplementary figures S10 and S12: noise and level diagnostics

Purpose
    (a) mean degradation ratio of every model per band under sensor noise;
    (b) decomposition-level sensitivity: the normalised scale count
        S_full/(L+1) and the per-band mean error of each level.
    The third panel of this appendix figure, the coherent-only sample, is drawn by figS03c_pod_dominant.py.
Data source
    artifacts/statistics/noise_propagation.json
    artifacts/statistics/level_sensitivity.json
Output
    artifacts/figures/figS03_noise_propagation.pdf
    artifacts/figures/figS03_level_sensitivity.pdf
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
SUPP = _ROOT / "artifacts" / "statistics"

BANDS = ps.BANDS
TAU = ps.TAU


def noise_propagation() -> None:
    """(a) mean degradation ratio per model and band under sensor noise."""
    d = json.loads((SUPP / "noise_propagation.json").read_text())
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


def level_sensitivity():
    """(b) normalised scale count and per-band error for each level L."""
    level_data = json.loads((SUPP / "level_sensitivity.json").read_text())["levels"]
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
    return 0


def main() -> int:
    ps.apply()
    noise_propagation()
    level_sensitivity()
    return 0


if __name__ == "__main__":
    sys.exit(main())

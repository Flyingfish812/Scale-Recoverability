#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS04_three_layer.py — Fig. S4 three-layer error decomposition

Purpose
    Mean per-band relative L2 error split into the total error, the POD
    truncation part and the model part for one VCNN configuration, with the
    mean criterion tau as a reference line.
Data source
    artifacts/statistics/band_error_records.json (via records)
Output
    artifacts/figures/figS04_three_layer.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import records
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"

BANDS = ps.BANDS
TAU = ps.TAU


def main() -> int:
    ps.apply()

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

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""extract_v5_ger_fits.py — 从 three_layer_fixed.json (v5) 提取 GER 幂律拟合参数, 落盘 JSON。

解决 OI-2 的数据源缺口: v5 管线的拟合参数目前仅存在于 regenerate_figures.py 的运行时,
未落盘为可核验的 JSON 文件。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RESULTS = ROOT / "artifacts" / "derived" / "main" / "statistics"

DATA_V5 = RESULTS / "20260715" / "three_layer_fixed.json"
GP_V4 = RESULTS / "20260714" / "gappy_pod_baseline.json"
OUT = RESULTS / "20260719" / "ger_fits_v5.json"


def main():
    if not DATA_V5.exists():
        print(f"ERROR: {DATA_V5} not found", file=sys.stderr)
        sys.exit(1)

    with open(DATA_V5) as f:
        d = json.load(f)
    results = d["results"]

    # Aggregate per model, per M at σ=0
    model_m_ger = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["noise_sigma"] == 0.0:
            model_m_ger[r["model_type"]][r["mask_num"]].append(r["GER"])

    M_ref = np.array([10, 15, 20, 30, 50], dtype=float)
    output = {"source": str(DATA_V5), "pipeline": "v5 (20260715 three_layer_fixed)", "models": {}}

    for mt in ["ridge", "mlp", "vcnn"]:
        gers = [np.mean(model_m_ger[mt].get(int(m), [np.nan])) for m in M_ref]
        gers_arr = np.array(gers)
        valid = ~np.isnan(gers_arr)
        if valid.sum() >= 3:
            log_m = np.log10(M_ref[valid])
            log_g = np.log10(gers_arr[valid])
            slope, intercept, r_val, p_val, std_err = stats.linregress(log_m, log_g)
            output["models"][mt] = {
                "alpha": round(float(slope), 3),
                "R2": round(float(r_val ** 2), 3),
                "intercept": round(float(intercept), 4),
                "p_value": float(p_val),
                "GER_by_M": {str(int(m)): round(float(g), 6) for m, g in zip(M_ref, gers_arr) if not np.isnan(g)},
            }
            print(f"{mt}: α={slope:.3f}, R²={r_val**2:.3f}")
        else:
            output["models"][mt] = {"error": "insufficient valid data points"}

    # Gappy POD
    if GP_V4.exists():
        with open(GP_V4) as f:
            gp = json.load(f)
        gp_data = {r["mask_num"]: r["GER_mean"] for r in gp["results"] if r["noise_sigma"] == 0.0}
        mg = sorted(gp_data.keys())
        gg = [gp_data[m] for m in mg]
        if len(mg) >= 3:
            log_mg = np.log10(mg)
            log_gg = np.log10(gg)
            slope_g, intercept_g, r_val_g, _, _ = stats.linregress(log_mg, log_gg)
            output["models"]["gappy"] = {
                "alpha": round(float(slope_g), 3),
                "R2": round(float(r_val_g ** 2), 3),
                "intercept": round(float(intercept_g), 4),
                "p_value": float(r_val_g),
                "GER_by_M": {str(m): round(float(g), 6) for m, g in zip(mg, gg)},
            }
            print(f"gappy: α={slope_g:.3f}, R²={r_val_g**2:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()

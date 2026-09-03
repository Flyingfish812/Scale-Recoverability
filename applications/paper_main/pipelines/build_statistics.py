#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_statistics.py — 中间统计产物编排 (检查跳过, 缺失才重算) (P0-10)。

所有论文中间统计 JSON 的正式重算入口。规则:
  - 目标产物已存在 (artifacts/derived/main/statistics/ 或 artifacts/ridge_closed_form_sweep_nc) → 跳过;
  - 缺失或 --force → 运行对应 _canonical 数据脚本重算 (输出同一目录)。

用法:
    conda run -n sana python -m applications.paper_main.pipelines.build_statistics
    conda run -n sana python -m applications.paper_main.pipelines.build_statistics --force
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATS = ROOT / "artifacts" / "derived" / "main" / "statistics"
RIDGE_NPZ = ROOT / "artifacts" / "ridge_closed_form_sweep_nc"

MODULE = "applications.paper_main.analyses._canonical."

# (目标产物, canonical 模块, 说明) — 目标缺失时运行模块
STEPS = [
    (RIDGE_NPZ / "ridge_n0010/seed000/tests/s0000/test_raw.npz",
     MODULE + "compute_ridge_closed_form", "闭式 Ridge 逐样本 NPZ"),
    (STATS / "s05_true_ridge.json",         MODULE + "compute_s05_ridge",      "S05 闭式 Ridge 统计"),
    (STATS / "s23_gappy_pod_fixed.json",    MODULE + "compute_s23_gappy",      "S2.3 Gappy POD 修正"),
    (STATS / "s021_gappy_pod_wavelet.json", MODULE + "compute_s021_gappy",     "S2.1 Gappy 小波"),
    (STATS / "s26_pass_probability.json",   MODULE + "compute_s26_phase",      "S2.6 相位图"),
    (STATS / "s27_threshold_sensitivity.json", MODULE + "compute_s27_tau",     "S2.7 τ 敏感性"),
    (STATS / "s10_tables_12_13.json",       MODULE + "compute_s10_tables",     "S10 表 12/13"),
    (STATS / "s08b_low_ger_final.json",     MODULE + "compute_s08b_lowger",    "S08b 低 GER"),
    (STATS / "s02_recomputed_values.json",  MODULE + "compute_s02_delta",      "S02 ΔE/恢复率/decile"),
    (STATS / "oracle_audit_testset.json",   MODULE + "compute_oa_testset",     "Oracle 审计 (测试集)"),
    (STATS / "ger_fits_v5.json",            MODULE + "compute_gf_v5_gerfits",  "GER 幂律拟合 v5"),
    (STATS / "fourier_spectral_baseline.json", MODULE + "compute_fb2_fourier", "Fourier 基线"),
    (STATS / "symmetric_control.json",      MODULE + "compute_sc_symmetric",   "对称控制"),
    (STATS / "fig03_data.json",             MODULE + "compute_equal_ger_42000", "equal-GER (42k 闭式 Ridge)"),
]


def main() -> int:
    force = "--force" in sys.argv
    ok = True
    for target, mod, desc in STEPS:
        if target.exists() and not force:
            print(f"[skip] {desc}: {target.relative_to(ROOT)}")
            continue
        print(f"\n=== 重算 {desc} ({mod}) ===")
        cmd = [sys.executable, "-m", mod] + (["--force"] if force else [])
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            ok = False
            print(f"[build_statistics] ❌ {mod} 失败")
        else:
            print(f"[build_statistics] ✅ {mod} 完成")
    print("\n[build_statistics] " + ("✅ 全部就绪" if ok else "❌ 存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

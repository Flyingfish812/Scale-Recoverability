#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage_statistics_pool.py — 将 main 中间统计产物落位到有序产物文件夹。

背景 (P0-10 归档重构):
  scripts/ 与 results/ 已整体归档至 _legacy/scripts_main/ 与 _legacy/results_main/。
  为避免 paper_main 到处软链接 / 从归档目录读取, 将论文所需的中间统计 JSON
  (已核验的正确版本) 复制到统一产物目录:

      artifacts/derived/main/statistics/

  每个文件登记来源 (legacy 路径) + sha256 + 落位时间, 写入 statistics_manifest.json。

用法:
    conda run -n sana python -m applications.paper_main.pipelines.stage_statistics_pool
    # --force 覆盖已存在文件
    conda run -n sana python -m applications.paper_main.pipelines.stage_statistics_pool --force

说明:
  - 本脚本只做"落位已有正确产物", 不重算。
  - 重算入口是 analyses/_canonical/compute_*.py (输出同一目录, 缺失或 --force 时执行)。
  - 外人不依赖 _legacy: 拿到本脚本 + 归档 JSON 即可重建 statistics/。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATS = ROOT / "artifacts" / "derived" / "main" / "statistics"
LEGACY_RESULTS = ROOT / "_legacy" / "results_main"

# (目标文件名, legacy 日期目录, legacy 文件名)
SOURCES = [
    # equal-GER (42k 闭式 Ridge, 2026-08-05 重算)
    ("fig03_data.json",              "20260805", "fig03_data.json"),
    ("s33_strict_equal_ger.json",    "20260805", "s33_strict_equal_ger.json"),
    # oracle 审计
    ("oracle_audit_testset.json",    "20260720", "oracle_audit_testset.json"),
    ("oracle_audit_refined.json",    "20260715", "oracle_audit_refined.json"),
    # 基线 / 物理
    ("fourier_spectral_baseline.json", "20260715", "fourier_spectral_baseline.json"),
    ("fourier_spectral_baseline.csv",  "20260715", "fourier_spectral_baseline.csv"),
    ("symmetric_control.json",       "20260715", "symmetric_control.json"),
    ("three_layer_fixed.json",       "20260715", "three_layer_fixed.json"),
    ("three_layer_errors_full.json", "20260714", "three_layer_errors_full.json"),
    ("thesis_data_audit.json",       "20260714", "thesis_data_audit.json"),
    ("gappy_pod_baseline.json",      "20260714", "gappy_pod_baseline.json"),
    # 稳健性配对 (legacy 参考)
    ("equal_ger_68_pairs_full.json", "20260720", "equal_ger_68_pairs_full.json"),
    # GER 拟合
    ("ger_fits_v5.json",             "20260719", "ger_fits_v5.json"),
    # S2 系列重算 (S02/S021/S05/S08b/S10/S23/S26/S27)
    ("s02_recomputed_values.json",   "20260722", "s02_recomputed_values.json"),
    ("s021_gappy_pod_wavelet.json",  "20260722", "s021_gappy_pod_wavelet.json"),
    ("s05_true_ridge.json",          "20260723", "s05_true_ridge.json"),
    ("s08b_low_ger_final.json",      "20260723", "s08b_low_ger_final.json"),
    ("s10_tables_12_13.json",        "20260723", "s10_tables_12_13.json"),
    ("s23_gappy_pod_fixed.json",     "20260721", "s23_gappy_pod_fixed.json"),
    ("s26_pass_probability.json",    "20260721", "s26_pass_probability.json"),
    ("s27_threshold_sensitivity.json", "20260721", "s27_threshold_sensitivity.json"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    force = "--force" in sys.argv
    STATS.mkdir(parents=True, exist_ok=True)
    manifest_path = STATS / "statistics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("generated", "")
    manifest.setdefault("entries", {})

    ok = True
    for target, date_dir, name in SOURCES:
        src = LEGACY_RESULTS / date_dir / name
        dst = STATS / target
        if not src.exists():
            print(f"[skip] 来源缺失: {src.relative_to(ROOT)}")
            ok = False
            continue
        entry = manifest["entries"].setdefault(target, {})
        if dst.exists() and not force:
            if entry.get("sha256") == sha256(dst):
                print(f"[ok  ] {target} (已落位, 未变)")
                continue
            print(f"[upd ] {target} (sha 变化, 重新复制)")
        shutil.copy2(src, dst)
        entry.update({
            "source": f"_legacy/results_main/{date_dir}/{name}",
            "sha256": sha256(dst),
            "staged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"[copy] {target} ← {date_dir}/{name}")

    manifest["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[manifest] {manifest_path.relative_to(ROOT)} ({len(manifest['entries'])} entries)")
    print("[stage_statistics_pool] " + ("✅ 完成" if ok else "⚠️ 存在缺失来源"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

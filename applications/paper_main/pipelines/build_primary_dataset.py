#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_primary_dataset.py — 构建 main 主结果池 (P0-4)。

42,000 记录口径 (sec04_validation.tex 明示):
    MLP  18,000 = 300 快照 × 20 配置 (M×σ) × 3 种子
    VCNN 18,000 = 300 × 20 × 3
    Ridge  6,000 = 300 × 20 (闭式解, 确定性, 无训练种子)

数据源:
    MLP/VCNN 逐记录指标: results/20260714/three_layer_errors_full.json
        (54,000 行 = 3 模型 × 3 种子 × 20 配置 × 300; seed 列有 bug 全部记 0,
         3 个种子藏在行三重结构中; 已验证行序 = [seed0, seed101, seed202])
    Ridge 闭式: results/20260723/s05_true_ridge.json (20 配置, 配置级均值)
    预测路径: artifacts/ 下 NPZ (与 scripts/20260722/s33_strict_equal_ger.py 约定一致)

输出:
    artifacts/derived/main/primary_records.parquet

用法:
    conda run -n sana python -m applications.paper_main.pipelines.build_primary_dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "artifacts" / "derived" / "main"

THREE_LAYER = ROOT / "artifacts" / "derived" / "main" / "statistics" / "three_layer_errors_full.json"
RIDGE_S05 = ROOT / "artifacts" / "derived" / "main" / "statistics" / "s05_true_ridge.json"

# 实验参数统一来自 configs/*.yaml (P0-5)
from applications.paper_main.config import get_config  # noqa: E402
_cfg = get_config()
MASK_NUMS = _cfg.M_values
SIGMAS = _cfg.sigma_values
SEEDS = _cfg.seeds
BANDS = _cfg.bands
POD_RANK = _cfg.pod_rank


def sigma_code(s: float) -> str:
    return {0.0: "s0000", 0.001: "s0010", 0.01: "s0100", 0.1: "s1000"}[s]


def npz_path(model: str, mask: int, sigma: float, seed: int) -> Path:
    code = sigma_code(sigma)
    if model == "vcnn":
        if seed == 0:
            return ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000" / f"vcnn_n{mask:04d}_seed000_custom/tests/{code}/test_raw.npz"
        return ROOT / "artifacts/vcnn_results" / f"vcnn_sweep_nc_2000_seed{seed:03d}" / f"vcnn_n{mask:04d}_seed000_custom/tests/{code}/test_raw.npz"
    return ROOT / "artifacts/pod_model_sweep_nc" / f"{model}_n{mask:04d}" / f"seed{seed:03d}/tests/{code}/test_raw.npz"


def build_mlp_vcnn() -> pd.DataFrame:
    """MLP/VCNN: 从 three_layer_errors_full 提取 18k+18k, 按行序恢复种子。"""
    recs = json.loads(THREE_LAYER.read_text(encoding="utf-8"))
    rows = []
    # 按 (model, mask, sigma, sample) 分组, 行序 → seed
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        if r["model_type"] not in ("mlp", "vcnn"):
            continue
        key = (r["model_type"], r["mask_num"], r["noise_sigma"], r["sample_idx"])
        groups.setdefault(key, []).append(r)

    for (model, mask, sigma, sample), copies in groups.items():
        for ord_, r in enumerate(copies):
            seed = SEEDS[ord_]  # 已验证: 行序 = [0, 101, 202]
            rows.append({
                "record_id": f"{model}_{mask:02d}_{sigma_code(sigma)}_{seed:03d}_{sample:03d}",
                "dataset": "nc", "model": model, "M": mask, "sigma": sigma,
                "training_seed": seed, "snapshot_id": sample, "pod_rank": POD_RANK,
                "prediction_path": str(npz_path(model, mask, sigma, seed).relative_to(ROOT)) if npz_path(model, mask, sigma, seed).exists() else None,
                "metrics_level": "per_record",
                "ger": r["GER"], "s_full": r["S_full_total"], "s_coh": r["S_coh_total"],
                **{f"e_total_{b}": r[f"E_total_{b}"] for b in BANDS},
                **{f"e_trunc_{b}": r[f"E_trunc_{b}"] for b in BANDS},
                **{f"e_pred_{b}": r[f"E_pred_{b}"] for b in BANDS},
            })
    df = pd.DataFrame(rows)
    return df


def build_ridge() -> pd.DataFrame:
    """Ridge 闭式: 20 配置 × 300 快照 = 6,000; 逐记录 E_total (复现 s10 计算)。"""
    # 逐记录 E_total 由 s10.compute_ridge_closed_form_recovery() 复算持久化
    rec_file = OUT / "_ridge_closed_form_records.json"
    if not rec_file.exists():
        print("[build_primary_dataset] ❌ 缺少 ridge 逐记录文件, 先运行 s10 复算")
        return pd.DataFrame()
    records = json.loads(rec_file.read_text(encoding="utf-8"))
    s05 = json.loads(RIDGE_S05.read_text(encoding="utf-8"))
    s05_cfg = {(r["mask_num"], r["sigma"]): r for r in s05["results"]}

    rows = []
    # records 按 (mask, sigma) 分组, 组内顺序 = 样本顺序
    from collections import defaultdict
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["mask_num"], r["noise_sigma"])].append(r)

    for (mask, sigma), recs in grouped.items():
        p = npz_path("ridge", mask, sigma, 0)
        cfg = s05_cfg.get((mask, sigma), {})
        for sample, r in enumerate(recs):
            rows.append({
                "record_id": f"ridge_{mask:02d}_{sigma_code(sigma)}_det_{sample:03d}",
                "dataset": "nc", "model": "ridge", "M": mask, "sigma": sigma,
                "training_seed": None,  # 确定性模型, 无训练种子
                "snapshot_id": sample, "pod_rank": POD_RANK,
                "prediction_path": str(p.relative_to(ROOT)) if p.exists() else None,
                "metrics_level": "per_record",
                "ger": cfg.get("GER_mean"), "s_full": round(cfg["S_full_mean"]) if cfg else None, "s_coh": None,
                **{f"e_total_{b}": r[f"E_total_{b}"] for b in BANDS},
                **{f"e_trunc_{b}": None for b in BANDS},
                **{f"e_pred_{b}": None for b in BANDS},
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[build_primary_dataset] MLP/VCNN from three_layer_errors_full ...")
    mv = build_mlp_vcnn()
    print(f"    mlp+vcnn rows: {len(mv)}")
    print("[build_primary_dataset] Ridge closed-form (s05) ...")
    rd = build_ridge()
    print(f"    ridge rows: {len(rd)}")

    pool = pd.concat([mv, rd], ignore_index=True)
    # 校验
    n_total = len(pool)
    n_mlp = int((pool["model"] == "mlp").sum())
    n_vcnn = int((pool["model"] == "vcnn").sum())
    n_ridge = int((pool["model"] == "ridge").sum())
    n_snapshot = int(pool["snapshot_id"].nunique())
    dup = int(pool["record_id"].duplicated().sum())
    print(f"    TOTAL={n_total}  mlp={n_mlp}  vcnn={n_vcnn}  ridge={n_ridge}  unique_snapshots={n_snapshot}  dup_pk={dup}")
    assert n_total == 42000, f"预期 42000, 实际 {n_total}"
    assert n_snapshot == 300
    assert dup == 0

    pool.to_parquet(OUT / "primary_records.parquet", index=False)
    print(f"    saved: {OUT / 'primary_records.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_gappy_records.py — Gappy POD 结果池 (P0-4)。

6,000 记录口径: 20 配置 (M×σ) × 300 快照。
数据源: results/20260714/gappy_pod_baseline.json (20 配置, 配置级均值)。
注: gappy 无逐记录持久化文件, 指标为配置级 (metrics_level=config_mean)。

输出: artifacts/derived/main/gappy_records.parquet

用法:
    conda run -n sana python -m applications.paper_main.pipelines.build_gappy_records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "artifacts" / "derived" / "main"
SRC = ROOT / "artifacts" / "derived" / "main" / "statistics" / "gappy_pod_baseline.json"

from applications.paper_main.config import get_config  # noqa: E402
_cfg = get_config()
BANDS = _cfg.bands


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for r in data["results"]:
        mask, sigma = r["mask_num"], r["noise_sigma"]
        for sample in range(r.get("n_samples", 300)):
            rows.append({
                "record_id": f"gappy_{mask:02d}_{sigma:g}_{sample:03d}",
                "dataset": "nc", "model": "gappy_pod", "M": mask, "sigma": sigma,
                "training_seed": None, "snapshot_id": sample, "pod_rank": data.get("pod_rank_used"),
                "metrics_level": "config_mean",
                "ger": r["GER_mean"], "s_full": round(r["S_full_mean"], 4), "s_coh": None,
                **{f"e_total_{b}": r["per_band_E_total_mean"].get(b) for b in BANDS},
            })
    df = pd.DataFrame(rows)
    n = len(df)
    dup = int(df["record_id"].duplicated().sum())
    n_snap = int(df["snapshot_id"].nunique())
    print(f"[build_gappy_records] total={n}  unique_snapshots={n_snap}  dup_pk={dup}")
    assert n == 6000, f"预期 6000, 实际 {n}"
    assert n_snap == 300 and dup == 0
    df.to_parquet(OUT / "gappy_records.parquet", index=False)
    print(f"    saved: {OUT / 'gappy_records.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

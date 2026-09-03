#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_oracle_records.py — Oracle 结果池 (P0-4)。

Oracle 截断误差只依赖 POD rank (不随 M/σ/seed 重复), 每个 (dataset, rank, band, stat)
一条记录。数据源: results/20260715/oracle_audit_refined.json (oa_r, 三统计量体系 mean/Q95/max)。

输出: artifacts/derived/main/oracle_records.parquet

用法:
    conda run -n sana python -m applications.paper_main.pipelines.build_oracle_records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "artifacts" / "derived" / "main"
SRC = ROOT / "artifacts" / "derived" / "main" / "statistics" / "oracle_audit_refined.json"

from applications.paper_main.config import get_config  # noqa: E402
_cfg = get_config()
BANDS = _cfg.bands


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for s in data["summaries"]:
        ds = s["dataset"]
        table = s["audit_table"]
        # audit_table: {rank_str: {"bands": {band: {"mean":..,"q95":..,"max":..}}} 或 {band: {...}}
        for rank_str, entry in (table.items() if isinstance(table, dict) else []):
            bands = entry.get("bands", entry) if isinstance(entry, dict) else {}
            for band, stats in (bands.items() if isinstance(bands, dict) else []):
                if not isinstance(stats, dict):
                    continue
                rows.append({
                    "record_id": f"oracle_{ds}_{rank_str}_{band}",
                    "dataset": ds, "model": "oracle", "pod_rank": int(rank_str),
                    "band": band,
                    "e_trunc_mean": stats.get("mean"), "e_trunc_q95": stats.get("q95"),
                    "e_trunc_max": stats.get("max"),
                    "M": None, "sigma": None, "training_seed": None,  # Oracle 与配置无关
                    "metrics_level": "per_rank",
                })
    df = pd.DataFrame(rows)
    dup = int(df["record_id"].duplicated().sum())
    print(f"[build_oracle_records] rows={len(df)}  dup_pk={dup}  datasets={sorted(df['dataset'].unique())}")
    assert dup == 0
    df.to_parquet(OUT / "oracle_records.parquet", index=False)
    print(f"    saved: {OUT / 'oracle_records.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

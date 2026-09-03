#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_all.py — one-click reproduction entrypoint for the main paper.

Pipeline:
  1. build_statistics            (intermediate statistics; recompute if missing)
  2. build_primary_dataset / build_gappy_records / build_oracle_records (data pools)
  3. sample_audit                (42,000-record closure audit)
  4. build_tables                (regenerate manuscript tables; skipped when the
                                  private thesis_src/ tree is absent)
  5. build_figures               (redraw all paper figures under build/)

Usage:
    conda run -n luna python -m applications.paper_main.pipelines.build_all
    conda run -n luna python -m applications.paper_main.pipelines.build_all --force
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PIPE = "applications.paper_main.pipelines."

STEPS = [
    (PIPE + "build_statistics",      "intermediate statistics (recompute if missing)"),
    (PIPE + "build_primary_dataset", "primary data pool 42,000"),
    (PIPE + "build_gappy_records",   "gappy data pool 6,000"),
    (PIPE + "build_oracle_records",  "oracle data pool 70"),
    ("applications.paper_main.analyses.sample_audit", "sample audit (42k closure)"),
    (PIPE + "build_tables",          "rebuild tables (skipped if thesis_src absent)"),
    (PIPE + "build_figures",         "redraw all paper figures"),
]


def main() -> int:
    force = "--force" in sys.argv
    ok = True
    for mod, desc in STEPS:
        print(f"\n########## {desc} ({mod}) ##########")
        cmd = [sys.executable, "-m", mod] + (["--force"] if force else [])
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            ok = False
            print(f"[build_all] ❌ {mod} 失败")
    print("\n[build_all] " + ("✅ 全部通过" if ok else "❌ 存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

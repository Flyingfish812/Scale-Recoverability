#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_tables.py — 从 paper_facts.yaml 重建论文全部表格与数值宏。

原则: thesis_src/ 只读。将生成器与真值层复制到 build/tools_gen + build/data,
在其内部运行 (ROOT=build), 输出 build/generated/*, 再与 thesis_src/generated/
哈希比对。

用法:
    conda run -n sana python -m applications.paper_main.pipelines.build_tables
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root

BUILD = ROOT / "applications" / "paper_main" / "build"
TOOLS_GEN = BUILD / "tools_gen"
DATA_GEN = BUILD / "data"
GENERATED = BUILD / "generated"
TABLES_OUT = GENERATED / "tables"

THESIS_SRC = ROOT / "thesis_src"
THESIS_TOOLS = THESIS_SRC / "tools"
THESIS_DATA = THESIS_SRC / "data"
THESIS_GENERATED = THESIS_SRC / "generated"

GENERATORS = ["generate_paper_tables.py", "generate_paper_numbers.py"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def compare_dir(a: Path, b: Path) -> list[str]:
    """返回 a 中与 b 对应文件哈希不一致的列表。"""
    diffs = []
    if not a.exists():
        return [f"{a} 不存在"]
    for fa in sorted(a.rglob("*")):
        if not fa.is_file():
            continue
        rel = fa.relative_to(a)
        fb = b / rel
        if not fb.exists():
            diffs.append(f"缺失: {rel}")
        elif sha256(fa) != sha256(fb):
            diffs.append(f"不一致: {rel}")
    return diffs


def main() -> int:
    # 论文表格生成器与真值层位于私有论文仓 (thesis_src/), 不在公开源码仓中。
    # 缺少时本步骤仅打印提示并跳过, 不视为失败。
    if not THESIS_SRC.exists():
        print("[build_tables] thesis_src/ 不存在 (公开仓不含私有论文源码); 跳过论文表格重建。")
        return 0

    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    DATA_GEN.mkdir(parents=True, exist_ok=True)
    TOOLS_GEN.mkdir(parents=True, exist_ok=True)

    # 1. 复制生成器与真值层到 build (只读执行, 不改 thesis_src)
    for name in GENERATORS:
        src = THESIS_TOOLS / name
        if src.exists():
            shutil.copy2(src, TOOLS_GEN / name)
    for y in ["paper_facts.yaml", "claims.yaml"]:
        src = THESIS_DATA / y
        if src.exists():
            shutil.copy2(src, DATA_GEN / y)

    # 2. 在 build/tools_gen 内运行 (生成器 ROOT=build)
    env = dict(sys._base_executable and {})  # noqa: C416
    for name in GENERATORS:
        print(f"[build_tables] running {name}")
        r = subprocess.run(
            [sys.executable, str(TOOLS_GEN / name)],
            cwd=TOOLS_GEN, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return r.returncode

    # 3. 与 thesis_src 比对
    print("[build_tables] 比对 thesis_src/generated ...")
    diffs = []
    diffs += compare_dir(TABLES_OUT, THESIS_GENERATED / "tables")
    num_b = GENERATED / "paper_numbers.tex"
    num_t = THESIS_GENERATED / "paper_numbers.tex"
    if num_b.exists() and num_t.exists() and sha256(num_b) != sha256(num_t):
        diffs.append("paper_numbers.tex 不一致")
    if not diffs:
        print("[build_tables] ✅ 全部一致: numbers + %d tables 与 thesis_src 匹配" % len(list(TABLES_OUT.glob("*.tex"))))
        return 0
    print("[build_tables] ⚠️ 差异:")
    for d in diffs:
        print("   -", d)
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_figS3_S4.py — 重绘附录相位图 (正确数据, 多折线图格式)。

figS3 (Ridge phase): 数据源 results/20260723/s05_true_ridge.json (闭式 Ridge, 20260723 反馈要求)
figS4 (VCNN phase):  数据源 thesis_src/data/paper_facts.yaml → results.phase_diagram_vcnn
                     (最终值, 来源 results/20260721/s26_pass_probability.json, 已核实一致)

格式: 多折线图 — X = M (sensor count), Y = mean S_full, 每个 σ 一条线。
      与 20260714 sfig_vcnn_phase (旧 figS9) 风格一致; 不使用热力图。

用途: 论文 figS3 / figS4 的替换版。thesis_src 冻结, 输出到 build/figures/。

用法:
    conda run -n sana python -m applications.paper_main.figures.build_figS3_S4
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from applications.paper_main.config import get_config  # noqa: E402
_cfg = get_config()
OUT_DIR = _cfg.figures_out

M_VALS = _cfg.M_values
# (sigma_key, marker, color, label)
SIGMA_CONFIGS = [
    ("0.0", "o-", "#2ecc71", "0"),
    ("0.001", "s-", "#3498db", "10\u207b\u00b3"),
    ("0.01", "D-", "#e67e22", "10\u207b\u00b2"),
    ("0.1", "v-", "#e74c3c", "10\u207b\u00b9"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def git_commit() -> str:
    import subprocess
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _draw(series: dict, title: str, out_stem: str, meta: dict) -> None:
    """series: {sigma_key: [S_full per M]}。绘制多折线图并保存 pdf/png + metadata。"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s_key, marker, color, s_label in SIGMA_CONFIGS:
        vals = series.get(s_key)
        if vals is None:
            continue
        ax.plot(M_VALS, vals, marker, color=color, label=f"\u03c3={s_label}", lw=1.5, markersize=7)
    ax.set_xlabel("M (sensor count)")
    ax.set_ylabel("mean $S_{\\mathrm{full}}$")
    ax.set_title(title)
    ax.set_xticks(M_VALS)
    ax.set_xlim(5, 55)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    pdf = OUT_DIR / f"{out_stem}.pdf"
    png = OUT_DIR / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta["outputs"] = [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))]
    meta["data_sha256"] = sha256(pdf)
    meta["git_commit"] = git_commit()
    meta["generated_at"] = datetime.now().isoformat(timespec="seconds")
    (OUT_DIR / f"{out_stem}.metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  \u2705 {out_stem}: pdf={pdf.relative_to(ROOT)} png={png.relative_to(ROOT)}")


def build_figS3() -> None:
    """Ridge 相位图 (闭式 Ridge, s05_true_ridge.json)。"""
    data = json.loads((ROOT / "artifacts/derived/main/statistics/s05_true_ridge.json").read_text(encoding="utf-8"))
    series = {k: [np.nan] * len(M_VALS) for k, *_ in SIGMA_CONFIGS}
    for r in data["results"]:
        si = M_VALS.index(r["mask_num"])
        series[str(r["sigma"])][si] = r["S_full_mean"]
    _draw(
        series,
        "Ridge Phase Diagram: $S_{\\mathrm{full}}$ vs Sensor Count (closed-form)",
        "figS3_ridge_phase",
        {
            "figure": "Figure S3 (替换版)",
            "file": "figS3_ridge_phase",
            "tex_ref": "appendix.tex:93 (sfig:ridge_phase)",
            "script": "applications/paper_main/figures/build_figS3_S4.py",
            "data_source": "artifacts/derived/main/statistics/s05_true_ridge.json (closed-form Ridge)",
            "format_note": "多折线图 (非热力图); 与 figS4 格式一致",
            "replaces": "thesis_src/figures/figS3_ridge_phase.png (legacy 旧数据)",
        },
    )


def build_figS4() -> None:
    """VCNN 相位图 (最终数据, paper_facts phase_diagram_vcnn)。"""
    import yaml
    facts = yaml.safe_load((ROOT / "thesis_src/data/paper_facts.yaml").read_text(encoding="utf-8"))
    matrix = facts["results"]["phase_diagram_vcnn"]["matrix"]  # 5 M 行 x 4 σ 列
    series = {}
    for i, s_key in enumerate(s for s, *_ in SIGMA_CONFIGS):
        series[s_key] = [float(row[i]) for row in matrix]
    _draw(
        series,
        "VCNN Phase Diagram: $S_{\\mathrm{full}}$ vs Sensor Count",
        "figS4_vcnn_phase",
        {
            "figure": "Figure S4 (替换版, 取代 figS9)",
            "file": "figS4_vcnn_phase",
            "tex_ref": "appendix.tex:104 (sfig:vcnn_phase)",
            "script": "applications/paper_main/figures/build_figS3_S4.py",
            "data_source": "thesis_src/data/paper_facts.yaml → results.phase_diagram_vcnn (= results/20260721/s26_pass_probability.json)",
            "format_note": "多折线图; 与 figS3 格式一致",
            "replaces": "thesis_src/figures/figS4_vcnn_phase.png (legacy) 与 figS9_vcnn_phase.pdf (20260714 过时数据, 待删除)",
        },
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[build_figS3_S4] figS3: Ridge phase (closed-form, line plot)")
    build_figS3()
    print("[build_figS3_S4] figS4: VCNN phase (final data, line plot)")
    build_figS4()
    return 0


if __name__ == "__main__":
    sys.exit(main())

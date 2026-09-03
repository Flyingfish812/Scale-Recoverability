#!/usr/bin/env python3
"""
P04: Figure S8 — Oracle Audit 三数据集对比（修复版）

问题:
  20260722 版 (s44) 使用 oracle_audit_testset.json 但访问路径写错:
    audit_table[rank].get("mean", {}).get(band)   →   实际结构是 audit_table[rank]["bands"][band]["mean"]
  导致所有误差值为 0，柱状图为空。

  20260717 版使用热图 (imshow)，但导师反馈不希望在附录中用热图。

修复:
  - 正确读取 audit_table[rank_str]["bands"][band]["mean"]
  - 3 面板 grouped bar chart（每数据集一面板）
  - 标注 safe rank，与正文一致（NC=32, RDB=None, SST=256）

数据源: results/20260720/oracle_audit_testset.json

用法:
  conda activate sana
  python scripts/20260723/p04_draw_figS8.py

输出:
  results/20260723/figS8_oracle_audit.pdf / .png
  thesis_src/figures/fig01_oracle_audit.pdf  (覆盖论文用图)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
THESIS_FIGURES = ROOT / "thesis_src" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

BANDS = ["A4", "W4", "W3", "W2", "W1"]
BAND_COLORS = {"A4": "#1f77b4", "W4": "#ff7f0e", "W3": "#2ca02c",
               "W2": "#d62728", "W1": "#9467bd"}
TAU = 0.05

DATASETS = {
    "nc": {
        "label": "NC (cylinder wake)",
        "ranks": [16, 32, 64, 128],
        "color": "blue",
    },
    "rdb_h5": {
        "label": "RDB (radial dam-break)",
        "ranks": [16, 32, 64, 128],
        "color": "orange",
    },
    "sst_weekly": {
        "label": "SST (sea surface temperature)",
        "ranks": [32, 64, 128, 256, 512],
        "color": "green",
    },
}


def compute_safe_rank(audit_table: dict, ranks: list[int]) -> int | None:
    """从审计表计算第一个满足 mean criterion 的 rank。"""
    for r in ranks:
        rs = str(r)
        if rs not in audit_table:
            continue
        bands = audit_table[rs].get("bands", {})
        if all(bands.get(b, {}).get("mean", 1) < TAU for b in BANDS):
            return r
    return None


def main():
    print("=" * 60)
    print("  P04: Figure S8 — Oracle Audit (3-dataset comparison)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n1. Loading data...")
    oa_path = ROOT / "artifacts" / "derived" / "main" / "statistics" / "oracle_audit_testset.json"
    with open(oa_path) as f:
        oa = json.load(f)

    summaries = {s["dataset"]: s for s in oa["summaries"]}
    print(f"   Datasets: {list(summaries.keys())}")

    # ── 2. Compute safe ranks from data ───────────────────────
    print("\n2. Computing safe ranks from data...")
    for ds_name, ds_cfg in DATASETS.items():
        s = summaries.get(ds_name, {})
        table = s.get("audit_table", {})
        data_safe = compute_safe_rank(table, ds_cfg["ranks"])
        ds_cfg["safe_rank"] = data_safe
        print(f"   {ds_name}: safe_rank={data_safe}")

    # ── 3. Build figure ───────────────────────────────────────
    print("\n3. Drawing figure...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    for ax, (ds_name, ds_cfg) in zip(axes, DATASETS.items()):
        summary = summaries.get(ds_name, {})
        audit_table = summary.get("audit_table", {})
        ranks = ds_cfg["ranks"]
        safe_rank = ds_cfg["safe_rank"]

        ax.set_title(ds_cfg["label"], fontsize=10)
        ax.set_xlabel("POD rank $r$")
        if ax == axes[0]:
            ax.set_ylabel("Mean truncation error per band")

        x_pos = np.arange(len(ranks))
        width = 0.12

        for bi, band in enumerate(BANDS):
            means = []
            for r in ranks:
                r_str = str(r)
                if r_str in audit_table:
                    # ✅ 正确路径: audit_table[rank]["bands"][band]["mean"]
                    bands_data = audit_table[r_str].get("bands", {})
                    band_info = bands_data.get(band, {})
                    m = band_info.get("mean", None)
                    means.append(m if m is not None else 0)
                else:
                    means.append(0)

            offset = (bi - 2) * width
            ax.bar(x_pos + offset, means, width, color=BAND_COLORS[band],
                   label=band if ds_name == "nc" else "", alpha=0.85)

        # τ reference line
        ax.axhline(y=TAU, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.98, TAU * 1.05, f"$\\tau={TAU}$",
                color="red", fontsize=8, ha="right", va="bottom")

        # Safe rank annotation
        if safe_rank is not None:
            if safe_rank in ranks:
                idx = ranks.index(safe_rank)
                # Use data coordinates to place annotation above threshold
                y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
                ax.annotate(
                    f"$r_{{\\mathrm{{mean}}}}={safe_rank}$",
                    xy=(idx, TAU), xytext=(idx, y_top * 0.85),
                    ha="center", fontsize=9, fontweight="bold",
                    color="green",
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                )
        else:
            ax.text(0.5, 0.9, "No rank satisfies\nmean criterion",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray", fontstyle="italic")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(1e-5, 2)

        if ds_name == "nc":
            ax.legend(loc="upper right", ncol=1, fontsize=7)

    plt.tight_layout(pad=1.5)

    # ── 4. Save ───────────────────────────────────────────────
    out_pdf = OUT_DIR / "figS8_oracle_audit.pdf"
    out_png = OUT_DIR / "figS8_oracle_audit.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\n  Saved: {out_pdf}")
    plt.close()
    # 发布由 publish_figures 统一处理 (figS8_oracle_audit.pdf)
    print("\n  ✓ Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fig05_combined.py — 重新绘制正文 Figure 5 为一张一行三列的长图 (三个独立面板)

背景:
  旧版 fig05 由两幅图拼成: (a) fig05_ger_sfull_vs_M.pdf (GER 与 S_full 两个
  子图挤在半栏) + (b) fig05_phase_mlp.pdf (单张 S_full 折线图, 与 (a) 右子图
  几乎重复, 且用旧数据源 thesis_data_audit 4.48 与正文 s26 的 4.60 不一致)。
  排版时呈现两窄一宽的不均匀布局。

本次改动:
  - 输出三张独立面板 PDF, 供正文以 0.32 栏宽并排为一行三列:
      fig05_ger_vs_M.pdf         (a) Mean GER vs M  (MLP, 4 噪声档, log y)
      fig05_sfull_vs_M.pdf       (b) Mean S_full vs M (MLP, 4 噪声档, S_full=3 参考线)
      fig05_phase_diagram.pdf    (c) S_full 相位图 (M x sigma 热力图, 三区制)
  - 全部使用权威数据源, 与正文宏/表格一致:
      GER    : artifacts/derived/main/statistics/three_layer_fixed.json (MLP)
      S_full : artifacts/derived/main/statistics/s26_pass_probability.json
               (phase_summary.mlp.mean_S_full; = paper_facts phase_diagram_mlp.matrix)

用法:
  conda run -n sana python3 applications/paper_main/figures/_canonical/fig05_combined.py

输出: applications/paper_main/build/figures_raw/fig05_{ger_vs_M,sfull_vs_M,phase_diagram}.pdf
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
# 数据文件 (s26/three_layer_fixed) 中 sigma 键的实际字符串格式:
#   "0.0", "0.001", "0.01", "0.1"  (注意 0 带小数点)
SIGMA_KEYS = ["0.0", "0.001", "0.01", "0.1"]
SIGMA_LABELS = ["0", r"10^{-3}", r"10^{-2}", r"10^{-1}"]
COLORS = ["#348ABD", "#E24A33", "#988ED5", "#8EBA42"]
MARKERS = ["o", "s", "^", "D"]


def load_mlp_ger() -> dict:
    """MLP 逐配置 mean GER (three_layer_fixed)."""
    tl = json.loads((ROOT / "artifacts/derived/main/statistics/three_layer_fixed.json").read_text())
    ger_by_cfg = defaultdict(list)
    for r in tl.get("results", []):
        if r.get("model_type") == "mlp":
            ger_by_cfg[(r["mask_num"], r["noise_sigma"])].append(r.get("GER", 0.0))
    return {k: float(np.mean(v)) for k, v in ger_by_cfg.items()}


def load_mlp_sfull() -> dict:
    """MLP 逐配置 mean S_full (s26, 权威源)."""
    s26 = json.loads((ROOT / "artifacts/derived/main/statistics/s26_pass_probability.json").read_text())
    ps = s26["phase_summary"]["mlp"]
    out = {}
    for m in MASK_NUMS:
        for s, skey in zip(SIGMA_VALS, SIGMA_KEYS):
            e = ps.get(str(m), {}).get(skey, {})
            out[(m, s)] = float(e.get("mean_S_full", 0.0)) if e else 0.0
    return out


def draw_panel_ger(ger):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for si, sigma in enumerate(SIGMA_VALS):
        means = [ger.get((m, sigma), 0.0) for m in MASK_NUMS]
        ax.plot(MASK_NUMS, means, color=COLORS[si], marker=MARKERS[si],
                label=f"$\\sigma={SIGMA_LABELS[si]}$", linewidth=1.5, markersize=6)
    ax.set_xlabel("Number of sensors $M$")
    ax.set_ylabel("Mean GER")
    ax.set_yscale("log")
    ax.legend(title="Test noise", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(MASK_NUMS)
    ax.set_title("Global error (MLP)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig05_ger_vs_M.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig05_ger_vs_M.pdf")


def draw_panel_sfull(sfull):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for si, sigma in enumerate(SIGMA_VALS):
        means = [sfull.get((m, sigma), 0.0) for m in MASK_NUMS]
        ax.plot(MASK_NUMS, means, color=COLORS[si], marker=MARKERS[si],
                label=f"$\\sigma={SIGMA_LABELS[si]}$", linewidth=1.5, markersize=6)
    ax.set_xlabel("Number of sensors $M$")
    ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax.legend(title="Test noise", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(MASK_NUMS)
    ax.set_ylim(-0.2, 5.2)
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title("Scale recoverability (MLP)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig05_sfull_vs_M.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig05_sfull_vs_M.pdf")


def draw_panel_phase(sfull):
    """S_full 相位图: 多折线图 (S_full vs M, 每 sigma 一条线)。

    全论文约定: 凡展示 5M x 4sigma 趋势的图一律使用多折线图,
    纵坐标 = 追踪数值 (这里为 mean S_full), 不使用热力图。
    """
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for si, sigma in enumerate(SIGMA_VALS):
        means = [sfull.get((m, sigma), 0.0) for m in MASK_NUMS]
        ax.plot(MASK_NUMS, means, color=COLORS[si], marker=MARKERS[si],
                label=f"$\\sigma={SIGMA_LABELS[si]}$", linewidth=1.5, markersize=6)
    ax.set_xlabel("Number of sensors $M$")
    ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax.legend(title="Test noise", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(MASK_NUMS)
    ax.set_ylim(-0.2, 5.2)
    ax.set_title("Phase diagram (MLP)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig05_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig05_phase_diagram.pdf (multi-line)")


def main():
    print("=" * 60)
    print("  fig05_combined: 正文 Figure 5 三面板重绘 (权威数据源)")
    print("=" * 60)
    ger = load_mlp_ger()
    sfull = load_mlp_sfull()
    draw_panel_ger(ger)
    draw_panel_sfull(sfull)
    draw_panel_phase(sfull)
    print("\n  输出:")
    for f in ["fig05_ger_vs_M.pdf", "fig05_sfull_vs_M.pdf", "fig05_phase_diagram.pdf"]:
        p = OUT_DIR / f
        print(f"    {'✓' if p.exists() else '✗'} {p}")


if __name__ == "__main__":
    main()

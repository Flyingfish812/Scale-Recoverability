#!/usr/bin/env python3
"""
20260725 重画 Figure S1/S2 — 修复 P0-28 (τ/5 误导线)

问题:
  Figure S1/S2 在 mean band truncation error 图上绘制 τ/5=0.01 线,
  但 robust criterion 使用的是 Q95, 而非 mean error。
  读者可能误以为 mean < 0.01 即满足 robust criterion。

修复:
  从 mean-error 图中删除 τ/5 线, 避免误导。

基础代码来自 scripts/20260723/p03_draw_figS1S2.py
数据源: results/20260720/oracle_audit_testset.json

Output: results/20260725/figS1_rdb_oracle.pdf/png, figS2_sst_oracle.pdf/png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = Path(__file__).resolve().parents[4] / "applications" / "paper_main" / "build" / "figures_raw"
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
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

BANDS = ["A4", "W4", "W3", "W2", "W1"]
TAU = 0.05

BAND_COLORS = {"A4": "#1f77b4", "W4": "#ff7f0e", "W3": "#2ca02c",
               "W2": "#d62728", "W1": "#9467bd"}
BAND_MARKERS = {"A4": "o", "W4": "s", "W3": "^", "W2": "D", "W1": "v"}

DATASETS = {
    "rdb_h5": {
        "label": "RDB (radial dam-break)",
        "out_name": "figS1_rdb_oracle",
        "ranks": [16, 32, 64, 128],
    },
    "sst_weekly": {
        "label": "SST (sea surface temperature)",
        "out_name": "figS2_sst_oracle",
        "ranks": [32, 64, 128, 256, 512],
    },
}


def load_data() -> dict:
    path = ROOT / "artifacts" / "derived" / "main" / "statistics" / "oracle_audit_testset.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return json.loads(path.read_text())


def extract_band_means(table: dict, rank: int, bands: list[str]) -> dict:
    rs = str(rank)
    entry = table.get(rs)
    if entry is None:
        return {}
    bands_data = entry.get("bands", {})
    return {b: bands_data.get(b, {}).get("mean", np.nan) for b in bands}


def draw_figure(ds_name: str, ds_cfg: dict, all_data: dict):
    summaries = {s["dataset"]: s for s in all_data["summaries"]}
    summary = summaries.get(ds_name)
    if summary is None:
        print(f"  [SKIP] No data for {ds_name}")
        return

    table = summary.get("audit_table", {})
    ranks = ds_cfg["ranks"]

    rank_data = {}
    for r in ranks:
        means = extract_band_means(table, r, BANDS)
        if any(not np.isnan(v) for v in means.values()):
            rank_data[r] = means

    if not rank_data:
        print(f"  [SKIP] No valid rank data for {ds_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for band in BANDS:
        valid_ranks = sorted([r for r in rank_data if band in rank_data[r]
                              and not np.isnan(rank_data[r][band])])
        vals = [rank_data[r][band] for r in valid_ranks]
        if valid_ranks:
            ax.plot(valid_ranks, vals,
                    marker=BAND_MARKERS[band], color=BAND_COLORS[band],
                    label=band, linewidth=1.5, markersize=6)

    # ═══════════════════════════════════════════════════════════════
    # P0-28 fix: REMOVED the τ/5=0.01 reference line.
    # The robust criterion uses Q95, not mean error — showing τ/5
    # on a mean-error plot misleads readers into thinking that
    # mean < τ/5 satisfies the robust criterion.
    # Only the τ=0.05 mean criterion line is kept.
    # ═══════════════════════════════════════════════════════════════
    ax.axhline(y=TAU, color="gray", linestyle="--", alpha=0.5,
               label=f"$\\tau={TAU}$ (mean criterion)")

    ax.set_xlabel("POD rank $r$")
    ax.set_ylabel("Mean band truncation error")
    ax.set_yscale("log")
    ax.set_title(f"{ds_cfg['label']} — POD truncation reference")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    # Mark safe rank
    sorted_ranks = sorted(rank_data.keys())
    safe_rank = None
    for r in sorted_ranks:
        bands_at_r = rank_data[r]
        if all(bands_at_r.get(b, np.inf) < TAU for b in BANDS if b in bands_at_r):
            safe_rank = r
            break

    if safe_rank is not None:
        ax.axvline(x=safe_rank, color="green", linestyle="--", alpha=0.3)
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        ax.text(safe_rank, y_top * 0.9,
                f"$r_{{\\mathrm{{mean}}}}={safe_rank}$",
                fontsize=9, color="green", ha="right", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="green", alpha=0.7, pad=2))
    else:
        ax.text(0.5, 0.02, "No rank satisfies mean criterion",
                transform=ax.transAxes, fontsize=9, ha="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()

    out_stem = OUT_DIR / ds_cfg["out_name"]
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    print(f"  [OK] {ds_cfg['out_name']}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  Redrawing S1/S2 — P0-28 fix (remove τ/5 line)")
    print("=" * 60)

    print("\n1. Loading data...")
    all_data = load_data()
    print(f"   Datasets: {[s['dataset'] for s in all_data['summaries']]}")

    print("\n2. Drawing figures...")
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n  --- {ds_cfg['label']} ---")
        draw_figure(ds_name, ds_cfg, all_data)

    print(f"\n  ✓ All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

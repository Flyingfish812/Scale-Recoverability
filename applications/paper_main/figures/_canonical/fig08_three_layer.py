#!/usr/bin/env python3
"""
P09: Figure 8 — Three-Layer Error Analysis (single panel, log y)

源脚本: scripts/20260714/generate_figures.py → fig02_three_layer()
数据源: results/20260714/three_layer_errors_full.json 的 compensation_effect

修改:
  - 标题从 "Three-Layer Error Decomposition" → "Three-Layer Error Analysis"
  - 保留单面板 + 对数纵坐标
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
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

BANDS = ["A4", "W4", "W3", "W2", "W1"]


def main():
    print("=" * 60)
    print("  P09: Figure 8 — Three-Layer Error Analysis")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1] Loading compensation_effect data...")
    with open(ROOT / "artifacts/derived/main/statistics/three_layer_errors_full.json") as f:
        data = json.load(f)
    # Check structure
    if isinstance(data, dict) and "compensation_effect" in data:
        comp = data["compensation_effect"]
    else:
        # Try to compute from the flat list
        print("  Computing compensation from raw records...")
        from collections import defaultdict
        groups = defaultdict(lambda: {"E_total": defaultdict(list), "E_trunc": defaultdict(list), "E_pred": defaultdict(list)})
        for r in data:
            if r["model_type"] == "vcnn" and r["mask_num"] == 10 and r["noise_sigma"] == 0.0:
                for b in BANDS:
                    groups["vcnn"]["E_total"][b].append(r.get(f"E_total_{b}", 0))
                    groups["vcnn"]["E_trunc"][b].append(r.get(f"E_trunc_{b}", 0))
                    groups["vcnn"]["E_pred"][b].append(r.get(f"E_pred_{b}", 0))
        n = len(next(iter(groups["vcnn"]["E_total"].values())))
        comp = {"n_samples": n}
        for b in BANDS:
            comp[b] = {
                "E_total_mean": float(np.mean(groups["vcnn"]["E_total"][b])),
                "E_trunc_mean": float(np.mean(groups["vcnn"]["E_trunc"][b])),
                "E_pred_mean": float(np.mean(groups["vcnn"]["E_pred"][b])),
            }
        print(f"  Computed from {n} VCNN M=10, σ=0 samples")

    print(f"  n_samples: {comp.get('n_samples', '?')}")
    for b in BANDS:
        print(f"  {b}: E_total={comp[b]['E_total_mean']:.4f}, "
              f"E_trunc={comp[b]['E_trunc_mean']:.4f}, "
              f"E_pred={comp[b]['E_pred_mean']:.4f}")

    # ── 2. Draw figure ────────────────────────────────────────
    print("\n[2] Drawing figure...")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(BANDS))
    w = 0.25
    etot = [max(comp[b]["E_total_mean"], 1e-6) for b in BANDS]
    etru = [max(comp[b]["E_trunc_mean"], 1e-6) for b in BANDS]
    epre = [max(comp[b]["E_pred_mean"], 1e-6) for b in BANDS]

    ax.bar(x - w, etot, w, label="$E_{\\mathrm{total}}$", color="#e74c3c", alpha=0.9)
    ax.bar(x, etru, w, label="$E_{\\mathrm{trunc}}$ (oracle)", color="#3498db", alpha=0.9)
    ax.bar(x + w, epre, w, label="$E_{\\mathrm{pred}}$ (model)", color="#9b59b6", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("Relative L2 Error (log scale)")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelet Band")
    ax.set_title(f"Three-Layer Error Analysis (VCNN, M=10, $\\sigma$=0, n={comp['n_samples']})")
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="$\\tau=0.05$")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")

    # ── 3. Save ───────────────────────────────────────────────
    out_pdf = OUT_DIR / "fig02_three_layer.pdf"
    out_png = OUT_DIR / "fig02_three_layer.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    # 发布由 publish_figures 统一处理 (fig02_three_layer → fig08_three_layer)
    print("  ✓ Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S3.10: Figure 11 — 阈值敏感性

增加 σ=0.01 面板。
展示 MLP M=20 在 τ=0.03, 0.05, 0.08 下的 S_full 分布。
两个面板: σ=0 (低噪声) 和 σ=0.01 (过渡区)
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "applications" / "paper_main" / "build" / "figures_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

TAU_VALS = [0.03, 0.05, 0.08]
TAU_LABELS = ["0.03", "0.05", "0.08"]
COLORS = ["#348ABD", "#E24A33", "#988ED5"]


def main():
    print("=" * 60)
    print("S3.10: Figure 11 — 阈值敏感性")
    print("=" * 60)

    s27_path = ROOT / "artifacts/derived/main/statistics/s27_threshold_sensitivity.json"
    if not s27_path.exists():
        print("[ERROR] S2.7 result not found")
        return

    s27 = json.loads(s27_path.read_text())
    results = s27.get("results", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for axi, (ax, sigma_title, sigma_val) in enumerate([
        (ax1, "$\\sigma=0$ (clean)", 0.0),
        (ax2, "$\\sigma=0.01$ (transition)", 0.01),
    ]):
        for ti, tau in enumerate(TAU_VALS):
            # 收集所有 M 的 mean S_full
            means = []
            for m in [10, 15, 20, 30, 50]:
                for r in results:
                    if (r["model"] == "mlp" and r["mask_num"] == m
                            and abs(r["sigma"] - sigma_val) < 1e-10
                            and abs(r["tau"] - tau) < 1e-10):
                        means.append(r["mean_S_full"])
                        break
                else:
                    means.append(0)

            ax.plot([10, 15, 20, 30, 50], means, "o-", color=COLORS[ti],
                    label=f"$\\tau={TAU_LABELS[ti]}$", linewidth=1.5, markersize=7)

        ax.set_xlabel("Number of sensors $M$")
        ax.set_ylabel("Mean $S_{\\mathrm{full}}$")
        ax.set_xticks([10, 15, 20, 30, 50])
        ax.set_ylim(-0.2, 5.2)
        ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Threshold $\\tau$")
        ax.set_title(sigma_title)

    plt.tight_layout()
    out_path = OUT_DIR / "fig11_threshold_sensitivity.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] 写入 {out_path}")
    fig.savefig(OUT_DIR / "fig11_threshold_sensitivity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 打印关键数据
    print(f"\nMLP M=20 阈值敏感性:")
    for sigma_val, sigma_label in [(0.0, "σ=0"), (0.01, "σ=0.01")]:
        print(f"  {sigma_label}:")
        for tau in TAU_VALS:
            for r in results:
                if r["model"] == "mlp" and r["mask_num"] == 20 and abs(r["sigma"] - sigma_val) < 1e-10 and abs(r["tau"] - tau) < 1e-10:
                    print(f"    τ={tau}: mean S_full={r['mean_S_full']:.2f}, P3={r['P3']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
P06: 批量重算 Ridge 影响的剩余图
  - Figure 5: 确认只含 MLP，重新输出到 results/20260723
  - sfig03:  Ridge 相位图（closed-form Ridge S_full 数据）
  - Figure 9: 能量 vs NRMSE（closed-form Ridge 逐模态 NRMSE）

用法:
  conda activate sana
  python scripts/20260723/p06_draw_fig5_sfig03_fig9.py
"""

from __future__ import annotations

import json
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
BANDS = ["A4", "W4", "W3", "W2", "W1"]

# ═══════════════════════════════════════════════════════════════════
# Figure 5 — MLP GER vs M (Ridge 不影响此图, 仅重新输出)
# ═══════════════════════════════════════════════════════════════════
def draw_fig5():
    """MLP GER + S_full vs M。不受 Ridge 影响，仅重新生成确保一致性。"""
    print("\n--- Figure 5: MLP GER+S_full vs M ---")
    s26 = json.loads((ROOT / "artifacts/derived/main/statistics/s26_pass_probability.json").read_text())
    ps_mlp = s26.get("phase_summary", {}).get("mlp", {})
    tl = json.loads((ROOT / "artifacts/derived/main/statistics/three_layer_fixed.json").read_text())
    tl_res = tl.get("results", [])

    from collections import defaultdict
    ger_by_config = defaultdict(list)
    for r in tl_res:
        if r["model_type"] == "mlp":
            ger_by_config[(r["mask_num"], r["noise_sigma"])].append(r.get("GER", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#348ABD", "#E24A33", "#988ED5", "#8EBA42"]
    markers = ["o", "s", "^", "D"]
    labels = ["0", "0.001", "0.01", "0.1"]

    for si, sigma in enumerate(SIGMA_VALS):
        means = [np.mean(ger_by_config.get((m, sigma), [0])) for m in MASK_NUMS]
        ax1.plot(MASK_NUMS, means, color=colors[si], marker=markers[si],
                 label=f"$\\sigma={labels[si]}$", linewidth=1.5, markersize=7)
    ax1.set_xlabel("Number of sensors $M$")
    ax1.set_ylabel("Mean GER")
    ax1.set_yscale("log")
    ax1.legend(title="Test noise")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(MASK_NUMS)
    ax1.set_title("Global error (MLP)")

    for si, sigma in enumerate(SIGMA_VALS):
        means = []
        for m in [str(x) for x in MASK_NUMS]:
            v = ps_mlp.get(m, {}).get(str(sigma), {})
            means.append(v.get("mean_S_full", 0) if v else 0)
        ax2.plot(MASK_NUMS, means, color=colors[si], marker=markers[si],
                 label=f"$\\sigma={labels[si]}$", linewidth=1.5, markersize=7)
    ax2.set_xlabel("Number of sensors $M$")
    ax2.set_ylabel("Mean $S_{\\mathrm{full}}$")
    ax2.legend(title="Test noise")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(MASK_NUMS)
    ax2.set_ylim(-0.2, 5.2)
    ax2.set_title("Scale recoverability (MLP)")
    ax2.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    fig.suptitle("MLP reconstructions (clean-trained, OOD noise test)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig05_ger_sfull_vs_M.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig05_ger_sfull_vs_M.png", bbox_inches="tight", dpi=150)
    print("  ✅ Figure 5 (MLP, 无 Ridge 影响)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# sfig03 — Ridge Phase Diagram (closed-form Ridge)
# ═══════════════════════════════════════════════════════════════════
def draw_sfig03():
    """Ridge 相位图: S_full 随 M×σ 变化，数据源 s05_true_ridge.json"""
    print("\n--- sfig03: Ridge Phase Diagram ---")
    with open(ROOT / "artifacts" / "derived" / "main" / "statistics" / "s05_true_ridge.json") as f:
        s05 = json.load(f)

    # 构建矩阵
    sfull_matrix = np.zeros((len(SIGMA_VALS), len(MASK_NUMS)))
    ger_matrix = np.zeros((len(SIGMA_VALS), len(MASK_NUMS)))
    for r in s05["results"]:
        mi = MASK_NUMS.index(r["mask_num"])
        si = SIGMA_VALS.index(r["sigma"])
        sfull_matrix[si, mi] = r["S_full_mean"]
        ger_matrix[si, mi] = r["GER_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左: S_full heatmap
    im1 = ax1.imshow(sfull_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=5)
    ax1.set_xticks(range(len(MASK_NUMS)))
    ax1.set_xticklabels([str(m) for m in MASK_NUMS])
    ax1.set_yticks(range(len(SIGMA_VALS)))
    ax1.set_yticklabels([str(s) for s in SIGMA_VALS])
    ax1.set_xlabel("Number of sensors $M$")
    ax1.set_ylabel("Noise $\\sigma$")
    ax1.set_title("Ridge $S_{\\mathrm{full}}$ (closed-form)")
    for i in range(len(SIGMA_VALS)):
        for j in range(len(MASK_NUMS)):
            val = sfull_matrix[i, j]
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=8, color="white" if val < 2.5 else "black")
    plt.colorbar(im1, ax=ax1, fraction=0.05, label="$S_{\\mathrm{full}}$")

    # 右: GER heatmap (log)
    im2 = ax2.imshow(np.log10(ger_matrix + 1e-12), aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(MASK_NUMS)))
    ax2.set_xticklabels([str(m) for m in MASK_NUMS])
    ax2.set_yticks(range(len(SIGMA_VALS)))
    ax2.set_yticklabels([str(s) for s in SIGMA_VALS])
    ax2.set_xlabel("Number of sensors $M$")
    ax2.set_ylabel("Noise $\\sigma$")
    ax2.set_title("Ridge $\\log_{10}(\\mathrm{GER})$ (closed-form)")
    for i in range(len(SIGMA_VALS)):
        for j in range(len(MASK_NUMS)):
            val = ger_matrix[i, j]
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=7, color="white" if val > 0.1 else "black")
    plt.colorbar(im2, ax=ax2, fraction=0.05, label="$\\log_{10}(\\mathrm{GER})$")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sfig03_ridge_phase.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sfig03_ridge_phase.png", bbox_inches="tight", dpi=150)
    print("  ✅ sfig03 Ridge Phase Diagram")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 9 — Energy vs NRMSE (closed-form Ridge)
# ═══════════════════════════════════════════════════════════════════
def compute_per_modal_nrmse(output_nchw, target_nchw, pod_basis_flat, mean_flat):
    """计算逐模态 NRMSE。output/target: (B, C, H, W) NCHW"""
    B = output_nchw.shape[0]
    N = output_nchw.shape[2] * output_nchw.shape[3] * output_nchw.shape[1]
    tgt_flat = target_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    out_flat = output_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    a_true = (tgt_flat - mean_flat[np.newaxis, :]) @ pod_basis_flat
    a_pred = (out_flat - mean_flat[np.newaxis, :]) @ pod_basis_flat
    eps = 1e-12
    numer = np.sum((a_pred - a_true) ** 2, axis=0)
    denom = np.sum(a_true ** 2, axis=0) + eps
    return np.sqrt(numer / denom)


def draw_fig9():
    """Figure 9: 能量 vs NRMSE（closed-form Ridge + MLP/VCNN from NPZ）"""
    print("\n--- Figure 9: Energy vs NRMSE ---")

    # Load POD basis
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    N = 80 * 160 * 2
    basis_flat = pod_basis_4d.reshape(128, N).T  # (N, 128)
    mean_flat = mean_field.ravel()

    # Mode energy from POD coefficients
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)
    mode_energy = np.var(full_coeffs, axis=0)
    mode_energy_norm = mode_energy / mode_energy[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))

    models_config = [
        ("MLP", "artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz", "#348ABD"),
        ("Ridge (closed-form)", None, "#E24A33"),  # computed from s05
        ("VCNN", "artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz", "#988ED5"),
    ]

    # Ridge closed-form: compute from s05 predictions
    with open(ROOT / "artifacts" / "derived" / "main" / "statistics" / "s05_true_ridge.json") as f:
        s05 = json.load(f)
    ridge_data = None
    for r in s05["results"]:
        if r["mask_num"] == 20 and r["sigma"] == 0.0:
            ridge_data = r
            break

    for idx, (name, npz_path, color) in enumerate(models_config):
        ax = axes[idx]

        if name == "Ridge (closed-form)":
            # Need to recompute Ridge predictions for per-modal NRMSE
            # s05 doesn't store per-modal data, so we load from s05's computation
            # Actually s05 doesn't store per-modal - it only has aggregate
            # For now, use AdamW Ridge NPZ (note: this is a limitation)
            # TODO: regenerate Ridge NPZ files with closed-form solution
            actual_npz = "artifacts/pod_model_sweep_nc/ridge_n0020/seed000/tests/s0000/test_raw.npz"
            data = dict(np.load(str(ROOT / actual_npz)))
            output_nchw = data["output_nchw"]
            target_nchw = data["target_nchw"]
            # Compute per-modal NRMSE
            nrmse = compute_per_modal_nrmse(output_nchw, target_nchw, basis_flat, mean_flat)
            ax.scatter(mode_energy_norm, nrmse, c=color, alpha=0.5, s=8, rasterized=True)
            ax.set_title(f"{name} (using AdamW NPZ)\nTODO: replace with closed-form", fontsize=9, color="red")
        else:
            data = dict(np.load(str(ROOT / npz_path)))
            output_nchw = data["output_nchw"]
            target_nchw = data["target_nchw"]
            nrmse = compute_per_modal_nrmse(output_nchw, target_nchw, basis_flat, mean_flat)
            ax.scatter(mode_energy_norm, nrmse, c=color, alpha=0.5, s=8, rasterized=True)
            ax.set_title(name, fontsize=10)

        # Spearman correlation
        from scipy.stats import spearmanr
        r_s, p_s = spearmanr(mode_energy_norm, nrmse)
        n_boot = 2000
        rng = np.random.RandomState(42)
        boot_r = np.zeros(n_boot)
        n_modes = len(mode_energy_norm)
        for b in range(n_boot):
            idx_b = rng.choice(n_modes, n_modes, replace=True)
            boot_r[b], _ = spearmanr(mode_energy_norm[idx_b], nrmse[idx_b])
        ci = (np.percentile(boot_r, 2.5), np.percentile(boot_r, 97.5))

        ax.text(0.05, 0.95, f"$\\rho = {r_s:.3f}$\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, pad=2))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda_j / \\lambda_1$" if idx == 1 else "")
        if idx == 0:
            ax.set_ylabel("NRMSE $e_j^{\\mathrm{NRMSE}}$")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-modal NRMSE vs relative energy ($M=20$, $\\sigma=0$)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig09_energy_vs_nrmse.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig09_energy_vs_nrmse.png", bbox_inches="tight", dpi=150)
    print("  ⚠ Figure 9: Ridge panel uses AdamW NPZ (closed-form Ridge per-modal NRMSE needs extended s05)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  P06: Remaining Ridge-affected figures")
    print("=" * 60)

    draw_fig5()       # MLP only, no Ridge impact
    draw_sfig03()     # Ridge phase diagram from s05 data
    # Figure 9 的权威实现是 fig09_energy_vs_nrmse.py (闭式 Ridge 面板);
    # 本脚本内的 draw_fig9 使用已废弃 AdamW NPZ, 不再调用 (P0-10 重构)。

    print("\n  Outputs:")
    for f in ["fig05_ger_sfull_vs_M.pdf", "sfig03_ridge_phase.pdf"]:
        p = OUT_DIR / f
        if p.exists():
            print(f"    ✅ {p}")
        else:
            print(f"    ❌ {p} (missing)")

    print("  Note: Figure 9 由 fig09_energy_vs_nrmse.py 生成 (闭式 Ridge 面板)。")

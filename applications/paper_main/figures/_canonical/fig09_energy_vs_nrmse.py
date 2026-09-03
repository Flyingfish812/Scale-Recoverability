#!/usr/bin/env python3
"""
P07: Figure 9 — Energy vs NRMSE (closed-form Ridge, 完整版)

修复: Ridge 面板使用真正闭式解的逐模态 NRMSE，不再依赖 AdamW NPZ。
三面板: MLP (NPZ) / Ridge (closed-form) / VCNN (NPZ)

数据源:
  - MLP/VCNN: artifacts 中 NPZ 文件
  - Ridge: 闭式解正规方程重算 W，应用到 M=20, σ=0 测试数据
  - POD 基: artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz

用法:
  conda activate sana
  python scripts/20260723/p07_draw_fig9.py

输出:
  results/20260723/fig09_energy_vs_nrmse.pdf / .png
"""

from __future__ import annotations

import time
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

H, W, C = 80, 160, 2
N = H * W * C
N_MODES = 128
TAU = 0.05
LAMBDA_GRID = np.logspace(-8, 2, 21)


def load_mask(mask_num: int) -> np.ndarray:
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def build_obs(fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
    obs_idx = np.argwhere(mask)
    n_obs = len(obs_idx)
    obs = np.zeros((fields.shape[0], n_obs * C), dtype=np.float64)
    for i in range(fields.shape[0]):
        obs[i] = fields[i, obs_idx[:, 0], obs_idx[:, 1], :].ravel()
    return obs


def compute_per_modal_nrmse(output_nchw, target_nchw, pod_basis_flat, mean_flat):
    """逐模态 NRMSE。output/target: NCHW (B, C, H, W)"""
    B = output_nchw.shape[0]
    tgt_flat = target_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    out_flat = output_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    a_true = (tgt_flat - mean_flat[np.newaxis, :]) @ pod_basis_flat
    a_pred = (out_flat - mean_flat[np.newaxis, :]) @ pod_basis_flat
    eps = 1e-12
    numer = np.sum((a_pred - a_true) ** 2, axis=0)
    denom = np.sum(a_true ** 2, axis=0) + eps
    return np.sqrt(numer / denom)


def main():
    print("=" * 60)
    print("  P07: Figure 9 — Energy vs NRMSE (closed-form Ridge)")
    print("=" * 60)
    t0 = time.time()

    # ── 1. Load POD basis ─────────────────────────────────────
    print("\n[1] Loading POD basis...")
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    pod_basis_flat = pod_basis_4d.reshape(N_MODES, N).T  # (N, 128)
    mean_flat = mean_field.ravel()

    # Mode energy
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)
    mode_energy = np.var(full_coeffs, axis=0)
    mode_energy_norm = mode_energy / mode_energy[0]

    # ── 2. Load full data & split ─────────────────────────────
    print("[2] Loading data and splitting...")
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))
    ref_npz = np.load(str(ROOT / "artifacts/pod_model_sweep_nc/mlp_n0010/seed000/tests/s0000/test_raw.npz"))
    test_indices = set(ref_npz["test_indices"].tolist())
    all_idx = set(range(full_fields.shape[0]))
    train_val = sorted(all_idx - test_indices)
    test_idx = sorted(test_indices)
    rng = np.random.RandomState(42)
    n_val = int(len(train_val) * 0.1)
    val_idx = set(rng.choice(train_val, n_val, replace=False))
    train_idx = [i for i in train_val if i not in val_idx]

    # ── 3. Closed-form Ridge W for M=20 ───────────────────────
    print("[3] Computing closed-form Ridge for M=20...")
    mask = load_mask(20)
    train_f = full_fields[train_idx]
    val_f = full_fields[list(val_idx)]
    train_c = pod["coefficients"][train_idx]
    val_c = pod["coefficients"][list(val_idx)]

    train_obs = build_obs(train_f, mask)
    val_obs = build_obs(val_f, mask)
    obs_mean = np.mean(train_f, axis=0).ravel()
    obs_std = np.std(train_f, axis=0).ravel() + 1e-8
    coeff_mean = np.mean(train_c, axis=0)
    coeff_std = np.std(train_c, axis=0) + 1e-8

    train_X = np.concatenate([
        (train_obs - obs_mean[:train_obs.shape[1]]) / obs_std[:train_obs.shape[1]],
        np.ones((train_obs.shape[0], 1))
    ], axis=1)
    val_X = np.concatenate([
        (val_obs - obs_mean[:val_obs.shape[1]]) / obs_std[:val_obs.shape[1]],
        np.ones((val_obs.shape[0], 1))
    ], axis=1)
    train_Y = (train_c - coeff_mean) / coeff_std
    val_Y = (val_c - coeff_mean) / coeff_std

    XTX = train_X.T @ train_X
    XTA = train_X.T @ train_Y
    d = XTX.shape[0]
    best_lambda, best_loss, best_W = None, float("inf"), None
    for lam in LAMBDA_GRID:
        I = np.eye(d); I[-1, -1] = 0.0
        W_mat = np.linalg.solve(XTX + lam * I, XTA)
        loss = np.mean((val_X @ W_mat - val_Y) ** 2)
        if loss < best_loss:
            best_loss, best_lambda, best_W = loss, lam, W_mat
    print(f"   λ*={best_lambda:.1e}, val_loss={best_loss:.6f}")

    # ── 4. Apply to test data & compute per-modal NRMSE ───────
    print("[4] Computing Ridge closed-form predictions...")
    test_f = full_fields[test_idx]
    test_obs = build_obs(test_f, mask)
    test_X = np.concatenate([
        (test_obs - obs_mean[:test_obs.shape[1]]) / obs_std[:test_obs.shape[1]],
        np.ones((test_obs.shape[0], 1))
    ], axis=1)
    pred_coeff = test_X @ best_W
    pred_coeff_norm = pred_coeff * coeff_std + coeff_mean
    pred_flat = mean_flat[np.newaxis, :] + (pred_coeff_norm @ pod_basis_flat.T)
    _H2, _W2, _C2 = int(H), int(W), int(C)
    pred_nchw = np.asarray(pred_flat).reshape(-1, _H2, _W2, _C2).transpose(0, 3, 1, 2)
    target_nchw = np.asarray(test_f).transpose(0, 3, 1, 2)

    nrmse_ridge = compute_per_modal_nrmse(pred_nchw, target_nchw, pod_basis_flat, mean_flat)
    print(f"   Ridge closed-form: NRMSE computed ({len(nrmse_ridge)} modes)")

    # ── 5. Load MLP/VCNN NPZ ──────────────────────────────────
    print("[5] Loading MLP and VCNN NPZ data...")
    def load_npz_nrmse(rel_path):
        d = dict(np.load(str(ROOT / rel_path)))
        return compute_per_modal_nrmse(d["output_nchw"], d["target_nchw"],
                                        pod_basis_flat, mean_flat)

    nrmse_mlp = load_npz_nrmse("artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz")
    nrmse_vcnn = load_npz_nrmse("artifacts/vcnn_results/vcnn_sweep_nc_2000/vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz")

    # ── 6. Draw figure ────────────────────────────────────────
    print("[6] Drawing figure...")
    models = [
        ("MLP", nrmse_mlp, "#348ABD"),
        ("Ridge (closed-form)", nrmse_ridge, "#E24A33"),
        ("VCNN", nrmse_vcnn, "#988ED5"),
    ]

    from scipy.stats import spearmanr
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))

    for idx, (name, nrmse, color) in enumerate(models):
        ax = axes[idx]
        ax.scatter(mode_energy_norm, nrmse, c=color, alpha=0.5, s=8, rasterized=True)

        # Spearman ρ + bootstrap CI
        r_s, _ = spearmanr(mode_energy_norm, nrmse)
        n_boot = 2000
        boot_r = np.zeros(n_boot)
        for b in range(n_boot):
            idx_b = rng.choice(len(mode_energy_norm), len(mode_energy_norm), replace=True)
            boot_r[b], _ = spearmanr(mode_energy_norm[idx_b], nrmse[idx_b])
        ci = (np.percentile(boot_r, 2.5), np.percentile(boot_r, 97.5))

        ax.text(0.05, 0.95,
                f"$\\rho = {r_s:.3f}$\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, pad=2))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda_j / \\lambda_1$" if idx == 1 else "")
        if idx == 0:
            ax.set_ylabel("NRMSE $e_j^{\\mathrm{NRMSE}}$")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-modal NRMSE vs relative energy ($M=20$, $\\sigma=0$)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig09_energy_vs_nrmse.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig09_energy_vs_nrmse.png", bbox_inches="tight", dpi=150)
    print(f"\n  ✅ Figure 9 saved ({time.time() - t0:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()

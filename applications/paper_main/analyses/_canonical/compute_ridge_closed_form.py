#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S01b-1: 生成闭式 Ridge 逐样本输出 NPZ (42,000 口径, P0-OI-08)。

背景: S01/S33 误用了废弃的 AdamW-Ridge NPZ (artifacts/pod_model_sweep_nc/ridge_*,
3 种子内容不同)。论文口径 = 闭式 Ridge (确定性, 20 配置 × 300 快照 = 6,000 条)。

本脚本: 用 s05_true_ridge_recompute 的闭式求解逻辑, 为 20 个 (M, σ) 配置生成
逐样本 output_nchw/target_nchw, 保存为与 MLP/VCNN 相同结构的 NPZ, 供 equal-GER
重算使用。输出目录与 pod_model_sweep_nc 平行, 避免污染原数据:

  artifacts/ridge_closed_form_sweep_nc/ridge_n{mask}/seed000/tests/{sigma_code}/test_raw.npz
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[4]

# ── Constants (同 s05) ───────────────────────────────────────────
H, W, C = 80, 160, 2
N = H * W * C          # 25600
N_MODES = 128
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
SIGMA_CODES = ["s0000", "s0010", "s0100", "s1000"]
TAU = 0.05
WAVELET = "db2"
LEVEL = 4
EPS = 1e-12
LAMBDA_GRID = np.logspace(-8, 2, 21)
NC_MEAN = np.array([1.0004944, -0.00017817653], dtype=np.float64)
NC_STD = np.array([0.21863055, 0.19121747], dtype=np.float64)

OUT_DIR = ROOT / "artifacts" / "ridge_closed_form_sweep_nc"


def load_mask(mask_num: int) -> np.ndarray:
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def build_observation_matrix(fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
    obs_indices = np.argwhere(mask)
    n_obs = len(obs_indices)
    obs = np.zeros((fields.shape[0], n_obs * C), dtype=np.float64)
    for i in range(fields.shape[0]):
        obs[i] = fields[i, obs_indices[:, 0], obs_indices[:, 1], :].ravel()
    return obs


def add_noise(fields: np.ndarray, sigma: float) -> np.ndarray:
    """与训练管线一致的归一化噪声 (RandomState(42) 确定性, 同 s05)。"""
    if sigma == 0.0:
        return fields.copy()
    phys = fields * NC_STD[np.newaxis, np.newaxis, np.newaxis, :] + NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]
    noise = np.random.RandomState(42).randn(*phys.shape).astype(np.float64) * sigma
    phys_noisy = phys + noise
    return (phys_noisy - NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]) / NC_STD[np.newaxis, np.newaxis, np.newaxis, :]


def main() -> None:
    t_start = time.time()
    print("=" * 72)
    print("  闭式 Ridge 逐样本 NPZ 生成 (20 配置 × 300 快照)")
    print("=" * 72)

    # 检查机制: 产物已存在则跳过 (--force 强制重算)
    ref_npz = OUT_DIR / "ridge_n0010" / "seed000" / "tests" / "s0000" / "test_raw.npz"
    if ref_npz.exists() and "--force" not in sys.argv:
        print(f"[skip] 闭式 Ridge NPZ 已存在 ({OUT_DIR.relative_to(ROOT)}) (--force 强制重算)")
        return 0

    # ── 1. 数据与划分 (同 s05) ────────────────────────────────
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))

    ref_npz = np.load(str(ROOT / "artifacts/pod_model_sweep_nc/mlp_n0010/seed000/tests/s0000/test_raw.npz"))
    test_indices = sorted(set(ref_npz["test_indices"].tolist()))
    all_indices = set(range(full_fields.shape[0]))
    train_val_indices = sorted(all_indices - set(test_indices))

    np.random.seed(42)
    n_train_val = len(train_val_indices)
    n_val = int(n_train_val * 0.1)
    val_idx = set(np.random.choice(train_val_indices, n_val, replace=False))
    train_idx = [i for i in train_val_indices if i not in val_idx]

    train_fields = full_fields[train_idx]
    val_fields = full_fields[list(val_idx)]
    test_fields = full_fields[test_indices]
    train_coeffs = full_coeffs[train_idx]
    val_coeffs = full_coeffs[list(val_idx)]

    obs_mean = np.mean(train_fields, axis=0).ravel()
    obs_std = np.std(train_fields, axis=0).ravel() + 1e-8
    coeff_mean = np.mean(train_coeffs, axis=0)
    coeff_std = np.std(train_coeffs, axis=0) + 1e-8

    basis_flat = pod_basis_4d.reshape(N_MODES, N).T
    mean_flat = mean_field.ravel()
    n_test = int(len(test_indices))
    target_nchw = np.asarray(test_fields).transpose(0, 3, 1, 2).astype(np.float64)
    test_idx_arr = np.asarray(test_indices, dtype=np.int64)

    # ── 2. 逐 M × σ ───────────────────────────────────────────
    for mask_num in MASK_NUMS:
        mask = load_mask(mask_num)
        n_obs = int(np.sum(mask))

        train_obs = build_observation_matrix(train_fields, mask)
        val_obs = build_observation_matrix(val_fields, mask)
        train_obs_norm = (train_obs - obs_mean[:train_obs.shape[1]]) / obs_std[:train_obs.shape[1]]
        val_obs_norm = (val_obs - obs_mean[:val_obs.shape[1]]) / obs_std[:val_obs.shape[1]]
        train_coeff_norm = (train_coeffs - coeff_mean) / coeff_std
        val_coeff_norm = (val_coeffs - coeff_mean) / coeff_std

        train_X = np.concatenate([train_obs_norm, np.ones((train_obs_norm.shape[0], 1))], axis=1)
        val_X = np.concatenate([val_obs_norm, np.ones((val_obs_norm.shape[0], 1))], axis=1)
        XTX = train_X.T @ train_X
        XTA = train_X.T @ train_coeff_norm
        d = XTX.shape[0]

        best_val_loss = float("inf")
        best_Wmat = None
        for lambda_val in LAMBDA_GRID:
            Ieye = np.eye(d)
            Ieye[-1, -1] = 0.0
            Wmat = np.linalg.solve(XTX + lambda_val * Ieye, XTA)
            loss = np.mean((val_X @ Wmat - val_coeff_norm) ** 2)
            if loss < best_val_loss:
                best_val_loss = loss
                best_Wmat = Wmat

        for sigma_val, sigma_code in zip(SIGMA_VALS, SIGMA_CODES):
            test_fields_noisy = test_fields if sigma_val == 0.0 else add_noise(test_fields, sigma_val)
            test_obs = build_observation_matrix(test_fields_noisy, mask)
            test_obs_norm = (test_obs - obs_mean[:test_obs.shape[1]]) / obs_std[:test_obs.shape[1]]
            test_X = np.concatenate([test_obs_norm, np.ones((test_obs_norm.shape[0], 1))], axis=1)

            test_pred = (test_X @ best_Wmat) * coeff_std + coeff_mean
            pred_flat = mean_flat[np.newaxis, :] + (test_pred @ basis_flat.T)
            pred_nchw = pred_flat.reshape(n_test, H, W, C).transpose(0, 3, 1, 2).astype(np.float32)
            tgt_nchw = target_nchw.astype(np.float32)

            out_npz_dir = OUT_DIR / f"ridge_n{mask_num:04d}" / "seed000" / "tests" / sigma_code
            out_npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(out_npz_dir / "test_raw.npz"),
                output_nchw=pred_nchw,
                target_nchw=tgt_nchw,
                test_indices=test_idx_arr,
                noise_sigma=np.float32(sigma_val),
            )
            print(f"  ridge_M{mask_num}_σ{sigma_val}: saved ({pred_nchw.shape}), "
                  f"λ*={best_val_loss:.3e}")

    # 元数据
    meta = {
        "task": "S01b-1",
        "description": "闭式 Ridge 逐样本输出 (42,000 口径; 替代废弃 AdamW-Ridge NPZ)",
        "n_configs": len(MASK_NUMS) * len(SIGMA_VALS),
        "n_samples_per_config": n_test,
        "total_samples": len(MASK_NUMS) * len(SIGMA_VALS) * n_test,
        "method": "closed-form Ridge with bias; λ 由 val-set 选 (grid 同 s05)",
        "data_split": {"train": len(train_idx), "val": n_val, "test": n_test},
        "noise": "RandomState(42) 确定性噪声 (同 s05)",
        "generated": "2026-08-05",
        "supersedes": "artifacts/pod_model_sweep_nc/ridge_* (AdamW, 已废弃)",
    }
    (OUT_DIR / "README.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] 生成 {len(MASK_NUMS) * len(SIGMA_VALS)} 个 NPZ → {OUT_DIR}")
    print(f"     总样本 {meta['total_samples']} (42,000 口径中的 Ridge 6,000)")
    print(f"     耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

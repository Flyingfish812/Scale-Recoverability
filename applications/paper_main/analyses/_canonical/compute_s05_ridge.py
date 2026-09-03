#!/usr/bin/env python3
"""
S05: 真正闭式 Ridge 全面重算（所有 M × σ 配置）

问题:
  当前论文的 "Ridge" 实际是用 AdamW 迭代训练的线性模型，
  其解不严格等价于闭式 Ridge 回归。反馈要求补算真正的 Ridge。

方法:
  1. 用正规方程求解 Ridge: W = (X^T X + λI)^{-1} X^T A
  2. 在验证集上选择 λ
  3. 对每个 (M, σ) 配置，用训练好的 W 在含噪观测上测试
  4. 计算 GER、S_full、per-band 小波误差等完整指标

与 AdamW-Ridge 的关键区别:
  - 闭式解直接最小化 ||A - YW||²_F + λ||W||²_F
  - AdamW 加 early stopping 不保证收敛到该最优解

输出:
  results/20260723/s05_true_ridge.json  — 完整结果
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────
H, W, C = 80, 160, 2
N = H * W * C  # 25600
N_MODES = 128
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
SIGMA_CODES = ["s0000", "s0010", "s0100", "s1000"]
BANDS = ["A4", "W4", "W3", "W2", "W1"]
TAU = 0.05
WAVELET = "db2"
LEVEL = 4
EPS = 1e-12

LAMBDA_GRID = np.logspace(-8, 2, 21)

# ── Noise parameters (matching the training pipeline)
NC_MEAN = np.array([1.0004944, -0.00017817653], dtype=np.float64)
NC_STD = np.array([0.21863055, 0.19121747], dtype=np.float64)


def load_mask(mask_num: int) -> np.ndarray:
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def build_observation_matrix(fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """从场数据提取传感器位置的观测值。fields: (B, H, W, C) NHWC"""
    obs_indices = np.argwhere(mask)
    n_obs = len(obs_indices)
    obs = np.zeros((fields.shape[0], n_obs * C), dtype=np.float64)
    for i in range(fields.shape[0]):
        vals = fields[i, obs_indices[:, 0], obs_indices[:, 1], :]
        obs[i] = vals.ravel()
    return obs


def add_noise(fields: np.ndarray, sigma: float) -> np.ndarray:
    """添加与训练管线一致的归一化噪声。"""
    if sigma == 0.0:
        return fields.copy()
    # NHWC format: fields is (B, H, W, C), NC_STD/MEAN are (C,)
    # Broadcast channels: (1, 1, 1, C)
    phys = fields * NC_STD[np.newaxis, np.newaxis, np.newaxis, :] + NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]
    noise = np.random.RandomState(42).randn(*phys.shape).astype(np.float64) * sigma
    phys_noisy = phys + noise
    # 重新标准化
    return (phys_noisy - NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]) / NC_STD[np.newaxis, np.newaxis, np.newaxis, :]


def compute_vorticity(field_2d):
    return np.gradient(np.gradient(field_2d, axis=0), axis=0) + \
           np.gradient(np.gradient(field_2d, axis=1), axis=1)


def compute_gradient_rmse(target, pred):
    gy_t = np.gradient(target, axis=0)
    gx_t = np.gradient(target, axis=1)
    gy_p = np.gradient(pred, axis=0)
    gx_p = np.gradient(pred, axis=1)
    err_x = np.sqrt(np.mean((gx_t - gx_p) ** 2))
    err_y = np.sqrt(np.mean((gy_t - gy_p) ** 2))
    return float(np.sqrt(err_x ** 2 + err_y ** 2))


def compute_full_metrics(target_nchw, pred_nchw):
    """计算与论文一致的完整指标: GER, S_full, per-band errors, vort/grad RMSE."""
    B = target_nchw.shape[0]
    gers = np.zeros(B)
    sfull_list = np.zeros(B, dtype=int)
    band_errs = {b: [] for b in BANDS}
    vort_rmses = []
    grad_rmses = []

    for i in range(B):
        # GER
        t = target_nchw[i].ravel()
        p = pred_nchw[i].ravel()
        gers[i] = float(np.linalg.norm(p - t) / (np.linalg.norm(t) + EPS))

        # Wavelet analysis on channel 0
        tgt_2d = target_nchw[i, 0]
        pred_2d = pred_nchw[i, 0]

        coeffs_tgt = pywt.wavedec2(tgt_2d, WAVELET, level=LEVEL, mode='periodization')
        coeffs_pred = pywt.wavedec2(pred_2d, WAVELET, level=LEVEL, mode='periodization')

        # A4
        a4_t = coeffs_tgt[0]
        a4_p = coeffs_pred[0]
        e_a4 = float(np.linalg.norm(a4_p - a4_t) / (np.linalg.norm(a4_t) + EPS))
        band_errs['A4'].append(e_a4)

        sfull = 1 if e_a4 < TAU else 0

        # W4-W1
        for j, (det_t, det_p) in enumerate(zip(coeffs_tgt[1:], coeffs_pred[1:])):
            err_sum = sum(np.sum((dt - dp) ** 2) for dt, dp in zip(det_t, det_p))
            norm_sum = sum(np.sum(dt ** 2) for dt in det_t)
            e = float(np.sqrt(err_sum) / (np.sqrt(norm_sum) + EPS))
            bn = BANDS[j + 1]
            band_errs[bn].append(e)
            if e < TAU:
                sfull += 1

        sfull_list[i] = sfull

        # Vorticity RMSE
        vort_t = compute_vorticity(tgt_2d)
        vort_p = compute_vorticity(pred_2d)
        vort_rmses.append(float(np.sqrt(np.mean((vort_t - vort_p) ** 2))))

        # Gradient RMSE
        grad_rmses.append(compute_gradient_rmse(tgt_2d, pred_2d))

    return {
        "GER_mean": float(np.mean(gers)),
        "GER_median": float(np.median(gers)),
        "GER_std": float(np.std(gers)),
        "S_full_mean": float(np.mean(sfull_list)),
        "S_full_median": float(np.median(sfull_list)),
        "S_full_std": float(np.std(sfull_list)),
        "per_band_mean": {b: float(np.mean(band_errs[b])) for b in BANDS},
        "per_band_median": {b: float(np.median(band_errs[b])) for b in BANDS},
        "vorticity_RMSE_mean": float(np.mean(vort_rmses)),
        "gradient_RMSE_mean": float(np.mean(grad_rmses)),
    }


def main():
    print("=" * 70)
    print("  S05: True Ridge — Closed-form recomputation (all M × σ)")
    print("=" * 70)
    t_start = time.time()

    # ── 1. Load POD basis and full data ────────────────────────
    print("\n[1] Loading POD basis and data...")
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)  # (128, H, W, C)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)  # (1501, 128)

    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))  # (1501, H, W, C)

    # ── 2. Train/val/test split ────────────────────────────────
    print("\n[2] Splitting data...")
    ref_npz = np.load(str(ROOT / "artifacts/pod_model_sweep_nc/mlp_n0010/seed000/tests/s0000/test_raw.npz"))
    test_indices = set(ref_npz["test_indices"].tolist())
    all_indices = set(range(full_fields.shape[0]))
    train_val_indices = sorted(all_indices - test_indices)
    test_idx_list = sorted(test_indices)

    np.random.seed(42)
    n_train_val = len(train_val_indices)
    n_val = int(n_train_val * 0.1)
    val_idx = set(np.random.choice(train_val_indices, n_val, replace=False))
    train_idx = [i for i in train_val_indices if i not in val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx_list)}")

    train_fields = full_fields[train_idx]
    val_fields = full_fields[list(val_idx)]
    test_fields = full_fields[test_idx_list]

    train_coeffs = full_coeffs[train_idx]
    val_coeffs = full_coeffs[list(val_idx)]
    test_coeffs = full_coeffs[test_idx_list]

    # Normalization params from training set
    obs_mean = np.mean(train_fields, axis=0).ravel()
    obs_std = np.std(train_fields, axis=0).ravel() + 1e-8
    coeff_mean = np.mean(train_coeffs, axis=0)
    coeff_std = np.std(train_coeffs, axis=0) + 1e-8

    # ── 3. For each M, compute closed-form W on clean data ─────
    print("\n[3] Computing closed-form Ridge for each M...")

    all_results = []
    all_ws = {}  # Save W for noise testing

    for mask_num in MASK_NUMS:
        mask = load_mask(mask_num)
        n_obs = np.sum(mask)
        print(f"\n  --- M={mask_num} ({n_obs} sensors) ---")

        # Clean train/val observations
        train_obs = build_observation_matrix(train_fields, mask)
        val_obs = build_observation_matrix(val_fields, mask)

        # Normalize
        train_obs_norm = (train_obs - obs_mean[:train_obs.shape[1]]) / obs_std[:train_obs.shape[1]]
        val_obs_norm = (val_obs - obs_mean[:val_obs.shape[1]]) / obs_std[:val_obs.shape[1]]
        train_coeff_norm = (train_coeffs - coeff_mean) / coeff_std
        val_coeff_norm = (val_coeffs - coeff_mean) / coeff_std

        # Add bias term
        train_X = np.concatenate([train_obs_norm, np.ones((train_obs_norm.shape[0], 1))], axis=1)
        val_X = np.concatenate([val_obs_norm, np.ones((val_obs_norm.shape[0], 1))], axis=1)

        XTX = train_X.T @ train_X
        XTA = train_X.T @ train_coeff_norm
        d = XTX.shape[0]

        # Search λ on validation set
        best_val_loss = float('inf')
        best_lambda = None
        best_W = None

        for lambda_val in LAMBDA_GRID:
            I = np.eye(d)
            I[-1, -1] = 0.0  # No regularization on bias
            W = np.linalg.solve(XTX + lambda_val * I, XTA)
            val_pred = val_X @ W
            val_loss = np.mean((val_pred - val_coeff_norm) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lambda = lambda_val
                best_W = W

        all_ws[mask_num] = best_W
        print(f"  λ*={best_lambda:.1e}, val_loss={best_val_loss:.6f}")

        # Test on all σ levels
        for sigma_val, sigma_code in zip(SIGMA_VALS, SIGMA_CODES):
            # Build test observations (potentially noisy)
            if sigma_val == 0.0:
                test_fields_noisy = test_fields.copy()
            else:
                test_fields_noisy = add_noise(test_fields, sigma_val)

            test_obs = build_observation_matrix(test_fields_noisy, mask)
            test_obs_norm = (test_obs - obs_mean[:test_obs.shape[1]]) / obs_std[:test_obs.shape[1]]
            test_X = np.concatenate([test_obs_norm, np.ones((test_obs_norm.shape[0], 1))], axis=1)

            # Predict
            test_pred_norm = test_X @ best_W
            test_pred = test_pred_norm * coeff_std + coeff_mean

            # Reconstruct field
            mean_flat = mean_field.ravel()
            n_test = len(test_idx_list)
            basis_flat = pod_basis_4d.reshape(int(N_MODES), int(N)).T

            test_fields_flat = np.asarray(test_fields).reshape(int(n_test), -1)
            pred_flat = mean_flat[np.newaxis, :] + (test_pred @ basis_flat.T)
            _H, _W, _C = 80, 160, 2
            pred_nchw = pred_flat.reshape(int(n_test), int(_H), int(_W), int(_C)).transpose(0, 3, 1, 2)  # NHWC → NCHW

            # Metrics
            # target is original test fields (clean ground truth)
            target_nchw = np.asarray(test_fields).transpose(0, 3, 1, 2).astype(np.float64)  # NHWC → NCHW

            metrics = compute_full_metrics(target_nchw, pred_nchw)
            metrics["mask_num"] = mask_num
            metrics["n_obs"] = int(n_obs)
            metrics["sigma"] = sigma_val
            metrics["sigma_code"] = sigma_code
            metrics["best_lambda"] = float(best_lambda)

            all_results.append(metrics)

            print(f"    σ={sigma_val}: "
                  f"GER={metrics['GER_mean']:.6f}, "
                  f"S_full={metrics['S_full_mean']:.2f}, "
                  f"W1={metrics['per_band_mean']['W1']:.4f}")

    # ── 4. Save results ────────────────────────────────────────
    print("\n[4] Saving results...")
    output = {
        "task": "S05_true_ridge",
        "description": "Closed-form Ridge recomputed for all M × σ configurations",
        "method": "W = (X^T X + λI)^{-1} X^T A, bias unregularized, λ selected via val set",
        "data_split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx_list)},
        "lambda_grid": LAMBDA_GRID.tolist(),
        "results": all_results,
    }
    out_path = OUT_DIR / "s05_true_ridge.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")

    # ── 5. Summary table ───────────────────────────────────────
    print("\n[5] Summary — True Ridge GER (all M × σ):")
    print(f"  {'M':>5} | {'σ=0':>10} {'σ=0.001':>10} {'σ=0.01':>10} {'σ=0.1':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}")
    for mask_num in MASK_NUMS:
        row = f"  {mask_num:>5} |"
        for sigma_val in SIGMA_VALS:
            for r in all_results:
                if r["mask_num"] == mask_num and r["sigma"] == sigma_val:
                    row += f" {r['GER_mean']:>9.4f}"
                    break
        print(row)

    print(f"\n  Comparison with AdamW Ridge (M=20, σ=0):")
    for r in all_results:
        if r["mask_num"] == 20 and r["sigma"] == 0.0:
            print(f"    Closed-form Ridge: GER={r['GER_mean']:.6f}, S_full={r['S_full_mean']:.2f}")
    print(f"    AdamW Ridge: GER≈0.0171, S_full≈1.14 (from three_layer_fixed)")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  ✓ Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
S2.1: Gappy POD 完整重算 — 修正后 (rank≤M, 验证集选 rank) 的每频带小波误差。

计算:
  - 对每个 (mask, sigma) 配置, 用修正后 Gappy POD 重建测试场
  - 计算每个频带的相对 L2 误差 (与小波 oracle 相同的口径)
  - 计算 delta = E_gappy - E_oracle
  - 输出供 tab_delta.tex 使用的数值

输出: results/20260722/s021_gappy_pod_wavelet.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W, C = 80, 160, 2
N = H * W * C
N_MODES = 128
BANDS = ["A4", "W4", "W3", "W2", "W1"]
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
CANDIDATE_RANKS = [4, 8, 12, 16, 20, 24, 32]


def load_mask(mask_num: int) -> np.ndarray:
    """加载传感器掩码, 返回 (H, W) bool 矩阵"""
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def compute_band_errors_single(pred_2d: np.ndarray, tgt_2d: np.ndarray,
                                wavelet: str = 'db2', level: int = 4) -> dict:
    """
    计算单个样本的每频带相对 L2 误差。

    Args:
        pred_2d: (H, W) 预测场
        tgt_2d:  (H, W) 目标场

    Returns:
        {band: relative_L2_error}
    """
    coeffs_pred = pywt.wavedec2(pred_2d, wavelet, level=level, mode='periodization')
    coeffs_tgt = pywt.wavedec2(tgt_2d, wavelet, level=level, mode='periodization')

    errors = {}

    # A4
    a4_p = coeffs_pred[0]
    a4_t = coeffs_tgt[0]
    errors['A4'] = float(np.linalg.norm(a4_p - a4_t) / (np.linalg.norm(a4_t) + 1e-12))

    # W4, W3, W2, W1
    band_names = ['W4', 'W3', 'W2', 'W1']
    for j, (det_p, det_t) in enumerate(zip(coeffs_pred[1:], coeffs_tgt[1:])):
        err_sum = 0.0
        norm_sum = 0.0
        for dp, dt in zip(det_p, det_t):
            err_sum += np.sum((dp - dt) ** 2)
            norm_sum += np.sum(dt ** 2)
        errors[band_names[j]] = float(np.sqrt(err_sum) / (np.sqrt(norm_sum) + 1e-12))

    return errors


def compute_oracle_errors(test_fields_hwc, pod_basis_4d, mean_field,
                          rank: int = 128) -> dict:
    """
    计算 POD oracle 的每频带误差 (rank=128, 全精度)。
    仅使用 channel 0 进行小波分析 (与论文一致)。

    Returns:
        {band: mean_relative_L2_error}
    """
    mean_flat = mean_field.ravel()
    basis_flat = pod_basis_4d[:rank].reshape(rank, N).T  # (N, rank)

    band_error_sum = {b: 0.0 for b in BANDS}
    n_test = test_fields_hwc.shape[0]

    for i in range(n_test):
        u_flat = test_fields_hwc[i].ravel()
        a_true = basis_flat.T @ (u_flat - mean_flat)  # (rank,)
        
        # Oracle reconstruction (全通道)
        recon_flat = mean_flat + basis_flat @ a_true
        
        # 提取 channel 0 进行 2D 小波分析
        recon_2d = recon_flat.reshape(H, W, C)[:, :, 0]
        tgt_2d = test_fields_hwc[i, :, :, 0]

        errors = compute_band_errors_single(recon_2d, tgt_2d)
        for b in BANDS:
            band_error_sum[b] += errors[b]

    return {b: band_error_sum[b] / n_test for b in BANDS}


def main():
    print("=" * 70)
    print("  S2.1: Gappy POD 完整重算 (rank≤M, val-select) + 小波每频带误差")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)  # (128, H, W, C)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)   # (H, W, C)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)  # (1501, 128)
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))  # (1501, H, W, C)

    # 2. 与 s23 相同的 train/val/test 划分
    print("[2] 划分数据集...")
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

    print(f"  训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx_list)}")

    train_fields = full_fields[train_idx]
    val_fields = full_fields[list(val_idx)]
    test_fields = full_fields[test_idx_list]

    # 3. 计算 oracle 误差 (M=20, σ=0 配置)
    print("[3] 计算 oracle 每频带误差...")
    oracle_errors = compute_oracle_errors(test_fields, pod_basis_4d, mean_field)
    for b in BANDS:
        print(f"  Oracle {b}: {oracle_errors[b]:.6f}")

    # 4. 遍历所有 mask 和 noise
    print("[4] 遍历配置计算 Gappy POD...")
    
    all_results = []
    pred_fields_cache = {}  # 按 (mask, sigma) 缓存重建场

    for mask_num in MASK_NUMS:
        print(f"\n--- M={mask_num} ---")
        mask = load_mask(mask_num)
        obs_indices = np.argwhere(mask)
        obs_rows = obs_indices[:, 0]
        obs_cols = obs_indices[:, 1]
        n_obs = len(obs_indices)
        
        available_ranks = [r for r in CANDIDATE_RANKS if r <= n_obs]
        if not available_ranks:
            available_ranks = [n_obs]

        print(f"  n_obs={n_obs}, candidate ranks: {available_ranks}")

        # 提取观测值
        def get_obs(fields_hwc):
            return fields_hwc[:, obs_rows, obs_cols, :].reshape(len(fields_hwc), n_obs * C)

        train_obs = get_obs(train_fields)
        val_obs = get_obs(val_fields)
        test_obs = get_obs(test_fields)
        obs_mean_flat = mean_field[obs_rows, obs_cols, :].ravel()

        mean_flat = mean_field.ravel()
        N = H * W * C

        for sigma_val in SIGMA_VALS:
            # 加噪
            rng = np.random.RandomState(42)
            val_obs_noisy = val_obs + rng.normal(0, sigma_val, val_obs.shape).astype(np.float64)
            test_obs_noisy = test_obs + rng.normal(0, sigma_val, test_obs.shape).astype(np.float64)

            # 验证集上选 rank
            best_val_ger = float('inf')
            best_rank = None

            for rank in available_ranks:
                phi_obs = pod_basis_4d[:rank, obs_rows, obs_cols, :].reshape(rank, n_obs * C).T  # (n_obs*C, rank)
                y_centered_val = val_obs_noisy - obs_mean_flat[np.newaxis, :]
                phi_pinv = np.linalg.pinv(phi_obs)
                a_pred_val = y_centered_val @ phi_pinv.T

                # 重建验证场
                basis_flat = pod_basis_4d[:rank].reshape(rank, N).T
                pred_fields_val = np.zeros((len(val_idx), H, W, C), dtype=np.float64)
                for i in range(len(val_idx)):
                    pred_flat = mean_flat + a_pred_val[i] @ basis_flat.T
                    pred_fields_val[i] = pred_flat.reshape(H, W, C)

                val_ger = float(np.mean([
                    np.linalg.norm((pred_fields_val[i] - val_fields[i]).ravel())
                    / (np.linalg.norm(val_fields[i].ravel()) + 1e-12)
                    for i in range(len(val_idx))
                ]))
                if val_ger < best_val_ger:
                    best_val_ger = val_ger
                    best_rank = rank

            # 用最优 rank 重建测试场
            phi_obs_best = pod_basis_4d[:best_rank, obs_rows, obs_cols, :].reshape(best_rank, n_obs * C).T
            y_centered_test = test_obs_noisy - obs_mean_flat[np.newaxis, :]
            phi_pinv_best = np.linalg.pinv(phi_obs_best)
            a_pred_test = y_centered_test @ phi_pinv_best.T

            basis_flat_best = pod_basis_4d[:best_rank].reshape(best_rank, N).T

            # 重建测试场 + 计算每频带小波误差
            band_error_sum = {b: 0.0 for b in BANDS}
            ger_sum = 0.0

            for i in range(len(test_idx_list)):
                pred_flat = mean_flat + a_pred_test[i] @ basis_flat_best.T
                # 提取 channel 0 进行 2D 小波分析
                pred_2d = pred_flat.reshape(H, W, C)[:, :, 0]
                tgt_2d = test_fields[i, :, :, 0]

                # GER (全通道)
                ger_sum += np.linalg.norm(pred_flat - test_fields[i].ravel()) / \
                           (np.linalg.norm(test_fields[i].ravel()) + 1e-12)

                # 每频带误差 (channel 0 only)
                errors = compute_band_errors_single(pred_2d, tgt_2d)
                for b in BANDS:
                    band_error_sum[b] += errors[b]

            n_test = len(test_idx_list)
            mean_band_errors = {b: band_error_sum[b] / n_test for b in BANDS}
            mean_ger = ger_sum / n_test

            # 计算 delta (vs oracle)
            delta = {b: mean_band_errors[b] - oracle_errors[b] for b in BANDS}

            result = {
                "mask_num": mask_num,
                "sigma": sigma_val,
                "selected_rank": best_rank,
                "val_ger": best_val_ger,
                "test_ger_mean": mean_ger,
                "oracle_errors": oracle_errors,
                "total_errors": mean_band_errors,
                "delta": delta,
                "n_test": n_test,
            }
            all_results.append(result)

            print(f"  σ={sigma_val}: rank={best_rank}, GER={mean_ger:.6f}")
            print(f"    delta: {', '.join(f'{b}={delta[b]:+.4f}' for b in BANDS)}")

    # 5. 汇总输出
    report = {
        "task": "S2.1",
        "description": "Gappy POD 修正后完整重算 (rank≤M, val-select) + 每频带小波误差",
        "method": "â = pinv(C·Φ_r)·(y - C·ū), rank ∈ {4,8,12,16,20,24,32} ∩ {r:r≤M}",
        "wavelet": {"family": "db2", "level": 4, "mode": "periodization"},
        "data_split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx_list),
        },
        "oracle_errors": oracle_errors,
        "results": all_results,
        "representative": None,
    }

    # 提取 M=20, σ=0 的代表值
    for r in all_results:
        if r["mask_num"] == 20 and r["sigma"] == 0.0:
            report["representative"] = {
                "config": "M=20, σ=0",
                "selected_rank": r["selected_rank"],
                "GER": r["test_ger_mean"],
                "per_band_total_error": r["total_errors"],
                "delta_excess_error": r["delta"],
            }
            print(f"\n  [代表配置 M=20, σ=0]")
            print(f"    选定 rank: {r['selected_rank']}")
            print(f"    GER: {r['test_ger_mean']:.6f}")
            print(f"    delta excess error:")
            for b in BANDS:
                print(f"      {b}: {r['delta'][b]:+.4f}")

    # 写入
    out_path = OUT_DIR / "s021_gappy_pod_wavelet.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 写入 {out_path}")

    # 更新修改指南中的 delta 值
    print("\n[更新建议] 用于 paper_facts.yaml delta_excess_band_error.gappy:")
    if report["representative"]:
        d = report["representative"]["delta_excess_error"]
        print(f"    gappy: {{A4: {d['A4']:.4f}, W4: {d['W4']:.4f}, "
              f"W3: {d['W3']:.4f}, W2: {d['W2']:.4f}, W1: {d['W1']:.4f}}}")


if __name__ == "__main__":
    main()

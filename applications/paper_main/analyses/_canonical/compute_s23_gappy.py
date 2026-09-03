#!/usr/bin/env python3
"""S2.3: Gappy POD 修正 — rank ≤ M + 验证集选 rank

实现:
  â = (C_M Φ_r)^† (y - C_M ū)
  限制 r ≤ M，在验证集上从 {4,8,12,16,20,24,32} ∩ {r:r≤M} 选择 rank

输出: results/20260721/s23_gappy_pod_fixed.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W, C = 80, 160, 2
N = H * W * C
N_MODES = 128
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_CODES = ['s0000', 's0010', 's0100', 's1000']
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
BANDS = ["A4", "W4", "W3", "W2", "W1"]

CANDIDATE_RANKS = [4, 8, 12, 16, 20, 24, 32]


def load_mask(mask_num: int) -> np.ndarray:
    """加载传感器掩码，返回 (H, W) 的 bool 矩阵"""
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def main():
    print("=" * 60)
    print("S2.3: Gappy POD 修正")
    print("=" * 60)

    # 1. 加载 POD 基
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)  # (128, H, W, C)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)  # (H, W, C)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)  # (1501, 128)

    # 加载数据
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))  # (1501, H, W, C)

    # 2. 使用与 S2.2 相同的 train/val/test 划分
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

    print(f"\n数据划分: 训练 {len(train_idx)}, 验证 {len(val_idx)}, 测试 {len(test_idx_list)}")

    # NHWC 格式
    train_fields = full_fields[train_idx]
    val_fields = full_fields[list(val_idx)]
    test_fields = full_fields[test_idx_list]

    # 3. 遍历所有 mask 和 noise
    all_results = []

    for mask_num in MASK_NUMS:
        mask = load_mask(mask_num)
        obs_indices = np.argwhere(mask)  # (n_obs, 2)
        n_obs = len(obs_indices)

        # 可用的 rank 候选 (r ≤ M)
        available_ranks = [r for r in CANDIDATE_RANKS if r <= n_obs]
        if not available_ranks:
            available_ranks = [n_obs]

        print(f"\n--- M={mask_num} ({n_obs} sensors), candidate ranks: {available_ranks} ---")

        # 提取观测位置的 POD 基和均值
        # Φ_r 在观测位置: pod_basis_4d[:r, obs_rows, obs_cols, :]
        # 形状: (r, n_obs, C) → 展平为 (r, n_obs*C)
        obs_rows = obs_indices[:, 0]
        obs_cols = obs_indices[:, 1]
        obs_mean = mean_field[obs_rows, obs_cols, :]  # (n_obs, C)

        # 提取观测值
        def get_obs(fields_hwc):
            """从 NHWC 场数据提取观测值"""
            # fields: (B, H, W, C)
            return fields_hwc[:, obs_rows, obs_cols, :].reshape(len(fields_hwc), n_obs * C)

        train_obs = get_obs(train_fields)
        val_obs = get_obs(val_fields)
        test_obs = get_obs(test_fields)
        obs_mean_flat = obs_mean.ravel()  # (n_obs*C,)

        # 对每个噪声水平，在验证集上选择最优 rank
        for sigma_code, sigma_val in zip(SIGMA_CODES, SIGMA_VALS):
            # 添加噪声到观测值
            rng = np.random.RandomState(42)
            val_obs_noisy = val_obs + rng.normal(0, sigma_val, val_obs.shape).astype(np.float64)
            test_obs_noisy = test_obs + rng.normal(0, sigma_val, test_obs.shape).astype(np.float64)

            best_val_ger = float('inf')
            best_rank = None
            best_a_pred = None

            for rank in available_ranks:
                # POD 基在观测位置: (rank, n_obs*C)
                phi_obs = pod_basis_4d[:rank, obs_rows, obs_cols, :]  # (rank, n_obs, C)
                phi_obs_flat = phi_obs.reshape(rank, n_obs * C).T  # (n_obs*C, rank)

                # 验证集: â = pinv(C Φ_r) (y - C ū)
                # C Φ_r = phi_obs_flat (已包含 C 和 Φ_r)
                # y - C ū = val_obs_noisy - obs_mean_flat
                y_centered_val = val_obs_noisy - obs_mean_flat[np.newaxis, :]  # (n_val, n_obs*C)
                y_centered_test = test_obs_noisy - obs_mean_flat[np.newaxis, :]

                # 伪逆
                phi_pinv = np.linalg.pinv(phi_obs_flat)  # (rank, n_obs*C)

                # POD 系数预测
                a_pred_val = y_centered_val @ phi_pinv.T  # (n_val, rank)
                a_pred_test = y_centered_test @ phi_pinv.T  # (n_test, rank)

                # 用验证集 GER 选择 rank
                pred_fields_val = np.zeros((len(val_idx), H, W, C), dtype=np.float64)
                mean_flat = mean_field.ravel()
                basis_flat = pod_basis_4d[:rank].reshape(rank, N).T

                for i in range(len(val_idx)):
                    coeffs_i = np.zeros(N_MODES, dtype=np.float64)
                    coeffs_i[:rank] = a_pred_val[i]
                    pred_flat = mean_flat + coeffs_i[:rank] @ basis_flat.T
                    pred_fields_val[i] = pred_flat.reshape(H, W, C)

                # 计算 val GER
                val_ger = np.mean([
                    np.linalg.norm((pred_fields_val[i] - val_fields[i]).ravel())
                    / (np.linalg.norm(val_fields[i].ravel()) + 1e-12)
                    for i in range(len(val_idx))
                ])

                if val_ger < best_val_ger:
                    best_val_ger = val_ger
                    best_rank = rank
                    best_a_pred_test = a_pred_test

            # 用最优 rank 计算测试指标
            phi_obs_best = pod_basis_4d[:best_rank, obs_rows, obs_cols, :]
            phi_obs_flat_best = phi_obs_best.reshape(best_rank, n_obs * C).T
            phi_pinv_best = np.linalg.pinv(phi_obs_flat_best)

            y_centered_test = test_obs_noisy - obs_mean_flat[np.newaxis, :]
            a_pred_test = y_centered_test @ phi_pinv_best.T
            a_true_test = full_coeffs[test_idx_list, :best_rank]

            # 计算 NRMSE
            eps = 1e-12
            numer = np.sum((a_pred_test - a_true_test) ** 2, axis=0)
            denom = np.sum(a_true_test ** 2, axis=0) + eps
            nrmse = np.sqrt(numer / denom)

            # 重建场
            mean_flat = mean_field.ravel()
            basis_flat = pod_basis_4d[:best_rank].reshape(best_rank, N).T
            pred_fields_test = np.zeros((len(test_idx_list), H, W, C), dtype=np.float64)
            ger_per_sample = np.zeros(len(test_idx_list))

            for i in range(len(test_idx_list)):
                pred_flat = mean_flat + a_pred_test[i] @ basis_flat.T
                pred_fields_test[i] = pred_flat.reshape(H, W, C)
                tgt = test_fields[i].ravel()
                prd = pred_flat
                ger_per_sample[i] = np.linalg.norm(tgt - prd) / (np.linalg.norm(tgt) + eps)

            result = {
                "mask_num": mask_num,
                "n_obs": int(n_obs),
                "sigma": sigma_val,
                "selected_rank": best_rank,
                "val_ger": float(best_val_ger),
                "test_ger_mean": float(np.mean(ger_per_sample)),
                "test_ger_median": float(np.median(ger_per_sample)),
                "nrmse_mean": float(np.mean(nrmse)),
                "nrmse_median": float(np.median(nrmse)),
                "n_test": int(len(test_idx_list)),
            }
            all_results.append(result)
            print(f"  σ={sigma_val}: selected rank={best_rank}, GER={result['test_ger_mean']:.6f}, NRMSE={result['nrmse_mean']:.4f}")

    # 4. 输出
    output = {
        "task": "S2.3",
        "description": "Gappy POD 修正 — rank ≤ M, 验证集选 rank",
        "candidate_ranks": CANDIDATE_RANKS,
        "data_split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx_list),
        },
        "results": all_results,
    }

    out_path = OUT_DIR / "s23_gappy_pod_fixed.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] 写入 {out_path}")

    # 与旧版 Gappy POD 比较
    print(f"\n{'='*60}")
    print("与旧版 Gappy POD (rank=32 固定) 比较")
    print(f"{'='*60}")
    try:
        old_gappy = json.loads((ROOT / "artifacts/derived/main/statistics/gappy_pod_baseline.json").read_text())
        for new_r in all_results:
            if new_r["sigma"] == 0.0:
                for old_r in old_gappy.get("results", []):
                    if old_r["mask_num"] == new_r["mask_num"] and old_r["noise_sigma"] == 0.0:
                        print(f"  M={new_r['mask_num']}: 旧 rank=32 GER={old_r['GER_mean']:.4f} → "
                              f"新 rank={new_r['selected_rank']} GER={new_r['test_ger_mean']:.4f}")
    except Exception as e:
        print(f"  无法加载旧结果: {e}")


if __name__ == "__main__":
    main()

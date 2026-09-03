#!/usr/bin/env python3
"""
S10: 重算 Table 12 (Decile, 方案A 128模式分组) 和 Table 13 (Recovery Rates, closed-form Ridge)

Table 12 — 方案A:
  将 128 个 SST POD 模态按能量降序排列，分为 10 个 decile:
    - Deciles 1-8: 每个 13 个模式 (8 × 13 = 104)
    - Deciles 9-10: 每个 12 个模式 (2 × 12 = 24)
    - Total = 128
  计算每个 decile 的 energy_range, NRMSE mean/median/Q25/Q75, energy_share。

Table 13 — 正确数据:
  从 three_layer_errors_full.json 取 MLP (18k) + VCNN (18k) = 36k records，
  对 Ridge 用闭式解计算 per-sample band errors (E_total_*)，阈值 τ=0.05 判定通过率。
  最终 recovery rate = 全部记录中 E_total_b ≤ 0.05 的比例。

输出:
  results/20260723/s10_tables_12_13.json
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

# Normalization params (from training pipeline)
NC_MEAN = np.array([1.0004944, -0.00017817653], dtype=np.float64)
NC_STD = np.array([0.21863055, 0.19121747], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
#  Part A: 工具函数 (复用 s05)
# ═══════════════════════════════════════════════════════════════════

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
        vals = fields[i, obs_indices[:, 0], obs_indices[:, 1], :]
        obs[i] = vals.ravel()
    return obs


def add_noise(fields: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return fields.copy()
    phys = fields * NC_STD[np.newaxis, np.newaxis, np.newaxis, :] + NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]
    noise = np.random.RandomState(42).randn(*phys.shape).astype(np.float64) * sigma
    phys_noisy = phys + noise
    return (phys_noisy - NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]) / NC_STD[np.newaxis, np.newaxis, np.newaxis, :]


def compute_band_errors(target_nchw: np.ndarray, pred_nchw: np.ndarray) -> dict:
    """计算每个样本的逐频带相对误差 E_total_b。

    返回 {band: np.ndarray of shape (B,)}。
    """
    B = target_nchw.shape[0]
    band_errs = {b: np.zeros(B, dtype=np.float64) for b in BANDS}

    for i in range(B):
        tgt_2d = target_nchw[i, 0]
        pred_2d = pred_nchw[i, 0]

        coeffs_tgt = pywt.wavedec2(tgt_2d, WAVELET, level=LEVEL, mode='periodization')
        coeffs_pred = pywt.wavedec2(pred_2d, WAVELET, level=LEVEL, mode='periodization')

        # A4
        a4_t = coeffs_tgt[0]
        a4_p = coeffs_pred[0]
        band_errs['A4'][i] = float(np.linalg.norm(a4_p - a4_t) / (np.linalg.norm(a4_t) + EPS))

        # W4-W1
        for j, (det_t, det_p) in enumerate(zip(coeffs_tgt[1:], coeffs_pred[1:])):
            err_sum = sum(np.sum((dt - dp) ** 2) for dt, dp in zip(det_t, det_p))
            norm_sum = sum(np.sum(dt ** 2) for dt in det_t)
            bn = BANDS[j + 1]
            band_errs[bn][i] = float(np.sqrt(err_sum) / (np.sqrt(norm_sum) + EPS))

    return band_errs


# ═══════════════════════════════════════════════════════════════════
#  Part A: Table 12 — Decile 重算 (方案A: 13×8 + 12×2 = 128)
# ═══════════════════════════════════════════════════════════════════

def recompute_decile_table() -> dict:
    """从 s21_nrmse_full.json 重算 decile 表，使用方案A分组。

    方案A: 128 modes → 10 deciles
      - Deciles 1-8: 每组 13 modes (8×13=104)
      - Deciles 9-10: 每组 12 modes (2×12=24)
    按能量降序排列后分组。
    """
    print("=" * 60)
    print("  Table 12: Decile 重算 (方案A: 13×8 + 12×2 = 128)")
    print("=" * 60)

    # 加载 s21 完整结果
    s21_path = ROOT / "artifacts" / "derived" / "main" / "statistics" / "s21_nrmse_full.json"
    if not s21_path.exists():
        print(f"  [ERROR] {s21_path} not found!")
        return {}

    with open(s21_path) as f:
        s21 = json.load(f)

    mode_energy_norm = np.array(s21["mode_energy_norm"])
    config_results = s21["config_results"]

    # 找 MLP M=20 σ=0 的代表性配置
    rep_config = None
    for r in config_results:
        if r["model_type"] == "mlp" and r["mask_num"] == 20 and r.get("sigma_val") == 0.0:
            rep_config = r
            break

    if rep_config is None:
        print("  [ERROR] MLP M=20 σ=0 not found in s21 results!")
        return {}

    nrmse_per_mode = np.array(rep_config["nrmse_per_mode"])

    # 方案A: 按能量降序排列，13×8 + 12×2 分组
    energy_order = np.argsort(mode_energy_norm)[::-1]  # 高→低

    # 构建分组: deciles 1-8 → 13, deciles 9-10 → 12
    decile_sizes = [13] * 8 + [12] * 2  # = 128
    assert sum(decile_sizes) == N_MODES, f"Decile sizes sum to {sum(decile_sizes)}, expected {N_MODES}"

    decile_table = []
    start = 0
    for d, size in enumerate(decile_sizes, 1):
        idx = energy_order[start:start + size]
        # Energy range (转换为百分比)
        energy_min = float(mode_energy_norm[idx].min() * 100)
        energy_max = float(mode_energy_norm[idx].max() * 100)
        # NRMSE stats
        nrmse_mean = float(np.mean(nrmse_per_mode[idx]))
        nrmse_median = float(np.median(nrmse_per_mode[idx]))
        nrmse_q25 = float(np.percentile(nrmse_per_mode[idx], 25))
        nrmse_q75 = float(np.percentile(nrmse_per_mode[idx], 75))
        # Energy share within this decile
        energy_share_pct = float(mode_energy_norm[idx].sum() * 100)

        decile_table.append({
            "decile": d,
            "n_modes": size,
            "energy_range_pct": [energy_min, energy_max],
            "energy_share_pct": energy_share_pct,
            "nrmse_mean": nrmse_mean,
            "nrmse_median": nrmse_median,
            "nrmse_q25": nrmse_q25,
            "nrmse_q75": nrmse_q75,
        })
        start += size

        print(f"  d{d:>2}: n_modes={size}, energy=[{energy_min:.6f}, {energy_max:.6f}]%, "
              f"share={energy_share_pct:.4f}%, "
              f"NRMSE: mean={nrmse_mean:.4f}, median={nrmse_median:.4f} "
              f"[Q25={nrmse_q25:.4f}, Q75={nrmse_q75:.4f}]")

    # 验证
    total_energy = sum(d["energy_share_pct"] for d in decile_table)
    print(f"\n  Total energy: {total_energy:.2f}% (should be ~100%)")
    print(f"  Bottom 40% modes energy share: "
          f"{sum(d['energy_share_pct'] for d in decile_table if d['decile'] > 6):.4f}%")

    # 准备输出格式 (类似 paper_facts.yaml 格式)
    rows = {}
    for d in decile_table:
        di = d["decile"]
        # energy_share 格式化为字符串
        es = d["energy_share_pct"]
        if es < 0.01:
            es_str = "<0.01"
        elif es < 0.1:
            es_str = f"~{es:.3f}".rstrip('0').rstrip('.')
        elif es < 1:
            es_str = f"~{es:.2f}".rstrip('0').rstrip('.')
        elif es < 10:
            es_str = f"~{es:.2f}".rstrip('0').rstrip('.')
        else:
            es_str = f"~{es:.2f}".rstrip('0').rstrip('.')

        rows[f"d{di}"] = [
            d["energy_range_pct"],
            d["nrmse_mean"],
            d["nrmse_median"],
            d["nrmse_q25"],
            d["nrmse_q75"],
            es_str,
        ]

    result = {
        "desc": f"模态级 NRMSE 按能量十分位分布 (MLP, M=20, σ=0, {N_MODES} modes, 方案A: {decile_sizes})",
        "columns": ["energy_range_pct", "nrmse_mean", "nrmse_median", "nrmse_q25", "nrmse_q75", "energy_share_pct"],
        "rows": rows,
        "decile_sizes": decile_sizes,
        "total_energy_pct": round(total_energy, 4),
        "bottom_40pct_energy": f"{sum(d['energy_share_pct'] for d in decile_table if d['decile'] > 6):.4f}%",
        "bottom_40pct_mean_error": f"{np.mean([d['nrmse_mean'] for d in decile_table if d['decile'] > 6]):.4f}",
    }

    # 也保存供后续使用的结构化数据
    result["_deciles_structured"] = decile_table

    return result


# ═══════════════════════════════════════════════════════════════════
#  Part B: Table 13 — Recovery Rates 重算 (closed-form Ridge)
# ═══════════════════════════════════════════════════════════════════

def compute_ridge_closed_form_recovery() -> dict:
    """用闭式 Ridge 重算所有 20 个配置的 per-sample band errors。

    输出每个 (mask, sigma) 配置的:
      - mask_num, sigma, n_test
      - per-sample E_total_A4, W4, W3, W2, W1 (list of float)
      - pass summary per band
    """
    print("\n" + "=" * 60)
    print("  Closed-form Ridge: per-sample band errors")
    print("=" * 60)

    # ── 1. Load POD basis and full data ────────────────────────
    print("\n  [1] Loading POD basis and data...")
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)  # (1501, 128)
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))  # (1501, H, W, C)

    # ── 2. Train/val/test split ────────────────────────────────
    print("  [2] Train/val/test split...")
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

    print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx_list)}")

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

    n_test = len(test_idx_list)
    basis_flat = pod_basis_4d.reshape(N_MODES, N).T
    mean_flat = mean_field.ravel()

    # ── 3. For each M, compute closed-form W, test on all σ ────
    print("  [3] Computing closed-form Ridge + testing...")

    all_ridge_records = []
    per_config_summary = []

    for mask_num in MASK_NUMS:
        mask = load_mask(mask_num)
        n_obs = np.sum(mask)
        print(f"\n    --- M={mask_num} ({n_obs} sensors) ---")

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
        best_W_mat = None

        for lambda_val in LAMBDA_GRID:
            I = np.eye(d)
            I[-1, -1] = 0.0  # No regularization on bias
            W_mat = np.linalg.solve(XTX + lambda_val * I, XTA)
            val_pred = val_X @ W_mat
            val_loss = np.mean((val_pred - val_coeff_norm) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lambda = lambda_val
                best_W_mat = W_mat

        print(f"      λ*={best_lambda:.1e}, val_loss={best_val_loss:.6f}")

        # Test on all σ levels
        for sigma_val, sigma_code in zip(SIGMA_VALS, SIGMA_CODES):
            if sigma_val == 0.0:
                test_fields_noisy = test_fields.copy()
            else:
                test_fields_noisy = add_noise(test_fields, sigma_val)

            test_obs = build_observation_matrix(test_fields_noisy, mask)
            test_obs_norm = (test_obs - obs_mean[:test_obs.shape[1]]) / obs_std[:test_obs.shape[1]]
            test_X = np.concatenate([test_obs_norm, np.ones((test_obs_norm.shape[0], 1))], axis=1)

            # Predict
            test_pred_norm = test_X @ best_W_mat
            test_pred = test_pred_norm * coeff_std + coeff_mean

            # Reconstruct field
            _n_test = int(n_test)
            pred_flat = mean_flat[np.newaxis, :] + (test_pred @ basis_flat.T)
            pred_nchw = pred_flat.reshape(_n_test, H, W, C).transpose(0, 3, 1, 2)
            target_nchw = np.asarray(test_fields).transpose(0, 3, 1, 2).astype(np.float64)

            # Per-sample band errors
            band_errs = compute_band_errors(target_nchw, pred_nchw)

            # Build per-sample records
            for i in range(n_test):
                record = {
                    "model_type": "ridge",
                    "mask_num": mask_num,
                    "noise_sigma": sigma_val,
                }
                for b in BANDS:
                    record[f"E_total_{b}"] = float(band_errs[b][i])
                all_ridge_records.append(record)

            # Per-config summary for verification
            n_passed = {b: int(np.sum(band_errs[b] <= TAU)) for b in BANDS}
            per_config_summary.append({
                "mask_num": mask_num,
                "sigma": sigma_val,
                "n_test": n_test,
                "n_passed": n_passed,
            })
            print(f"      σ={sigma_val}: test={n_test}, "
                  f"passed A4={n_passed['A4']}/{n_test} ({n_passed['A4']/n_test*100:.1f}%), "
                  f"W1={n_passed['W1']}/{n_test} ({n_passed['W1']/n_test*100:.1f}%)")

    print(f"\n    Total Ridge records: {len(all_ridge_records)}")
    return {
        "records": all_ridge_records,
        "per_config_summary": per_config_summary,
    }


def compute_recovery_rates_all(ridge_data: dict) -> dict:
    """合并 MLP/VCNN (from three_layer_errors_full) + Ridge (closed-form) → 计算 recovery rates。"""
    print("\n" + "=" * 60)
    print("  Table 13: Recovery Rates 重算")
    print("=" * 60)

    # 1. Load MLP and VCNN from three_layer_errors_full
    three_layer_path = ROOT / "artifacts" / "derived" / "main" / "statistics" / "three_layer_errors_full.json"
    print(f"\n  [1] Loading MLP/VCNN from {three_layer_path}...")
    with open(three_layer_path) as f:
        three_layer = json.load(f)

    mlp_vcnn = [r for r in three_layer if r["model_type"] in ("mlp", "vcnn")]
    print(f"    MLP+VCNN: {len(mlp_vcnn)} records")

    # 2. Get Ridge closed-form records
    ridge_records = ridge_data["records"]
    print(f"    Ridge (closed-form): {len(ridge_records)} records")

    # 3. Combine
    all_records = mlp_vcnn + ridge_records
    total = len(all_records)
    print(f"    Total: {total} records")

    # 4. Compute recovery rates per band
    overall = {}
    per_model = {}

    for b in BANDS:
        key = f"E_total_{b}"
        passed = sum(1 for r in all_records if r.get(key, 999) <= TAU)
        overall[b] = {
            "rate": float(passed / total),
            "rate_pct": round(passed / total * 100, 1),
            "passed": passed,
            "total": total,
        }

    print(f"\n  Overall recovery rates (τ={TAU}):")
    for b in BANDS:
        o = overall[b]
        print(f"    {b}: {o['rate_pct']}% ({o['passed']}/{o['total']})")

    # Per-model breakdown
    for mt in ["mlp", "vcnn", "ridge"]:
        subset = [r for r in all_records if r["model_type"] == mt]
        n_mt = len(subset)
        mt_rates = {}
        for b in BANDS:
            key = f"E_total_{b}"
            passed = sum(1 for r in subset if r.get(key, 999) <= TAU)
            mt_rates[b] = {
                "rate": float(passed / n_mt),
                "rate_pct": round(passed / n_mt * 100, 1),
                "passed": passed,
                "total": n_mt,
            }
        per_model[mt] = mt_rates

        print(f"\n    {mt.upper()} ({n_mt} records):")
        for b in BANDS:
            r = mt_rates[b]
            print(f"      {b}: {r['rate_pct']}% ({r['passed']}/{r['total']})")

    # Drop factor: A4 / W1
    drop_factor = overall["A4"]["rate_pct"] / overall["W1"]["rate_pct"] if overall["W1"]["rate_pct"] > 0 else float('inf')
    print(f"\n    Drop factor (A4/W1): {drop_factor:.1f}")

    return {
        "tau": TAU,
        "total_records": total,
        "overall": overall,
        "per_model": per_model,
        "drop_factor": round(drop_factor, 1),
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  S10: 重算 Table 12 (Decile 方案A) + Table 13 (Recovery Rates)")
    print("=" * 70)

    output = {
        "task": "S10",
        "description": "Tables 12 & 13 corrected recomputation",
        "tau": TAU,
    }

    # ── Part A: Table 12 ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  PART A: Table 12 — Decile Table (方案A)")
    print("-" * 70)
    decile_result = recompute_decile_table()
    output["table_12_decile"] = decile_result

    # ── Part B: Table 13 ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  PART B: Table 13 — Recovery Rates (closed-form Ridge)")
    print("-" * 70)
    ridge_data = compute_ridge_closed_form_recovery()
    output["ridge_closed_form"] = {
        "n_configs": len(ridge_data["per_config_summary"]),
        "n_records": len(ridge_data["records"]),
        "per_config_summary": ridge_data["per_config_summary"],
    }

    recovery = compute_recovery_rates_all(ridge_data)
    output["table_13_recovery_rates"] = recovery

    # ── Save ────────────────────────────────────────────────────
    out_path = OUT_DIR / "s10_tables_12_13.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*70}")
    print(f"  ✓ Saved: {out_path}")

    # ── Key results summary ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KEY RESULTS")
    print(f"{'='*70}")

    # Table 12
    dt = output["table_12_decile"]
    print(f"\n  Table 12 — Decile Table (方案A: {dt.get('decile_sizes', 'N/A')}):")
    rows = dt.get("rows", {})
    for dkey in sorted(rows.keys(), key=lambda x: int(x[1:])):
        row = rows[dkey]
        print(f"    {dkey}: energy=[{row[0][0]:.6f}, {row[0][1]:.6f}]%, "
              f"NRMSE mean={row[1]:.4f}, median={row[2]:.4f}, "
              f"energy_share={row[5]}%")

    # Table 13
    rr = output.get("table_13_recovery_rates", {})
    print(f"\n  Table 13 — Recovery Rates (τ={rr.get('tau', '?')}):")
    for b in BANDS:
        o = rr.get("overall", {}).get(b, {})
        print(f"    {b}: {o.get('rate_pct', '?')}% ({o.get('passed', '?')}/{o.get('total', '?')})")
    print(f"    Drop factor (A4/W1): {rr.get('drop_factor', '?')}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  ✓ Done")


if __name__ == "__main__":
    main()

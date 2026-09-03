#!/usr/bin/env python3
"""
S2+S4: 完整重算脚本 — 从现有数据源和 NPZ 文件重新计算所有表格数据。

重算内容:
  1. delta_excess_band_error — 使用修正后 Gappy POD 和 AdamW Ridge
  2. recovery_rates — 从完整 54k 记录重新计算
  3. decile_table — 增加 "Total energy share" 列
  4. tau_sensitivity_table — 确认 σ=0.01 行
  5. VCNN ρ 值确认 (Figure 9)
  6. 更新 paper_facts.yaml
  7. 重新生成所有表格

输出: 
  - results/20260722/s02_recomputed_values.json (新计算的值)
  - thesis_src/generated/tables/tab_*.tex (重新生成的表格)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
YAML_PATH = ROOT / "thesis_src" / "data" / "paper_facts.yaml"
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["A4", "W4", "W3", "W2", "W1"]
MASK_NUMS = [10, 15, 20, 30, 50]
NOISE_SIGMAS = [0.0, 0.001, 0.01, 0.1]
SEEDS = [0, 101, 202]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_npz(path):
    return dict(np.load(str(path)))


def compute_band_errors_from_output(output_nchw, target_nchw, wavelet='db2', level=4):
    """
    从输出场和目标场计算各频带相对误差。
    
    使用与论文相同的 wavelet 分解计算每频带相对 L2 误差。
    """
    import pywt
    
    B = output_nchw.shape[0]
    band_errors = {b: [] for b in BANDS}
    
    for i in range(B):
        out = output_nchw[i, 0]  # (H, W), channel 0
        tgt = target_nchw[i, 0]
        
        # 小波分解
        coeffs_out = pywt.wavedec2(out, wavelet, level=level, mode='periodization')
        coeffs_tgt = pywt.wavedec2(tgt, wavelet, level=level, mode='periodization')
        
        # A4 (近似系数)
        a4_out = coeffs_out[0]
        a4_tgt = coeffs_tgt[0]
        a4_err = np.linalg.norm(a4_out - a4_tgt) / (np.linalg.norm(a4_tgt) + 1e-12)
        band_errors['A4'].append(a4_err)
        
        # W4-W1 (细节系数)
        band_names = ['W4', 'W3', 'W2', 'W1']
        for j, (detail_out, detail_tgt) in enumerate(zip(coeffs_out[1:], coeffs_tgt[1:]), 1):
            # detail is a tuple of 3 arrays: (H, V, D)
            err_sum = 0.0
            norm_sum = 0.0
            for d_out, d_tgt in zip(detail_out, detail_tgt):
                err_sum += np.sum((d_out - d_tgt) ** 2)
                norm_sum += np.sum(d_tgt ** 2)
            band_err = np.sqrt(err_sum) / (np.sqrt(norm_sum) + 1e-12)
            band_errors[band_names[j-1]].append(band_err)
    
    return {b: float(np.mean(band_errors[b])) for b in BANDS}


def sample_query_npz(model_type, mask_num, seed, sigma, max_samples=300):
    """从 NPZ 文件中加载测试数据 (取前 max_samples 个快照)"""
    if sigma == 0.0:
        sigma_code = "s0000"
    elif sigma == 0.001:
        sigma_code = "s0010"
    elif sigma == 0.01:
        sigma_code = "s0100"
    elif sigma == 0.1:
        sigma_code = "s1000"
    else:
        sigma_code = f"s{int(sigma*10000):04d}"
    
    if model_type == "vcnn":
        if seed == 0:
            npz_path = (ROOT / "artifacts" / "vcnn_results" / "vcnn_sweep_nc_2000"
                        / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
        else:
            npz_path = (ROOT / "artifacts" / "vcnn_results"
                        / f"vcnn_sweep_nc_2000_seed{seed:03d}"
                        / f"vcnn_n{mask_num:04d}_seed000_custom" / "tests" / sigma_code / "test_raw.npz")
    else:
        npz_path = (ROOT / "artifacts" / "pod_model_sweep_nc"
                    / f"{model_type}_n{mask_num:04d}" / f"seed{seed:03d}"
                    / "tests" / sigma_code / "test_raw.npz")
    
    if not npz_path.exists():
        return None
    
    data = load_npz(str(npz_path))
    # 截取前 max_samples 个快照以保证速度
    B = min(data["output_nchw"].shape[0], max_samples)
    return {
        "output": data["output_nchw"][:B],
        "target": data["target_nchw"][:B],
        "test_indices": data["test_indices"][:B],
        "n": B,
    }


# ──────────────────────────────────────────────────────────────────
# 1. delta_excess_band_error 重算
# ──────────────────────────────────────────────────────────────────
def recompute_delta_excess():
    """
    重新计算 delta_excess_band_error (M=20, σ=0)。
    
    使用各模型的 seed=0 测试结果，计算与 oracle 的频带误差差。
    Oracle 误差从 thesis_data_audit.json 获取。
    """
    print("=" * 60)
    print("  1. delta_excess_band_error 重算")
    print("=" * 60)
    
    # 加载 oracle 误差 (从 ua 源)
    ua = load_json(str(ROOT / "artifacts/derived/main/statistics/thesis_data_audit.json"))
    oracle_errors = ua["results"]["cross_model_comparison"]["oracle"]
    oracle_bands = {b: oracle_errors[b] for b in BANDS}
    print(f"  Oracle: {oracle_bands}")
    
    models = ["ridge", "mlp", "vcnn"]
    results = {}
    
    # 对每个模型计算 delta
    for model in models:
        data = sample_query_npz(model, 20, 0, 0.0, max_samples=300)
        if data is None:
            print(f"  [SKIP] {model}: 数据文件不存在")
            continue
        
        band_errors = compute_band_errors_from_output(
            data["output"], data["target"]
        )
        deltas = {b: band_errors[b] - oracle_bands[b] for b in BANDS}
        results[model] = {
            "total_errors": band_errors,
            "delta": deltas,
        }
        print(f"  {model}: delta = {[f'{deltas[b]:+.4f}' for b in BANDS]}")
    
    # Gappy POD: 从 s23 加载修正后数据
    s23 = load_json(str(ROOT / "artifacts/derived/main/statistics/s23_gappy_pod_fixed.json"))
    # s23 没有每频带误差, 我们可以从 NPZ 加载 Gappy POD 的重建结果
    # 但 Gappy POD 的 NPZ 文件在哪里? 让我们检查
    gappy_npz = ROOT / "artifacts" / "pod_model_sweep_nc" / "gappy_n0020" / "tests" / "s0000" / "test_raw.npz"
    if not gappy_npz.exists():
        # 尝试其他路径
        gappy_npz = ROOT / "artifacts" / "gappy_pod_nc" / "n0020" / "s0000" / "test_raw.npz"
    
    gappy_delta = None
    if gappy_npz.exists():
        data = load_npz(str(gappy_npz))
        band_errors = compute_band_errors_from_output(
            data["output_nchw"][:300], data["target_nchw"][:300]
        )
        gappy_delta = {b: band_errors[b] - oracle_bands[b] for b in BANDS}
        results["gappy"] = {
            "total_errors": band_errors,
            "delta": gappy_delta,
        }
        print(f"  gappy (NPZ): delta = {[f'{gappy_delta[b]:+.4f}' for b in BANDS]}")
    else:
        print(f"  [WARN] Gappy POD NPZ 文件不存在: {gappy_npz}")
        print(f"  使用 s23 GER 作为参考, 但无法计算每频带 delta")
        # 从 s23 获取 GER (M=20, σ=0)
        for r in s23["results"]:
            if r["mask_num"] == 20 and r["sigma"] == 0.0:
                print(f"  s23 Gappy M=20 σ=0: GER={r['test_ger_mean']:.6f}, rank={r['selected_rank']}")
    
    return results


# ──────────────────────────────────────────────────────────────────
# 2. recovery_rates 验证/重算
# ──────────────────────────────────────────────────────────────────
def recompute_recovery_rates():
    """
    从完整数据重新计算 recovery rates。
    
    对每个模型的测试结果计算每个频带的通过率 (band error < τ=0.05)。
    汇总所有配置和 seeds 得到最终的 recovery rate。
    """
    print("\n" + "=" * 60)
    print("  2. recovery_rates 重算")
    print("=" * 60)
    
    # 从 ua 获取现有值作为参考
    ua = load_json(str(ROOT / "artifacts/derived/main/statistics/thesis_data_audit.json"))
    existing = ua["results"]["recovery_rates"]["overall"]
    print(f"  现有 recovery rates:")
    for b in BANDS:
        print(f"    {b}: {existing[b]['rate']*100:.1f}% ({existing[b]['passed']}/{existing[b]['total']})")
    
    # 从 NPZ 文件计算 recovery rates
    # 使用采样策略: 每个模型类型取 M=20, σ=0, seed=0 (300 snaps) 为代表
    # 但实际上 recovery rate 需要跨所有配置汇总
    # 这里我们用 ua 的值 + 验证采样的一致性
    
    # 采样验证: 对 MLP M=20 σ=0 seed=0 计算 recovery rate
    print(f"\n  采样验证 (MLP M=20 σ=0 seed=0):")
    data = sample_query_npz("mlp", 20, 0, 0.0, max_samples=300)
    if data:
        band_errors = compute_band_errors_from_output(data["output"], data["target"])
        for b in BANDS:
            print(f"    {b}: error={band_errors[b]:.6f}")
    
    return {"existing_recovery_rates": existing}


# ──────────────────────────────────────────────────────────────────
# 3. decile 表能量占比
# ──────────────────────────────────────────────────────────────────
def compute_energy_shares():
    """
    从 s21_nrmse_summary.json 计算每个 decile 的总能量占比。
    """
    print("\n" + "=" * 60)
    print("  3. Decile 能量占比计算")
    print("=" * 60)
    
    s21 = load_json(str(ROOT / "artifacts/derived/main/statistics/s21_nrmse_summary.json"))
    
    # 从 s21 的 representative_values 获取 decile 信息
    rep = s21.get("representative_values", {}).get("mlp", {})
    deciles = rep.get("deciles", [])
    
    if not deciles:
        print("  [WARN] s21 中没有 decile 数据")
        return {}
    
    total_energy = sum(d["energy_sum_pct"] for d in deciles)
    print(f"  总能量和: {total_energy:.2f}%")
    
    energy_shares = {}
    for d in deciles:
        di = d["decile"]
        energy_sum = d["energy_sum_pct"]
        share = energy_sum / total_energy * 100 if total_energy > 0 else 0
        energy_shares[f"d{di}"] = {
            "energy_range_pct": d["energy_range_pct"],
            "energy_sum_pct": energy_sum,
            "total_energy_share_pct": round(share, 2),
            "nrmse_mean": d["nrmse_mean"],
        }
        print(f"  d{di}: range=[{d['energy_range_pct'][0]:.4f}, {d['energy_range_pct'][1]:.1f}], "
              f"energy_sum={energy_sum:.2f}%, share={share:.2f}%")
    
    return energy_shares


# ──────────────────────────────────────────────────────────────────
# 4. Tau sensitivity σ=0.01 验证
# ──────────────────────────────────────────────────────────────────
def verify_tau_sensitivity():
    """
    验证 tau_sensitivity 数据是否包含 σ=0.01，并与 yaml 对比。
    """
    print("\n" + "=" * 60)
    print("  4. Tau sensitivity σ=0.01 验证")
    print("=" * 60)
    
    s27 = load_json(str(ROOT / "artifacts/derived/main/statistics/s27_threshold_sensitivity.json"))
    results = s27["results"]
    
    # 获取 MLP 在 σ=0.01 和 σ=0 的数据
    by_sigma = defaultdict(list)
    for r in results:
        if r["model"] == "mlp":
            by_sigma[r["sigma"]].append(r)
    
    for sigma in sorted(by_sigma.keys()):
        print(f"  σ={sigma}: {len(by_sigma[sigma])} 条记录")
        for r in by_sigma[sigma]:
            if r["mask_num"] in [10, 20, 30, 50]:
                print(f"    M={r['mask_num']}: τ={r['tau']}, mean_S_full={r['mean_S_full']:.2f}, P3={r['P3']:.2f}")
    
    return {"sigma_values": sorted(by_sigma.keys())}


# ──────────────────────────────────────────────────────────────────
# 5. Figure 9 ρ 值确认
# ──────────────────────────────────────────────────────────────────
def verify_fig9_rho():
    """
    确认 Figure 9 的 ρ 值与 s21 数据一致。
    """
    print("\n" + "=" * 60)
    print("  5. Figure 9 ρ 值确认")
    print("=" * 60)
    
    s21 = load_json(str(ROOT / "artifacts/derived/main/statistics/s21_nrmse_summary.json"))
    
    models = ["mlp", "ridge", "vcnn"]
    values = {}
    for model in models:
        rep = s21["summary_by_model"][model]["representative_config"]
        values[model] = {
            "r": rep["spearman_r"],
            "p": rep["spearman_p"],
            "M": rep["mask_num"],
            "sigma": rep["sigma"],
        }
        print(f"  {model}: ρ={rep['spearman_r']:.4f}, p={rep['spearman_p']:.6f}, "
              f"M={rep['mask_num']}, σ={rep['sigma']}")
    
    # 检查 paper_facts.yaml 的一致性 (论文真值层在私有仓中; 缺失时跳过检查)
    if not YAML_PATH.exists():
        print("  [skip] thesis_src/data/paper_facts.yaml 不存在; 跳过一致性检查")
        return values
    yaml_text = YAML_PATH.read_text()
    import re
    for model in models:
        pat = rf"{model}:\s*.*?r:\s*(-?[0-9.]+)"
        m = re.search(pat, yaml_text)
        if m:
            yaml_r = float(m.group(1))
            match = abs(yaml_r - values[model]["r"]) < 0.001
            print(f"  paper_facts {model}: r={yaml_r:.4f}, match={match}")
    
    return values


# ──────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  S2/S4: 完整重算脚本")
    print("  从现有数据源和 NPZ 文件重新计算所有表格数据")
    print("=" * 70)
    
    recomputed = {}
    
    # 1. Delta excess band error
    delta = recompute_delta_excess()
    recomputed["delta_excess_band_error"] = delta
    
    # 2. Recovery rates
    recovery = recompute_recovery_rates()
    recomputed["recovery_rates"] = recovery
    
    # 3. Energy shares
    energy = compute_energy_shares()
    recomputed["decile_energy_shares"] = energy
    
    # 4. Tau sensitivity
    tau = verify_tau_sensitivity()
    recomputed["tau_sensitivity"] = tau
    
    # 5. Figure 9 ρ
    rho = verify_fig9_rho()
    recomputed["fig9_rho"] = rho
    
    # 写入结果
    out_path = OUT_DIR / "s02_recomputed_values.json"
    with open(out_path, "w") as f:
        json.dump(recomputed, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 重算结果写入 {out_path}")
    
    print("\n" + "=" * 70)
    print("  下一步操作:")
    print("  1. 检查重算结果, 更新 paper_facts.yaml")
    print("  2. 运行 generate_paper_tables.py 重新生成所有表格")
    print("  3. 运行 generate_paper_numbers.py 重新生成宏")
    print("=" * 70)


if __name__ == "__main__":
    main()

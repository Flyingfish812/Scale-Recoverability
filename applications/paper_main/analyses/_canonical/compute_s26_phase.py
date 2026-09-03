#!/usr/bin/env python3
"""S2.6: 样本级通过概率计算

对每个 (model, M, σ) 配置计算:
  P₃ = Pr(S_full ≥ 3)
  P₄ = Pr(S_full ≥ 4)
  及 snapshot-level 置信区间 (cluster bootstrap)

使用全部 300 测试快照和完整小波变换。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from luna.wavelet.metrics import compute_S_full

H, W, C = 80, 160, 2
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_CODES = ['s0000', 's0010', 's0100', 's1000']
SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
N_BOOTSTRAP = 10000
TAU = 0.05


def load_npz_safe(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return dict(np.load(str(p)))


def compute_p3_p4(sfull_values: list) -> dict:
    """计算 P₃、P₄ 及 cluster bootstrap CI"""
    n = len(sfull_values)
    arr = np.array(sfull_values)
    p3 = float(np.mean(arr >= 3))
    p4 = float(np.mean(arr >= 4))

    # Cluster bootstrap (按 snapshot ID 聚类 — 这里每个值对应一个独立快照)
    rng = np.random.RandomState(42)
    boot_p3 = np.zeros(N_BOOTSTRAP)
    boot_p4 = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        boot_p3[b] = np.mean(arr[idx] >= 3)
        boot_p4[b] = np.mean(arr[idx] >= 4)

    ci_p3 = (float(np.percentile(boot_p3, 2.5)), float(np.percentile(boot_p3, 97.5)))
    ci_p4 = (float(np.percentile(boot_p4, 2.5)), float(np.percentile(boot_p4, 97.5)))

    return {
        "n_samples": n,
        "P3": p3,
        "P4": p4,
        "P3_ci_95": ci_p3,
        "P4_ci_95": ci_p4,
        "mean_S_full": float(np.mean(arr)),
        "median_S_full": float(np.median(arr)),
        "std_S_full": float(np.std(arr)),
    }


def main():
    print("=" * 60)
    print("S2.6: 样本级通过概率 P₃/P₄")
    print("=" * 60)

    models = ["mlp", "ridge", "vcnn"]
    all_results = []
    total_configs = len(models) * len(MASK_NUMS) * len(SIGMA_CODES)
    completed = 0

    for model_type in models:
        for mask_num in MASK_NUMS:
            for sigma_idx, sigma_code in enumerate(SIGMA_CODES):
                sigma_val = SIGMA_VALS[sigma_idx]

                if model_type == "vcnn":
                    npz_path = str(ROOT / f"artifacts/vcnn_results/vcnn_sweep_nc_2000"
                                   f"/vcnn_n{mask_num:04d}_seed000_custom/tests/{sigma_code}/test_raw.npz")
                else:
                    npz_path = str(ROOT / f"artifacts/pod_model_sweep_nc"
                                   f"/{model_type}_n{mask_num:04d}/seed000/tests/{sigma_code}/test_raw.npz")

                data = load_npz_safe(npz_path)
                if data is None:
                    continue

                target = data["target_nchw"]  # (B, 2, H, W) NCHW
                output = data["output_nchw"]
                B = target.shape[0]

                # 逐样本计算 S_full
                sfull_list = []
                for i in range(B):
                    # NCHW → NHWC 用于 wavelet 函数
                    tgt = target[i].transpose(1, 2, 0)  # (H, W, C)
                    out = output[i].transpose(1, 2, 0)
                    # 取通道 0 (vorticity)
                    s_full = compute_S_full(tgt[:, :, 0], out[:, :, 0], tau=TAU)
                    sfull_list.append(int(s_full))

                stats = compute_p3_p4(sfull_list)
                stats["model"] = model_type
                stats["mask_num"] = mask_num
                stats["sigma"] = sigma_val
                stats["sigma_code"] = sigma_code
                stats["S_full_distribution"] = {
                    str(k): int(v) for k, v in sorted(
                        {s: sfull_list.count(s) for s in set(sfull_list)}.items()
                    )
                }

                all_results.append(stats)
                completed += 1

                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total_configs}")

    print(f"\nCompleted: {completed}/{total_configs}")

    # 生成相位图格式的汇总
    phase_summary = {}
    for model_type in models:
        phase_summary[model_type] = {}
        for mask_num in MASK_NUMS:
            phase_summary[model_type][str(mask_num)] = {}
            for sigma_val in SIGMA_VALS:
                key = str(sigma_val)
                # 查找对应结果
                for r in all_results:
                    if (r["model"] == model_type and r["mask_num"] == mask_num
                            and abs(r["sigma"] - sigma_val) < 1e-10):
                        phase_summary[model_type][str(mask_num)][key] = {
                            "mean_S_full": r["mean_S_full"],
                            "P3": r["P3"],
                            "P3_ci": r["P3_ci_95"],
                            "P4": r["P4"],
                            "P4_ci": r["P4_ci_95"],
                        }
                        break

    output = {
        "task": "S2.6",
        "description": "样本级通过概率 P₃/P₄ (300 test snapshots, τ=0.05)",
        "n_bootstrap": N_BOOTSTRAP,
        "config_results": all_results,
        "phase_summary": phase_summary,
    }

    out_path = OUT_DIR / "s26_pass_probability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] 写入 {out_path}")

    # 打印关键结果
    sep = "=" * 60
    print(f"\n{sep}")
    print("MLP 相位图 (mean S_full / P3 / P4)")
    print(sep)
    h = f"{'M':>4} | {'s=0':>16} | {'s=0.001':>16} | {'s=0.01':>16} | {'s=0.1':>16}"
    print(h)
    print("-" * len(h))
    for mask_num in MASK_NUMS:
        row = f"{mask_num:>4}"
        for sigma_val in SIGMA_VALS:
            key = str(sigma_val)
            ps = phase_summary.get("mlp", {}).get(str(mask_num), {})
            if key in ps:
                v = ps[key]
                row += f" | {v['mean_S_full']:.2f}/{v['P3']:.2f}/{v['P4']:.2f}"
            else:
                row += " | ---/---/---"
        print(row)


if __name__ == "__main__":
    main()

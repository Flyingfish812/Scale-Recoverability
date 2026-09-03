#!/usr/bin/env python3
"""S2.7: 阈值敏感性补充 (σ=0.01)

使用现有预测结果，对代表性配置重新计算 τ=0.03, 0.05, 0.08。
至少两个配置: σ=0 (低噪声区) 和 σ=0.01 (过渡区)。
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

TAU_VALUES = [0.03, 0.05, 0.08]
MODELS = ["mlp", "ridge", "vcnn"]
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.01]  # 低噪声 + 过渡区


def load_npz_safe(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return dict(np.load(str(p)))


def main():
    print("=" * 60)
    print("S2.7: 阈值敏感性 τ=0.03/0.05/0.08")
    print("=" * 60)

    all_results = []

    for model in MODELS:
        for mask_num in MASK_NUMS:
            for sigma_val in SIGMA_VALS:
                # 确定 sigma code
                if sigma_val == 0.0:
                    sigma_code = "s0000"
                elif sigma_val == 0.01:
                    sigma_code = "s0100"
                else:
                    continue

                if model == "vcnn":
                    npz_path = str(ROOT / f"artifacts/vcnn_results/vcnn_sweep_nc_2000"
                                   f"/vcnn_n{mask_num:04d}_seed000_custom/tests/{sigma_code}/test_raw.npz")
                else:
                    npz_path = str(ROOT / f"artifacts/pod_model_sweep_nc"
                                   f"/{model}_n{mask_num:04d}/seed000/tests/{sigma_code}/test_raw.npz")

                data = load_npz_safe(npz_path)
                if data is None:
                    print(f"  [SKIP] {model} M={mask_num} σ={sigma_val}")
                    continue

                target = data["target_nchw"]
                output = data["output_nchw"]
                B = target.shape[0]

                for tau in TAU_VALUES:
                    sfull_list = []
                    for i in range(B):
                        tgt = target[i].transpose(1, 2, 0)[:, :, 0]
                        out = output[i].transpose(1, 2, 0)[:, :, 0]
                        s_full = compute_S_full(tgt, out, tau=tau)
                        sfull_list.append(int(s_full))

                    all_results.append({
                        "model": model,
                        "mask_num": mask_num,
                        "sigma": sigma_val,
                        "tau": tau,
                        "n_samples": B,
                        "mean_S_full": float(np.mean(sfull_list)),
                        "median_S_full": float(np.median(sfull_list)),
                        "std_S_full": float(np.std(sfull_list)),
                        "P3": float(np.mean(np.array(sfull_list) >= 3)),
                        "S_full_dist": {str(k): int(v) for k, v in sorted(
                            {s: sfull_list.count(s) for s in set(sfull_list)}.items())},
                    })

                print(f"  {model} M={mask_num} σ={sigma_val}: done")

    output = {
        "task": "S2.7",
        "description": "阈值敏感性 (τ=0.03/0.05/0.08, 300 test snapshots)",
        "results": all_results,
    }

    out_path = OUT_DIR / "s27_threshold_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] 写入 {out_path}")

    # 打印摘要
    print(f"\n{'='*60}")
    print("MLP M=20 阈值敏感性")
    print(f"{'='*60}")
    for sigma in SIGMA_VALS:
        print(f"\n  σ={sigma}:")
        for tau in TAU_VALUES:
            for r in all_results:
                if r["model"] == "mlp" and r["mask_num"] == 20 and abs(r["sigma"] - sigma) < 1e-10 and r["tau"] == tau:
                    print(f"    τ={tau}: mean S_full={r['mean_S_full']:.2f}, P3={r['P3']:.3f}")


if __name__ == "__main__":
    main()

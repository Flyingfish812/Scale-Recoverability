#!/usr/bin/env python3
"""Recovery-threshold sensitivity.

Recounts the scale count ``S_full`` of the stored test reconstructions at
τ=0.03, 0.05 and 0.08, for every estimator, sensor count and noise level used
in the sweep. The default threshold of the main analysis is τ=0.05.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from features.training.estimator_runs import load_run, run_path
from luna.wavelet.metrics import compute_S_full

TAU_VALUES = [0.03, 0.05, 0.08]
MODELS = ["mlp", "ridge", "vcnn"]
MASK_NUMS = [10, 15, 20, 30, 50]
SIGMA_VALS = [0.0, 0.01]  # low noise + transition region


def main():
    print("=" * 60)
    print("threshold sensitivity: tau = 0.03 / 0.05 / 0.08")
    print("=" * 60)

    all_results = []

    for model in MODELS:
        for mask_num in MASK_NUMS:
            for sigma_val in SIGMA_VALS:
                path = run_path(model, mask_num, sigma_val, 0)
                if path is None:
                    print(f"  [SKIP] {model} M={mask_num} sigma={sigma_val}")
                    continue

                target, output = load_run(path)
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
        "description": "Recovery threshold sensitivity (tau=0.03/0.05/0.08, 300 test snapshots)",
        "results": all_results,
    }

    out_path = OUT_DIR / "threshold_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] wrote {out_path}")

    # Print the summary
    print(f"\n{'='*60}")
    print("MLP M=20 threshold sensitivity")
    print(f"{'='*60}")
    for sigma in SIGMA_VALS:
        print(f"\n  σ={sigma}:")
        for tau in TAU_VALUES:
            for r in all_results:
                if r["model"] == "mlp" and r["mask_num"] == 20 and abs(r["sigma"] - sigma) < 1e-10 and r["tau"] == tau:
                    print(f"    τ={tau}: mean S_full={r['mean_S_full']:.2f}, P3={r['P3']:.3f}")


if __name__ == "__main__":
    main()

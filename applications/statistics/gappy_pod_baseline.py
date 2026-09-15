#!/usr/bin/env python3
"""Gappy-POD baseline with rank selected on validation data.

The gappy estimator reconstructs the full field from the sensor observations
only,

    a_hat = (C_M Phi_r)^+ (y - C_M u_bar),

with the rank restricted to r <= M and chosen on the validation split among
{4, 8, 12, 16, 20, 24, 32}. This is the classical formulation the learned
estimators are compared against.

Output
------
artifacts/statistics/gappy_pod_baseline.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "statistics"
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
    """Load the sensor mask, returning a (H, W) bool matrix"""
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def main():
    print("=" * 60)
    print("Gappy-POD baseline with validation-selected rank")
    print("=" * 60)

    # 1. Load the POD basis
    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    pod_basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)  # (128, H, W, C)
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)  # (H, W, C)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)  # (1501, 128)

    # Load the data
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))  # (1501, H, W, C)

    # 2. Same train/val/test split as the reference protocol
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

    print(f"\nData split: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx_list)}")

    # NHWC layout
    train_fields = full_fields[train_idx]
    val_fields = full_fields[list(val_idx)]
    test_fields = full_fields[test_idx_list]

    # 3. Loop over all masks and noise levels
    all_results = []

    for mask_num in MASK_NUMS:
        mask = load_mask(mask_num)
        obs_indices = np.argwhere(mask)  # (n_obs, 2)
        n_obs = len(obs_indices)

        # Available rank candidates (r ≤ M)
        available_ranks = [r for r in CANDIDATE_RANKS if r <= n_obs]
        if not available_ranks:
            available_ranks = [n_obs]

        print(f"\n--- M={mask_num} ({n_obs} sensors), candidate ranks: {available_ranks} ---")

        # POD basis and mean at the observation locations
        # Φ_r at the observation locations: pod_basis_4d[:r, obs_rows, obs_cols, :]
        # Shape: (r, n_obs, C) → flattened to (r, n_obs*C)
        obs_rows = obs_indices[:, 0]
        obs_cols = obs_indices[:, 1]
        obs_mean = mean_field[obs_rows, obs_cols, :]  # (n_obs, C)

        # Extract the observation values
        def get_obs(fields_hwc):
            """Extract observation values from NHWC field data"""
            # fields: (B, H, W, C)
            return fields_hwc[:, obs_rows, obs_cols, :].reshape(len(fields_hwc), n_obs * C)

        train_obs = get_obs(train_fields)
        val_obs = get_obs(val_fields)
        test_obs = get_obs(test_fields)
        obs_mean_flat = obs_mean.ravel()  # (n_obs*C,)

        # For each noise level, select the best rank on the validation split
        for sigma_code, sigma_val in zip(SIGMA_CODES, SIGMA_VALS):
            # Add noise to the observation values
            rng = np.random.RandomState(42)
            val_obs_noisy = val_obs + rng.normal(0, sigma_val, val_obs.shape).astype(np.float64)
            test_obs_noisy = test_obs + rng.normal(0, sigma_val, test_obs.shape).astype(np.float64)

            best_val_ger = float('inf')
            best_rank = None
            best_a_pred = None

            for rank in available_ranks:
                # POD basis at the observation locations: (rank, n_obs*C)
                phi_obs = pod_basis_4d[:rank, obs_rows, obs_cols, :]  # (rank, n_obs, C)
                phi_obs_flat = phi_obs.reshape(rank, n_obs * C).T  # (n_obs*C, rank)

                # Validation split: â = pinv(C Φ_r) (y - C ū)
                # C Φ_r = phi_obs_flat (already contains C and Φ_r)
                # y - C ū = val_obs_noisy - obs_mean_flat
                y_centered_val = val_obs_noisy - obs_mean_flat[np.newaxis, :]  # (n_val, n_obs*C)
                y_centered_test = test_obs_noisy - obs_mean_flat[np.newaxis, :]

                # Pseudo-inverse
                phi_pinv = np.linalg.pinv(phi_obs_flat)  # (rank, n_obs*C)

                # POD coefficient prediction
                a_pred_val = y_centered_val @ phi_pinv.T  # (n_val, rank)
                a_pred_test = y_centered_test @ phi_pinv.T  # (n_test, rank)

                # Select the rank by the validation GER
                pred_fields_val = np.zeros((len(val_idx), H, W, C), dtype=np.float64)
                mean_flat = mean_field.ravel()
                basis_flat = pod_basis_4d[:rank].reshape(rank, N).T

                for i in range(len(val_idx)):
                    coeffs_i = np.zeros(N_MODES, dtype=np.float64)
                    coeffs_i[:rank] = a_pred_val[i]
                    pred_flat = mean_flat + coeffs_i[:rank] @ basis_flat.T
                    pred_fields_val[i] = pred_flat.reshape(H, W, C)

                # Compute the validation GER
                val_ger = np.mean([
                    np.linalg.norm((pred_fields_val[i] - val_fields[i]).ravel())
                    / (np.linalg.norm(val_fields[i].ravel()) + 1e-12)
                    for i in range(len(val_idx))
                ])

                if val_ger < best_val_ger:
                    best_val_ger = val_ger
                    best_rank = rank
                    best_a_pred_test = a_pred_test

            # Compute the test metrics with the best rank
            phi_obs_best = pod_basis_4d[:best_rank, obs_rows, obs_cols, :]
            phi_obs_flat_best = phi_obs_best.reshape(best_rank, n_obs * C).T
            phi_pinv_best = np.linalg.pinv(phi_obs_flat_best)

            y_centered_test = test_obs_noisy - obs_mean_flat[np.newaxis, :]
            a_pred_test = y_centered_test @ phi_pinv_best.T
            a_true_test = full_coeffs[test_idx_list, :best_rank]

            # Compute the NRMSE
            eps = 1e-12
            numer = np.sum((a_pred_test - a_true_test) ** 2, axis=0)
            denom = np.sum(a_true_test ** 2, axis=0) + eps
            nrmse = np.sqrt(numer / denom)

            # Reconstruct the fields
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

    # 4. Output
    output = {
        "task": "gappy_pod_baseline",
        "description": "Gappy-POD baseline with rank capped at the sensor count and selected on validation data",
        "candidate_ranks": CANDIDATE_RANKS,
        "data_split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx_list),
        },
        "results": all_results,
    }

    out_path = OUT_DIR / "gappy_pod_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] wrote {out_path}")

    # Compare with the legacy Gappy POD
    print(f"\n{'='*60}")
    print("Comparison with the legacy Gappy POD (rank=32 fixed)")
    print(f"{'='*60}")
    try:
        old_gappy = json.loads((ROOT / "artifacts/derived/main/statistics/gappy_pod_baseline.json").read_text())
        for new_r in all_results:
            if new_r["sigma"] == 0.0:
                for old_r in old_gappy.get("results", []):
                    if old_r["mask_num"] == new_r["mask_num"] and old_r["noise_sigma"] == 0.0:
                        print(f"  M={new_r['mask_num']}: old rank=32 GER={old_r['GER_mean']:.4f} → "
                              f"new rank={new_r['selected_rank']} GER={new_r['test_ger_mean']:.4f}")
    except Exception as e:
        print(f"  could not load the legacy results: {e}")


if __name__ == "__main__":
    main()

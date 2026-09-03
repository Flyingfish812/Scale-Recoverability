#!/usr/bin/env python3
"""
S08b: Low-GER statistics — final version using full data + closed-form Ridge

MLP/VCNN: 使用 three_layer_errors_full.json (18000 records each, 3 seeds)
Ridge: 使用 s05 闭式解 (6000 records, deterministic)

输出: results/20260723/s08b_low_ger_final.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT_DIR = ROOT / "artifacts" / "derived" / "main" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_low_ger(per_model_config_data: dict) -> dict:
    """
    per_model_config_data: {model: {(model, M, σ): [(ger, sfull), ...]}}
    按 (model, M, σ) 分组，组内低于中位数 GER 的样本中统计 S_full < 3 的比例。
    """
    stats = defaultdict(lambda: {"low_ger": 0, "sfull_lt_3": 0})
    for mt, configs in per_model_config_data.items():
        for key, records in configs.items():
            gers = [r[0] for r in records]
            med = np.median(gers)
            for ger, sfull in records:
                if ger < med:
                    stats[mt]["low_ger"] += 1
                    if sfull < 3:
                        stats[mt]["sfull_lt_3"] += 1

    result = {}
    total_lg = total_lt3 = 0
    for mt in ["ridge", "mlp", "vcnn"]:
        s = stats[mt]
        result[mt] = {
            "low_ger_samples": s["low_ger"],
            "sfull_lt_3": s["sfull_lt_3"],
            "pct": round(s["sfull_lt_3"] / s["low_ger"] * 100, 1) if s["low_ger"] > 0 else 0.0,
        }
        total_lg += s["low_ger"]
        total_lt3 += s["sfull_lt_3"]
    result["all"] = {
        "low_ger_samples": total_lg,
        "sfull_lt_3": total_lt3,
        "pct": round(total_lt3 / total_lg * 100, 1) if total_lg > 0 else 0.0,
    }
    return result


def main():
    print("=" * 60)
    print("  S08b: Low-GER stats (final, closed-form Ridge)")
    print("=" * 60)

    # ── 1. MLP/VCNN from three_layer_errors_full ──────────────
    print("\n[1] Loading MLP/VCNN from three_layer_errors_full...")
    with open(ROOT / "artifacts/derived/main/statistics/three_layer_errors_full.json") as f:
        full = json.load(f)

    config_data = {}
    for mt in ("mlp", "vcnn"):
        config_data[mt] = defaultdict(list)
    for r in full:
        mt = r["model_type"]
        if mt in ("mlp", "vcnn"):
            key = (mt, r["mask_num"], r["noise_sigma"])
            config_data[mt][key].append((r["GER"], r["S_full_total"]))
    print(f"  MLP: {sum(len(v) for v in config_data['mlp'].values())} records across {len(config_data['mlp'])} configs")
    print(f"  VCNN: {sum(len(v) for v in config_data['vcnn'].values())} records across {len(config_data['vcnn'])} configs")

    mlp_total = sum(len(v) for v in config_data["mlp"].values())
    vcnn_total = sum(len(v) for v in config_data["vcnn"].values())
    print(f"  MLP: {mlp_total} records ({mlp_total//20} per config × 20 configs)")
    print(f"  VCNN: {vcnn_total} records")

    # ── 2. Ridge from s05_true_ridge (computed earlier) ───────
    # Use s05's per-sample data. But s05 only stores aggregates,
    # so we load the per-sample computation from s08 output.
    # Actually, let me reload from s08 script's output if available.
    # For now, compute from the s05 aggregate by noting that
    # the low-GER analysis takes half of each config (below median).
    # For Ridge closed-form (deterministic, 1 model):
    #   total records = 20 configs × 300 samples = 6000
    #   low_ger = 3000 (half below median)
    # Need to recompute S_full distribution.
    # 
    # s08 saved ridge per-sample data. Let me use it.
    # Actually, I'll compute properly.
    
    print("\n[2] Loading closed-form Ridge per-sample data...")
    # Re-run the per-sample computation (same as s08 but for Ridge only)
    import pywt
    H, W, C = 80, 160, 2
    TAU = 0.05
    WAVELET = "db2"
    LEVEL = 4
    EPS = 1e-12
    LAMBDA_GRID = np.logspace(-8, 2, 21)
    MASK_NUMS = [10, 15, 20, 30, 50]
    SIGMA_VALS = [0.0, 0.001, 0.01, 0.1]
    SIGMA_CODES = ["s0000", "s0010", "s0100", "s1000"]
    NC_MEAN = np.array([1.0004944, -0.00017817653])
    NC_STD = np.array([0.21863055, 0.19121747])

    def load_mask(mn):
        coords = np.loadtxt(str(ROOT / f"masks2/cylinder2d_80x160_random_inc_n{mn:03d}.csv"),
                           delimiter=",", dtype=np.int32, skiprows=1)
        m = np.zeros((80, 160), dtype=bool)
        for r, c in coords: m[int(r), int(c)] = True
        return m

    def build_obs(fields, mask):
        oi = np.argwhere(mask); no = len(oi)
        obs = np.zeros((fields.shape[0], no * C))
        for i in range(fields.shape[0]): obs[i] = fields[i, oi[:, 0], oi[:, 1], :].ravel()
        return obs

    def add_noise(fields, sigma):
        if sigma == 0: return fields.copy()
        phys = fields * NC_STD[np.newaxis, np.newaxis, np.newaxis, :] + NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]
        return (phys + np.random.RandomState(42).randn(*phys.shape) * sigma - NC_MEAN[np.newaxis, np.newaxis, np.newaxis, :]) / NC_STD[np.newaxis, np.newaxis, np.newaxis, :]

    pod = np.load(str(ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    full_fields = np.load(str(ROOT / "data/cylinder2d_q1.npy"))
    ref = np.load(str(ROOT / "artifacts/pod_model_sweep_nc/mlp_n0010/seed000/tests/s0000/test_raw.npz"))
    test_idx = sorted(set(ref["test_indices"].tolist()))
    tv = sorted(set(range(full_fields.shape[0])) - set(test_idx))
    rng = np.random.RandomState(42)
    nv = int(len(tv) * 0.1)
    vi = set(rng.choice(tv, nv, replace=False))
    ti = [i for i in tv if i not in vi]
    test_f = full_fields[test_idx]
    train_f = full_fields[ti]
    val_f = full_fields[list(vi)]

    ridge_samples = []
    for mn in MASK_NUMS:
        mask = load_mask(mn)
        train_obs = build_obs(train_f, mask)
        val_obs = build_obs(val_f, mask)
        om = np.mean(train_f, axis=0).ravel()
        os = np.std(train_f, axis=0).ravel() + 1e-8
        cm = np.mean(pod["coefficients"][ti], axis=0)
        cs = np.std(pod["coefficients"][ti], axis=0) + 1e-8

        def prep(X):
            n = X.shape[1]
            return np.concatenate([(X - om[:n]) / os[:n], np.ones((X.shape[0], 1))], axis=1)

        tX = prep(train_obs)
        vX = prep(val_obs)
        tY = (pod["coefficients"][ti] - cm) / cs
        vY = (pod["coefficients"][list(vi)] - cm) / cs
        XTX = tX.T @ tX; XTA = tX.T @ tY
        d = XTX.shape[0]
        best_W, best_loss = None, float("inf")
        for lam in LAMBDA_GRID:
            I = np.eye(d); I[-1, -1] = 0.0
            W_mat = np.linalg.solve(XTX + lam * I, XTA)
            loss = np.mean((vX @ W_mat - vY) ** 2)
            if loss < best_loss: best_loss, best_W = loss, W_mat

        for sigma, sc in zip(SIGMA_VALS, SIGMA_CODES):
            test_noisy = add_noise(test_f, sigma)
            test_obs = build_obs(test_noisy, mask)
            test_X = prep(test_obs)
            pc = test_X @ best_W
            pc = pc * cs + cm
            _H3, _W3, _C3 = 80, 160, 2
            bf = pod["pod_basis"].reshape(128, _H3 * _W3 * _C3).T
            mf = pod["mean_field"].ravel()
            pf = mf[np.newaxis, :] + (pc @ bf.T)
            pn = np.asarray(pf).reshape(-1, _H3, _W3, _C3).transpose(0, 3, 1, 2)
            tn = np.asarray(test_f).transpose(0, 3, 1, 2)

            for i in range(pn.shape[0]):
                t = tn[i].ravel(); p = pn[i].ravel()
                ger = np.linalg.norm(p - t) / (np.linalg.norm(t) + EPS)
                tgt2d = tn[i, 0]; out2d = pn[i, 0]
                ct = pywt.wavedec2(tgt2d, WAVELET, level=LEVEL, mode='periodization')
                co = pywt.wavedec2(out2d, WAVELET, level=LEVEL, mode='periodization')
                scount = 0
                if np.linalg.norm(co[0] - ct[0]) / (np.linalg.norm(ct[0]) + EPS) < TAU: scount += 1
                for do, dt in zip(co[1:], ct[1:]):
                    es = sum(np.sum((a - b)**2) for a, b in zip(do, dt))
                    ns = sum(np.sum(b**2) for b in dt)
                    if np.sqrt(es) / (np.sqrt(ns) + EPS) < TAU: scount += 1
                ridge_samples.append((float(ger), scount))

    # Store ridge data with proper config grouping
    # ridge_samples were collected in order: for each M, for each sigma, 300 samples
    idx = 0
    config_data["ridge"] = defaultdict(list)
    for mn in MASK_NUMS:
        for sigma in SIGMA_VALS:
            key = ("ridge", mn, sigma)
            for _ in range(300):  # 300 samples per config
                config_data["ridge"][key].append(ridge_samples[idx])
                idx += 1
    ridge_total = len(ridge_samples)
    print(f"  Ridge (closed-form): {ridge_total} records ({ridge_total//20} per config × 20 configs)")

    # ── 3. Run analysis ────────────────────────────────────────
    print("\n[3] Running low-GER analysis...")
    result = run_low_ger(config_data)

    print("\n  Results:")
    for k in ["ridge", "mlp", "vcnn", "all"]:
        v = result[k]
        print(f"  {k}: low_ger={v['low_ger_samples']}, sfull_lt3={v['sfull_lt_3']}, pct={v['pct']}%")

    # ── 4. Compare with old YAML values ────────────────────────
    old = {
        "ridge": {"low_ger_samples": 9000, "sfull_lt_3": 8405, "pct": 93.4},
        "mlp": {"low_ger_samples": 9000, "sfull_lt_3": 3304, "pct": 36.7},
        "vcnn": {"low_ger_samples": 9000, "sfull_lt_3": 3548, "pct": 39.4},
        "all": {"low_ger_samples": 27000, "sfull_lt_3": 15257, "pct": 56.5},
    }
    print("\n  Comparison (new → old):")
    for k in ["ridge", "mlp", "vcnn", "all"]:
        n, o = result[k], old[k]
        print(f"  {k}: low_ger {n['low_ger_samples']}→{o['low_ger_samples']}, "
              f"lt3 {n['sfull_lt_3']}→{o['sfull_lt_3']}, "
              f"pct {n['pct']}%→{o['pct']}%")

    # ── 5. Save ────────────────────────────────────────────────
    out = OUT_DIR / "s08b_low_ger_final.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")
    print("  ✓ Done")


if __name__ == "__main__":
    main()

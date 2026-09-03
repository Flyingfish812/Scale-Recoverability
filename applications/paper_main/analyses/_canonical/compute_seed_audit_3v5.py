#!/usr/bin/env python3
"""
Seed audit (fix v2): 3-seed vs 5-seed stability on representative conditions.

Reuses the official metric stack (luna.wavelet.metrics) and the existing
trained artifacts. Representative conditions (per 2026-08-26 feedback):
  clean      — M=20, sigma=0
  transition — M=30, sigma=0.01
  noisy      — M=20, sigma=0.1
Models: MLP, VCNN. Seeds: existing 0/101/202 + new 303/404.

Outputs
-------
    artifacts/derived/main/statistics/seed_audit_3v5.json
    artifacts/derived/main/statistics/seed_audit_3v5.csv

Usage
-----
    conda run -n sana python applications/paper_main/analyses/_canonical/compute_seed_audit_3v5.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_LEVEL, DEFAULT_MODE
from luna.pod.band_pod import fit_band_pod
from luna.wavelet.metrics import band_errors_all, compute_S_full, compute_S_coh, rel_l2

STATS_DIR = _PROJECT_ROOT / "artifacts" / "derived" / "main" / "statistics"
NC_DATA = _PROJECT_ROOT / "data" / "cylinder2d_q1.npy"

SEEDS_ALL = [0, 101, 202, 303, 404]
SEEDS_OLD = [0, 101, 202]
SIGMA_CODES = {0.0: "s0000", 0.001: "s0010", 0.01: "s0100", 0.1: "s1000"}
CONDITIONS = [
    ("clean", 20, 0.0),
    ("transition", 30, 0.01),
    ("noisy", 20, 0.1),
]
N_TRAIN_POD = 400
TAU = TAU_DEFAULT


def mlp_path(mask: int, seed: int, sigma: float) -> Path:
    return (_PROJECT_ROOT / "artifacts/pod_model_sweep_nc"
            / f"mlp_n{mask:04d}" / f"seed{seed:03d}" / "tests" / SIGMA_CODES[sigma] / "test_raw.npz")


def vcnn_path(mask: int, seed: int, sigma: float) -> Path:
    root = _PROJECT_ROOT / "artifacts/vcnn_results" / (
        "vcnn_sweep_nc_2000" if seed == 0 else f"vcnn_sweep_nc_2000_seed{seed:03d}")
    for inner in (f"vcnn_n{mask:04d}_seed{seed:03d}_custom", f"vcnn_n{mask:04d}_seed000_custom"):
        p = root / inner / "tests" / SIGMA_CODES[sigma] / "test_raw.npz"
        if p.exists():
            return p
    return root / f"vcnn_n{mask:04d}_seed{seed:03d}_custom" / "tests" / SIGMA_CODES[sigma] / "test_raw.npz"


def per_seed_metrics(model: str, mask: int, sigma: float, seed: int, band_pod) -> dict | None:
    path = mlp_path(mask, seed, sigma) if model == "mlp" else vcnn_path(mask, seed, sigma)
    if not path.exists():
        return None
    d = np.load(path)
    u = d["target_nchw"][:, 0, :, :].astype(np.float64)
    uh = d["output_nchw"][:, 0, :, :].astype(np.float64)
    gers, sfs, scohs = [], [], []
    errs = {b: [] for b in BANDS_CF}
    for i in range(u.shape[0]):
        gers.append(rel_l2(uh[i], u[i]))
        e = band_errors_all(u[i], uh[i], "db2", DEFAULT_LEVEL, DEFAULT_MODE)
        for b in BANDS_CF:
            errs[b].append(e[b])
        sfs.append(compute_S_full(u[i], uh[i], TAU, "db2", DEFAULT_LEVEL, DEFAULT_MODE))
        scohs.append(compute_S_coh(u[i], uh[i], band_pod, TAU, "db2", DEFAULT_LEVEL, DEFAULT_MODE))
    return {
        "GER_mean": float(np.mean(gers)),
        "S_full_mean": float(np.mean(sfs)),
        "S_full_std": float(np.std(sfs)),
        "S_full_mode": int(np.bincount(np.asarray(sfs, dtype=int)).argmax()),
        "S_coh_mean": float(np.mean(scohs)),
        "S_coh_mode": int(np.bincount(np.asarray(scohs, dtype=int)).argmax()),
        "E_direct_mean": {b: float(np.mean(errs[b])) for b in BANDS_CF},
    }


def main() -> None:
    # band-POD (same protocol as compute_p0_2_wavelet_sensitivity.py)
    ref = np.load(mlp_path(20, 0, 0.0))
    test_idx = sorted(set(ref["test_indices"].tolist()))
    all_fields = np.load(NC_DATA)
    train_idx = sorted(set(range(all_fields.shape[0])) - set(test_idx))
    rng = np.random.RandomState(7)
    sub = sorted(rng.choice(train_idx, min(N_TRAIN_POD, len(train_idx)), replace=False))
    band_pod = fit_band_pod(all_fields[sub][:, :, :, 0].astype(np.float64), pod_energy_threshold=0.99,
                            wavelet="db2", level=DEFAULT_LEVEL, mode=DEFAULT_MODE)

    rows_per_seed = []
    summary = {}
    for cond, mask, sigma in CONDITIONS:
        summary[cond] = {}
        for model in ("mlp", "vcnn"):
            per = {}
            for seed in SEEDS_ALL:
                m = per_seed_metrics(model, mask, sigma, seed, band_pod)
                if m is None:
                    print(f"[warn] missing: {model} M{mask} sigma={sigma} seed={seed}")
                    continue
                per[seed] = m
                rows_per_seed.append({"condition": cond, "M": mask, "sigma": sigma,
                                      "model": model, "seed": seed, **{k: v for k, v in m.items()
                                                                       if k != "E_direct_mean"}})
            if not per:
                continue
            # 3-seed vs 5-seed comparison on S_full / GER
            def agg(seeds, key):
                vals = [per[s][key] for s in seeds if s in per]
                return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

            sf3, sd3 = agg(SEEDS_OLD, "S_full_mean")
            sf5, sd5 = agg(SEEDS_ALL, "S_full_mean")
            ge3, gsd3 = agg(SEEDS_OLD, "GER_mean")
            ge5, gsd5 = agg(SEEDS_ALL, "GER_mean")
            summary[cond][model] = {
                "n_seeds_old": len([s for s in SEEDS_OLD if s in per]),
                "n_seeds_all": len(per),
                "S_full_mean_3sd": sf3, "S_full_sd_across_seeds_3": sd3,
                "S_full_mean_5sd": sf5, "S_full_sd_across_seeds_5": sd5,
                "GER_mean_3sd": ge3, "GER_sd_across_seeds_3": gsd3,
                "GER_mean_5sd": ge5, "GER_sd_across_seeds_5": gsd5,
                "per_seed": {str(s): per[s] for s in sorted(per)},
                "ranking_3sd": "MLP>VCNN" if per.get(0, {}).get("S_full_mean", 0) >= per.get(0, {}).get("S_full_mean", 0) else "check",
            }
            print(f"{cond:10s} {model:5s} | S_full 3-seed {sf3:.3f}±{sd3:.3f} -> 5-seed {sf5:.3f}±{sd5:.3f}"
                  f" | GER 3-seed {ge3:.5f}±{gsd3:.5f} -> 5-seed {ge5:.5f}±{gsd5:.5f}")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    (STATS_DIR / "seed_audit_3v5.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    with open(STATS_DIR / "seed_audit_3v5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_per_seed[0].keys()))
        w.writeheader()
        for r in rows_per_seed:
            w.writerow(r)
    print(f"[ok] {STATS_DIR / 'seed_audit_3v5.json'}")


if __name__ == "__main__":
    main()

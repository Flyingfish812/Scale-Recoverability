#!/usr/bin/env python3
"""
=============================================================================
P1-4: 标准 Fourier Spectral Baseline (2026-07-15 审计反馈)
=============================================================================

Problem:
    当前 Fourier 对照只包含 S_FFT (dyadic count)，缺少导师要求的常用指标:
      - L_spectral: 功率谱的规范化差异
      - RMSE_low: 低频 (k < k_cut) RMSE
      - RMSE_high: 高频 (k ≥ k_cut) RMSE

Fix:
    定义并计算标准 Fourier 指标:
      1. L_spectral = ‖|FFT(u)| − |FFT(û)|‖₂ / ‖|FFT(u)|‖₂
      2. RMSE_low:  k < k_cut 区域的 RMSE
      3. RMSE_high: k ≥ k_cut 区域的 RMSE
      4. 在 equal-GER 案例中报告这些指标
      5. 计算这些指标与 S_full、S_FFT 的相关性

Output:
    results/20260715/fourier_spectral_baseline.json
    results/20260715/fourier_spectral_baseline.csv

Usage:
    conda run -n sana python3 scripts/20260715/04_fourier_spectral_baseline.py
    conda run -n sana python3 scripts/20260715/04_fourier_spectral_baseline.py --lite

Dependencies: luna.*, numpy, scipy (sana 环境已安装)
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.core.constants import (
    BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE, EPS,
)
from luna.data.io import load_npz, save_json
from luna.pod.oracle import pod_oracle_reconstruct
from luna.wavelet.metrics import band_errors_all, compute_S_full


# ══════════════════════════════════════════════════════════════════════
# Fourier Metrics — Standard Definitions
# ══════════════════════════════════════════════════════════════════════

def spectral_loss(target: np.ndarray, pred: np.ndarray) -> float:
    """L_spectral: Normalized L2 error of Fourier magnitude spectra.

    L_spec = ‖|F(u)| − |F(û)|‖₂ / ‖|F(u)|‖₂

    This measures global spectral energy distribution discrepancy.
    """
    F_u = np.abs(np.fft.fft2(target))
    F_hat = np.abs(np.fft.fft2(pred))
    num = np.linalg.norm((F_u - F_hat).ravel())
    den = np.linalg.norm(F_u.ravel())
    return float(num / (den + EPS))


def fourier_low_high_rmse(
    target: np.ndarray,
    pred: np.ndarray,
    cutoff_ratio: float = 0.5,
) -> dict[str, float]:
    """Low-frequency and high-frequency RMSE in Fourier domain.

    Splits the 2D frequency plane at k_cutoff = cutoff_ratio * k_max.
      Low:  ‖k‖ ≤ k_cutoff
      High: ‖k‖ > k_cutoff

    Returns:
        {"low_rmse": ..., "high_rmse": ..., "low_rmse_rel": ..., "high_rmse_rel": ...}
    """
    H, W = target.shape
    F_u = np.fft.fft2(target)
    F_hat = np.fft.fft2(pred)
    diff = F_u - F_hat

    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(KX**2 + KY**2)

    k_max = kmag.max()
    k_cutoff = cutoff_ratio * k_max

    low_mask = kmag <= k_cutoff
    high_mask = kmag > k_cutoff

    # Absolute RMSE
    low_rmse = float(np.sqrt(np.mean(np.abs(diff[low_mask]) ** 2))) if low_mask.any() else 0.0
    high_rmse = float(np.sqrt(np.mean(np.abs(diff[high_mask]) ** 2))) if high_mask.any() else 0.0

    # Relative RMSE (normalized by target energy in that region)
    low_target_norm = float(np.sqrt(np.mean(np.abs(F_u[low_mask]) ** 2))) if low_mask.any() else 1.0
    high_target_norm = float(np.sqrt(np.mean(np.abs(F_u[high_mask]) ** 2))) if high_mask.any() else 1.0

    low_rmse_rel = low_rmse / (low_target_norm + EPS)
    high_rmse_rel = high_rmse / (high_target_norm + EPS)

    return {
        "low_rmse": low_rmse,
        "high_rmse": high_rmse,
        "low_rmse_rel": low_rmse_rel,
        "high_rmse_rel": high_rmse_rel,
    }


def fourier_dyadic_band_errors(
    target: np.ndarray,
    pred: np.ndarray,
    n_bands: int = 5,
) -> dict[str, float]:
    """Per-band Fourier errors using dyadic wavenumber partitioning.

    Returns relative L2 error per Fourier band (coarse → fine).
    """
    H, W = target.shape
    F_u = np.fft.fft2(target)
    F_hat = np.fft.fft2(pred)
    diff = F_u - F_hat

    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(KX**2 + KY**2)
    k_nyq = kmag.max()

    errors: dict[str, float] = {}
    for i in range(n_bands):
        if i == 0:
            k_low, k_high = 0.0, k_nyq / (2 ** (n_bands - 1))
        else:
            k_low = k_nyq / (2 ** (n_bands - i))
            k_high = k_nyq / (2 ** (n_bands - i - 1))

        band_mask = (kmag >= k_low) & (kmag < k_high)

        if band_mask.any():
            err = float(np.sqrt(np.mean(np.abs(diff[band_mask]) ** 2)))
            target_norm = float(np.sqrt(np.mean(np.abs(F_u[band_mask]) ** 2)))
            rel_err = err / (target_norm + EPS)
        else:
            rel_err = 0.0

        errors[f"F{n_bands - 1 - i}"] = rel_err  # F4=coarsest, F0=finest

    return errors


def compute_S_FFT(
    target: np.ndarray,
    pred: np.ndarray,
    tau: float = TAU_DEFAULT,
    n_bands: int = 5,
) -> int:
    """S_FFT: contiguous recoverable Fourier bands (analogous to S_full)."""
    errors = fourier_dyadic_band_errors(target, pred, n_bands)
    band_order = [f"F{n_bands - 1 - i}" for i in range(n_bands)]
    err_arr = np.array([errors.get(b, np.inf) for b in band_order])
    # contiguous_recoverable_index logic
    out = 0
    for i, v in enumerate(err_arr, start=1):
        if np.isfinite(v) and v <= tau:
            out = i
        else:
            break
    return out


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_vcnn_samples(
    npz_path: str,
    n_samples: int,
    project_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load VCNN test samples and de-normalize.

    Returns (target_ch0, output_ch0) as (N, H, W) arrays in physical units.
    """
    data = np.load(str(project_root / npz_path), allow_pickle=True)
    target_all = np.asarray(data["target_nchw"], dtype=np.float64)
    output_all = np.asarray(data["output_nchw"], dtype=np.float64)
    N_avail = target_all.shape[0]
    N_use = min(n_samples, N_avail)

    NC_MEAN = np.array([1.0004944, -0.00017817653], dtype=np.float64)
    NC_STD = np.array([0.21863055, 0.19121747], dtype=np.float64)

    targets = np.zeros((N_use, target_all.shape[2], target_all.shape[3]), dtype=np.float64)
    outputs = np.zeros_like(targets)

    for i in range(N_use):
        t_phys = (target_all[i] * NC_STD[:, None, None] + NC_MEAN[:, None, None]).astype(np.float64)
        o_phys = (output_all[i] * NC_STD[:, None, None] + NC_MEAN[:, None, None]).astype(np.float64)
        targets[i] = t_phys[0]  # channel 0
        outputs[i] = o_phys[0]

    return targets, outputs


# ══════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════

def run_fourier_spectral_analysis(
    test_npz_path: str,
    n_samples: int,
    tau: float,
    wavelet: str,
    level: int,
    mode: int,
    n_fourier_bands: int,
    project_root: Path,
    lite: bool = False,
) -> dict:
    """Compute all Fourier spectral metrics and compare with wavelet S_full."""
    # Load data
    print(f"  Loading: {test_npz_path}")
    targets, outputs = load_vcnn_samples(test_npz_path, n_samples, project_root)
    N = targets.shape[0]
    print(f"  Samples: {N}")

    # Per-sample metrics
    results = []
    for i in range(N):
        u = targets[i]
        u_hat = outputs[i]

        # Wavelet metrics
        s_full = compute_S_full(u, u_hat, tau, wavelet, level, mode)
        wav_errors = band_errors_all(u, u_hat, wavelet, level, mode)

        # Fourier metrics
        l_spec = spectral_loss(u, u_hat)
        lh_rmse = fourier_low_high_rmse(u, u_hat)
        s_fft = compute_S_FFT(u, u_hat, tau, n_fourier_bands)
        fft_errors = fourier_dyadic_band_errors(u, u_hat, n_fourier_bands)
        fft_lh_rel = fourier_low_high_rmse(u, u_hat)

        results.append({
            "sample_idx": i,
            "GER": float(np.linalg.norm((u - u_hat).ravel()) / (np.linalg.norm(u.ravel()) + EPS)),
            "S_full": s_full,
            "S_FFT": s_fft,
            "spectral_loss": l_spec,
            "fourier_low_rmse": lh_rmse["low_rmse"],
            "fourier_high_rmse": lh_rmse["high_rmse"],
            "fourier_low_rmse_rel": lh_rmse["low_rmse_rel"],
            "fourier_high_rmse_rel": lh_rmse["high_rmse_rel"],
            **{f"wav_{b}": wav_errors[b] for b in BANDS_CF},
            **{f"fft_{b}": fft_errors.get(b.replace("W", "F").replace("A4", "F4"), np.nan)
               for b in BANDS_CF},
        })

    # ── Aggregate summary ──────────────────────────────────────────
    agg = {}
    for key in results[0]:
        if key == "sample_idx":
            continue
        vals = [r[key] for r in results if np.isfinite(r[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_median"] = float(np.median(vals))

    # ── Correlation analysis ───────────────────────────────────────
    corr_results = {}
    metric_pairs = [
        ("S_full", "S_FFT"),
        ("S_full", "spectral_loss"),
        ("S_full", "GER"),
        ("S_FFT", "spectral_loss"),
        ("S_full", "fourier_low_rmse_rel"),
        ("S_full", "fourier_high_rmse_rel"),
        ("GER", "spectral_loss"),
    ]

    for m1, m2 in metric_pairs:
        v1 = np.array([r[m1] for r in results])
        v2 = np.array([r[m2] for r in results])
        valid = np.isfinite(v1) & np.isfinite(v2)
        if valid.sum() >= 3:
            rho, p_val = stats.spearmanr(v1[valid], v2[valid])
            corr_results[f"{m1}_vs_{m2}"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_valid": int(valid.sum()),
            }
        else:
            corr_results[f"{m1}_vs_{m2}"] = {
                "spearman_rho": None,
                "p_value": None,
                "n_valid": int(valid.sum()),
            }

    # ── Equal-GER subset analysis ──────────────────────────────────
    # Find pairs with similar GER and check if S_full vs spectral metrics differ
    equal_ger_analysis = None
    if N >= 20:
        equal_ger_analysis = analyze_equal_ger_subset(results)

    return {
        "n_samples": N,
        "config": {
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
            "tau": tau,
            "n_fourier_bands": n_fourier_bands,
        },
        "aggregates": agg,
        "correlations": corr_results,
        "equal_ger_analysis": equal_ger_analysis,
        "results": results,
    }


def analyze_equal_ger_subset(results: list[dict]) -> dict:
    """Analyze samples with similar GER but different S_full.

    Finds pairs with GER difference < 5% and S_full difference ≥ 2.
    Compares their spectral metrics.
    """
    n = len(results)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ger_i = results[i]["GER"]
            ger_j = results[j]["GER"]
            s_i = results[i]["S_full"]
            s_j = results[j]["S_full"]

            ger_diff = abs(ger_i - ger_j)
            ger_min = min(ger_i, ger_j)
            if ger_diff > 0.05 * ger_min:
                continue
            if abs(s_i - s_j) < 2:
                continue

            if s_i < s_j:
                lo, hi = i, j
            else:
                lo, hi = j, i

            pairs.append({
                "GER_low_S_full": results[lo]["GER"],
                "GER_high_S_full": results[hi]["GER"],
                "S_full_low": results[lo]["S_full"],
                "S_full_high": results[hi]["S_full"],
                "spectral_loss_low": results[lo]["spectral_loss"],
                "spectral_loss_high": results[hi]["spectral_loss"],
                "low_rmse_rel_low": results[lo]["fourier_low_rmse_rel"],
                "low_rmse_rel_high": results[hi]["fourier_low_rmse_rel"],
                "high_rmse_rel_low": results[lo]["fourier_high_rmse_rel"],
                "high_rmse_rel_high": results[hi]["fourier_high_rmse_rel"],
            })

    if not pairs:
        return {"n_pairs_found": 0}

    # Aggregate statistics
    spec_loss_diffs = [abs(p["spectral_loss_low"] - p["spectral_loss_high"]) for p in pairs]
    low_rmse_diffs = [abs(p["low_rmse_rel_low"] - p["low_rmse_rel_high"]) for p in pairs]
    high_rmse_diffs = [abs(p["high_rmse_rel_low"] - p["high_rmse_rel_high"]) for p in pairs]

    return {
        "n_pairs_found": len(pairs),
        "mean_spectral_loss_diff": float(np.mean(spec_loss_diffs)),
        "median_spectral_loss_diff": float(np.median(spec_loss_diffs)),
        "mean_low_rmse_rel_diff": float(np.mean(low_rmse_diffs)),
        "mean_high_rmse_rel_diff": float(np.mean(high_rmse_diffs)),
        "representative_pair": pairs[0] if pairs else None,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def print_summary(result: dict) -> None:
    """Print human-readable summary."""
    print(f"\n{'='*60}")
    print("  FOURIER SPECTRAL BASELINE")
    print(f"{'='*60}")
    print(f"  Samples: {result['n_samples']}")

    agg = result["aggregates"]
    print(f"\n  Aggregate metrics:")
    print(f"    S_full:           {agg.get('S_full_mean', 'N/A'):.2f} ± {agg.get('S_full_std', 0):.2f}")
    print(f"    S_FFT:            {agg.get('S_FFT_mean', 'N/A'):.2f} ± {agg.get('S_FFT_std', 0):.2f}")
    print(f"    Spectral Loss:    {agg.get('spectral_loss_mean', 'N/A'):.4f}")
    print(f"    Low RMSE (rel):   {agg.get('fourier_low_rmse_rel_mean', 'N/A'):.4f}")
    print(f"    High RMSE (rel):  {agg.get('fourier_high_rmse_rel_mean', 'N/A'):.4f}")

    print(f"\n  Correlations with S_full:")
    for pair_name, corr in result.get("correlations", {}).items():
        if "S_full_vs" in pair_name or "vs_S_full" in pair_name:
            rho = corr.get("spearman_rho")
            if rho is not None:
                print(f"    {pair_name}: ρ={rho:.3f} (p={corr['p_value']:.4f})")

    eq = result.get("equal_ger_analysis")
    if eq:
        print(f"\n  Equal-GER subset analysis:")
        print(f"    Pairs found (similar GER, different S_full): {eq['n_pairs_found']}")
        if eq.get("mean_spectral_loss_diff") is not None:
            print(f"    Mean spectral loss diff within pairs: {eq['mean_spectral_loss_diff']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fourier Spectral Baseline — Standard Metrics")
    parser.add_argument("--test-npz", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--n-fourier-bands", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/derived/main/statistics")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if args.lite:
        args.n_samples = 10
        print("[LITE MODE] 10 samples")

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default test case: VCNN M=20, σ=0
    default_npz = (
        "artifacts/vcnn_results/vcnn_sweep_nc_2000/"
        "vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz"
    )
    test_npz_path = args.test_npz or default_npz

    if not (project_root / test_npz_path).exists():
        print(f"ERROR: {test_npz_path} not found.")
        sys.exit(1)

    t0 = time.time()
    result = run_fourier_spectral_analysis(
        test_npz_path=test_npz_path,
        n_samples=args.n_samples,
        tau=args.tau,
        wavelet=args.wavelet,
        level=args.level,
        mode=args.mode,
        n_fourier_bands=args.n_fourier_bands,
        project_root=project_root,
        lite=args.lite,
    )
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    print_summary(result)

    # Save JSON (exclude per-sample results for size)
    save_dict = {k: v for k, v in result.items() if k != "results"}
    if result.get("results"):
        save_dict["n_total_samples"] = len(result["results"])
        save_dict["results_summary"] = {
            "GER_range": [float(np.min([r["GER"] for r in result["results"]])),
                          float(np.max([r["GER"] for r in result["results"]]))],
            "S_full_range": [int(np.min([r["S_full"] for r in result["results"]])),
                             int(np.max([r["S_full"] for r in result["results"]]))],
        }
    json_path = output_dir / "fourier_spectral_baseline.json"
    save_json(str(json_path), save_dict)
    print(f"JSON saved to: {json_path}")

    # Save CSV
    csv_path = output_dir / "fourier_spectral_baseline.csv"
    if result.get("results"):
        fieldnames = list(result["results"][0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["results"])
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

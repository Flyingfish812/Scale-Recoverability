#!/usr/bin/env python3
"""
=============================================================================
P1-5: Fourier-小波对称控制实验 (2026-07-15 审计反馈)
=============================================================================

Problem:
    当前受控实验是用 wavelet bands 构造目标，再验证 wavelet 指标是否识别
    这些 bands，这天然偏向 S_full。缺少对称的 Fourier-domain 控制实验。

Fix:
    构造三种受控目标，比较 S_full 和 S_FFT 的识别能力:
      1. Wavelet-band-limited targets (已有) — 只有指定 wavelet band 的能量
      2. Fourier-annulus-limited targets (新增) — 只有指定 Fourier annulus 的能量
      3. Local spatial perturbation targets (新增) — 局部高频缺失或局部尾涡扰动

    预期结论:
      - Fourier 指标更直接衡量全局频率能量偏差
      - Wavelet 指标同时保留尺度和空间局部信息
      - S_full 在局部空间扰动下仍能正确诊断，而 Fourier 指标可能低估局部问题

Output:
    results/20260715/symmetric_control.json
    results/20260715/symmetric_control.csv

Usage:
    conda run -n sana python3 scripts/20260715/05_fourier_wavelet_symmetric.py
    conda run -n sana python3 scripts/20260715/05_fourier_wavelet_symmetric.py --lite

Dependencies: luna.*, numpy, pywt (sana 环境已安装)
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
import pywt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.core.constants import (
    BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE, EPS,
)
from luna.data.io import load_npy, save_json
from luna.wavelet.transform import decompose_field_2d, zero_coeff_like
from luna.wavelet.metrics import compute_S_full, band_errors_all
from luna.data.io import load_npz


# ══════════════════════════════════════════════════════════════════════
# Target Construction Functions
# ══════════════════════════════════════════════════════════════════════

def build_wavelet_band_target(
    field: np.ndarray,
    keep_bands: list[str],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    """Build target containing only specified wavelet bands.

    Args:
        field: Source 2D field (H, W).
        keep_bands: List of band names to keep (e.g. ['A4'], ['A4', 'W4']).
        wavelet, level, mode: Wavelet parameters.

    Returns:
        (H, W) filtered field with only the specified bands.
    """
    coeffs = pywt.wavedec2(field, wavelet=wavelet, level=level, mode=mode)
    h, w = field.shape

    # Zero out all coefficients first
    result = np.zeros_like(field, dtype=np.float64)

    for band_name in keep_bands:
        if band_name == "A4":
            c = zero_coeff_like(coeffs)
            c[0] = coeffs[0]
            component = pywt.waverec2(c, wavelet=wavelet, mode=mode)
            result += component[:h, :w]
        elif band_name.startswith("W"):
            idx = int(band_name[1:])  # W4 → 4
            i = len(coeffs) - idx  # index from end
            if 1 <= i < len(coeffs):
                c = zero_coeff_like(coeffs)
                c[i] = coeffs[i]
                component = pywt.waverec2(c, wavelet=wavelet, mode=mode)
                result += component[:h, :w]

    return np.asarray(result, dtype=np.float64)


def build_fourier_annulus_target(
    field: np.ndarray,
    keep_annuli: list[int],
    n_annuli: int = 5,
) -> np.ndarray:
    """Build target containing only specified Fourier annuli.

    Args:
        field: Source 2D field (H, W).
        keep_annuli: List of annulus indices to keep (0=coarsest, n_annuli-1=finest).
        n_annuli: Total number of annuli.

    Returns:
        (H, W) Fourier-filtered field.
    """
    H, W = field.shape
    F = np.fft.fft2(field)

    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(KX**2 + KY**2)
    k_nyq = kmag.max()

    keep_mask = np.zeros_like(kmag, dtype=bool)
    for ai in keep_annuli:
        if ai == 0:
            k_low, k_high = 0.0, k_nyq / (2 ** (n_annuli - 1))
        else:
            k_low = k_nyq / (2 ** (n_annuli - ai))
            k_high = k_nyq / (2 ** (n_annuli - ai - 1))
        keep_mask |= (kmag >= k_low) & (kmag < k_high)

    F_filtered = F * keep_mask
    return np.real(np.fft.ifft2(F_filtered)).astype(np.float64)


def build_local_perturbation_target(
    field: np.ndarray,
    perturbation_type: str = "high_freq_dropout",
    region: str = "wake",
    severity: float = 0.5,
) -> np.ndarray:
    """Build target with localized spatial perturbation.

    Args:
        field: Source 2D field (H, W).
        perturbation_type:
            "high_freq_dropout": Remove high-frequency content in a local region.
            "wake_perturb": Add noise in the wake region.
        region: "wake" (right portion), "center", "random_block".
        severity: Strength of perturbation (0-1).

    Returns:
        (H, W) perturbed field.
    """
    H, W = field.shape
    result = field.copy()

    # Define region mask
    if region == "wake":
        # Right half of domain (cylinder wake)
        mask = np.zeros((H, W), dtype=bool)
        mask[:, W // 2:] = True
    elif region == "center":
        # Central block
        h_start, h_end = H // 4, 3 * H // 4
        w_start, w_end = W // 4, 3 * W // 4
        mask = np.zeros((H, W), dtype=bool)
        mask[h_start:h_end, w_start:w_end] = True
    elif region == "random_block":
        # Random block (deterministic seed 42)
        rng = np.random.RandomState(42)
        h_start = rng.randint(0, H // 2)
        w_start = rng.randint(0, W // 2)
        h_size = H // 4
        w_size = W // 4
        mask = np.zeros((H, W), dtype=bool)
        mask[h_start:h_start + h_size, w_start:w_start + w_size] = True
    else:
        mask = np.ones((H, W), dtype=bool)

    if perturbation_type == "high_freq_dropout":
        # Remove high frequencies in the masked region
        F = np.fft.fft2(field)
        ky = np.fft.fftfreq(H) * H
        kx = np.fft.fftfreq(W) * W
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        kmag = np.sqrt(KX**2 + KY**2)
        k_nyq = kmag.max()

        # Low-pass filter: keep only k < severity * k_nyq
        low_pass = kmag < severity * k_nyq

        # Apply low-pass only in masked region
        # Convert to spatial domain: for each pixel in mask, replace with low-passed version
        F_low = F * low_pass
        field_low = np.real(np.fft.ifft2(F_low)).astype(np.float64)

        result[mask] = field_low[mask]

    elif perturbation_type == "wake_perturb":
        # Add spatially correlated noise in the masked region
        rng = np.random.RandomState(42)
        noise = rng.randn(H, W).astype(np.float64)
        # Smooth noise
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=3.0)
        noise = noise / np.std(noise) * severity * np.std(field)
        result[mask] += noise[mask]

    return result


# ══════════════════════════════════════════════════════════════════════
# Fourier metrics (same definitions as 04_fourier_spectral_baseline.py)
# ══════════════════════════════════════════════════════════════════════

def spectral_loss(target: np.ndarray, pred: np.ndarray) -> float:
    F_u = np.abs(np.fft.fft2(target))
    F_hat = np.abs(np.fft.fft2(pred))
    return float(np.linalg.norm((F_u - F_hat).ravel()) / (np.linalg.norm(F_u.ravel()) + EPS))


def fourier_dyadic_band_errors(target: np.ndarray, pred: np.ndarray, n_bands: int = 5) -> dict[str, float]:
    H, W = target.shape
    F_u = np.fft.fft2(target)
    F_hat = np.fft.fft2(pred)
    diff = F_u - F_hat
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(KX**2 + KY**2)
    k_nyq = kmag.max()
    errors = {}
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
        errors[f"F{n_bands - 1 - i}"] = rel_err
    return errors


def compute_S_FFT(target: np.ndarray, pred: np.ndarray, tau: float = TAU_DEFAULT, n_bands: int = 5) -> int:
    errors = fourier_dyadic_band_errors(target, pred, n_bands)
    band_order = [f"F{n_bands - 1 - i}" for i in range(n_bands)]
    err_arr = np.array([errors.get(b, np.inf) for b in band_order])
    out = 0
    for i, v in enumerate(err_arr, start=1):
        if np.isfinite(v) and v <= tau:
            out = i
        else:
            break
    return out


# ══════════════════════════════════════════════════════════════════════
# Controlled validation experiment
# ══════════════════════════════════════════════════════════════════════

def run_symmetric_control(
    field: np.ndarray,
    tau: float,
    wavelet: str,
    level: int,
    mode: str,
    n_fourier_bands: int,
) -> list[dict]:
    """Run symmetric control experiments comparing S_full vs S_FFT.

    Tests three families of controlled targets and evaluates whether
    each metric correctly identifies the known scale content.

    Args:
        field: Source 2D field (H, W).

    Returns:
        List of result dicts, one per test configuration.
    """
    results = []
    H, W = field.shape

    # ── 1. Wavelet-band-limited targets ────────────────────────────
    wavelet_configs = {
        "A4 only": ["A4"],
        "A4+W4": ["A4", "W4"],
        "A4+W4+W3": ["A4", "W4", "W3"],
        "A4+W4+W3+W2": ["A4", "W4", "W3", "W2"],
        "All bands": ["A4", "W4", "W3", "W2", "W1"],
    }

    for label, keep_bands in wavelet_configs.items():
        target = build_wavelet_band_target(field, keep_bands, wavelet, level, mode)
        expected = len(keep_bands)

        s_full = compute_S_full(target, field, tau, wavelet, level, mode)
        s_fft = compute_S_FFT(target, field, tau, n_fourier_bands)
        ger = float(np.linalg.norm((target - field).ravel()) / (np.linalg.norm(field.ravel()) + EPS))

        results.append({
            "experiment": "wavelet_band_limited",
            "config_label": label,
            "kept_components": str(keep_bands),
            "expected_recoverable": expected,
            "S_full": s_full,
            "S_FFT": s_fft,
            "GER": ger,
            "S_full_correct": s_full == expected,
            "S_FFT_correct": s_fft == expected,
        })

    # ── 2. Fourier-annulus-limited targets ─────────────────────────
    fourier_configs = {
        "F4 only (coarsest)": [0],
        "F4+F3": [0, 1],
        "F4+F3+F2": [0, 1, 2],
        "F4+F3+F2+F1": [0, 1, 2, 3],
        "All annuli": [0, 1, 2, 3, 4],
    }

    for label, keep_annuli in fourier_configs.items():
        target = build_fourier_annulus_target(field, keep_annuli, n_fourier_bands)
        expected = len(keep_annuli)

        s_full = compute_S_full(target, field, tau, wavelet, level, mode)
        s_fft = compute_S_FFT(target, field, tau, n_fourier_bands)
        ger = float(np.linalg.norm((target - field).ravel()) / (np.linalg.norm(field.ravel()) + EPS))

        results.append({
            "experiment": "fourier_annulus_limited",
            "config_label": label,
            "kept_components": str(keep_annuli),
            "expected_recoverable": expected,
            "S_full": s_full,
            "S_FFT": s_fft,
            "GER": ger,
            "S_full_correct": s_full == expected,
            "S_FFT_correct": s_fft == expected,
        })

    # ── 3. Local spatial perturbation targets ──────────────────────
    local_configs = [
        ("high_freq_dropout_wake_mild", "high_freq_dropout", "wake", 0.3),
        ("high_freq_dropout_wake_moderate", "high_freq_dropout", "wake", 0.5),
        ("high_freq_dropout_wake_severe", "high_freq_dropout", "wake", 0.7),
        ("wake_perturb_moderate", "wake_perturb", "wake", 0.3),
        ("wake_perturb_severe", "wake_perturb", "wake", 0.5),
    ]

    for label, ptype, region, severity in local_configs:
        target = build_local_perturbation_target(field, ptype, region, severity)

        # Expected: depends on perturbation severity
        # For high_freq_dropout, fine scales are lost → S_full should drop
        # For wake_perturb, all scales may be affected
        s_full = compute_S_full(target, field, tau, wavelet, level, mode)
        s_fft = compute_S_FFT(target, field, tau, n_fourier_bands)
        ger = float(np.linalg.norm((target - field).ravel()) / (np.linalg.norm(field.ravel()) + EPS))

        # Per-band errors for detailed diagnosis
        band_errs = band_errors_all(target, field, wavelet, level, mode)

        results.append({
            "experiment": "local_spatial_perturbation",
            "config_label": label,
            "kept_components": f"{ptype},{region},{severity}",
            "expected_recoverable": None,  # unknown a priori
            "S_full": s_full,
            "S_FFT": s_fft,
            "GER": ger,
            **{f"E_wav_{b}": band_errs[b] for b in BANDS_CF},
            "S_full_correct": None,
            "S_FFT_correct": None,
        })

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def print_results(results: list[dict]) -> None:
    """Print formatted results."""
    print(f"\n{'='*60}")
    print("  SYMMETRIC CONTROL EXPERIMENT")
    print(f"{'='*60}")

    for exp_type in ["wavelet_band_limited", "fourier_annulus_limited", "local_spatial_perturbation"]:
        subset = [r for r in results if r["experiment"] == exp_type]
        if not subset:
            continue

        print(f"\n  --- {exp_type} ---")
        print(f"  {'Config':<30} {'Expected':>8} {'S_full':>8} {'S_FFT':>8} {'GER':>10} "
              f"{'S_full✓':>8} {'S_FFT✓':>8}")
        print(f"  {'-'*80}")

        for r in subset:
            exp = r["expected_recoverable"]
            exp_str = str(exp) if exp is not None else "N/A"
            sc = "✓" if r["S_full_correct"] else "✗" if r["S_full_correct"] is not None else "—"
            fc = "✓" if r["S_FFT_correct"] else "✗" if r["S_FFT_correct"] is not None else "—"
            print(f"  {r['config_label']:<30} {exp_str:>8} {r['S_full']:>8} {r['S_FFT']:>8} "
                  f"{r['GER']:>10.4f} {sc:>8} {fc:>8}")

        if exp_type == "local_spatial_perturbation":
            print(f"\n  Per-band errors for local perturbations:")
            print(f"  {'Config':<35} ", end="")
            for b in BANDS_CF:
                print(f"{b:>8}", end="")
            print()
            for r in subset:
                print(f"  {r['config_label']:<35} ", end="")
                for b in BANDS_CF:
                    val = r.get(f"E_wav_{b}", 0)
                    print(f"{val:>8.4f}", end="")
                print()

    # Summary: how often does each metric match expectation?
    print(f"\n  --- Accuracy Summary ---")
    for exp_type in ["wavelet_band_limited", "fourier_annulus_limited"]:
        subset = [r for r in results if r["experiment"] == exp_type and r["expected_recoverable"] is not None]
        if not subset:
            continue
        s_full_acc = sum(1 for r in subset if r["S_full_correct"]) / len(subset) * 100
        s_fft_acc = sum(1 for r in subset if r["S_FFT_correct"]) / len(subset) * 100
        print(f"  {exp_type}:")
        print(f"    S_full correct: {s_full_acc:.0f}% ({sum(1 for r in subset if r['S_full_correct'])}/{len(subset)})")
        print(f"    S_FFT correct:  {s_fft_acc:.0f}% ({sum(1 for r in subset if r['S_FFT_correct'])}/{len(subset)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fourier-Wavelet Symmetric Control Experiment")
    parser.add_argument("--data-path", default="data/cylinder2d_q1.npy")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--n-fourier-bands", type=int, default=5)
    parser.add_argument("--n-test-fields", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/derived/main/statistics")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if args.lite:
        args.n_test_fields = 2
        print("[LITE MODE] 2 test fields")

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = project_root / args.data_path
    print(f"Loading data: {data_path}")
    fields = load_npy(str(data_path))  # (T, H, W, C) or (T, H, W)
    print(f"Data shape: {fields.shape}")

    # Extract single-channel fields
    if fields.ndim == 4:
        test_fields = fields[:args.n_test_fields, :, :, 0].astype(np.float64)
    else:
        test_fields = fields[:args.n_test_fields].astype(np.float64)

    print(f"Test fields: {test_fields.shape}")

    # Run control experiment for each field
    all_results = []
    t0 = time.time()

    for i in range(test_fields.shape[0]):
        field = test_fields[i]
        results = run_symmetric_control(
            field, args.tau, args.wavelet, args.level, args.mode, args.n_fourier_bands,
        )
        for r in results:
            r["field_idx"] = i
        all_results.extend(results)
        print(f"  Field {i + 1}/{test_fields.shape[0]} done")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    # Print summary
    print_results(all_results)

    # Save JSON
    json_path = output_dir / "symmetric_control.json"
    save_json(str(json_path), {
        "n_fields": test_fields.shape[0],
        "config": {
            "wavelet": args.wavelet,
            "level": args.level,
            "mode": args.mode,
            "tau": args.tau,
            "n_fourier_bands": args.n_fourier_bands,
        },
        "results": all_results,
    })
    print(f"JSON saved to: {json_path}")

    # Save CSV (handle varying keys by using a union of all fieldnames)
    csv_path = output_dir / "symmetric_control.csv"
    if all_results:
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

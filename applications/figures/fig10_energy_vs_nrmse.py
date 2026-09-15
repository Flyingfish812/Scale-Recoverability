#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig10_energy_vs_nrmse.py — Fig. 10 modal energy vs NRMSE (v3-3, 2026-08-31)

Data sources: the same as the canonical energy-vs-NRMSE figure (closed-form
Ridge recomputes W, MLP/VCNN from the NPZ files, POD basis from artifacts/pod_bases).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"

H, W, C = 80, 160, 2
N = H * W * C
N_MODES = 128
TAU = 0.05
LAMBDA_GRID = np.logspace(-8, 2, 21)


def load_mask(mask_num: int) -> np.ndarray:
    mask_path = _ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def build_obs(fields, mask):
    obs_idx = np.argwhere(mask)
    n_obs = len(obs_idx)
    obs = np.zeros((fields.shape[0], n_obs * C), dtype=np.float64)
    for i in range(fields.shape[0]):
        obs[i] = fields[i, obs_idx[:, 0], obs_idx[:, 1], :].ravel()
    return obs


def compute_per_modal_nrmse(output_nchw, target_nchw, pod_basis_flat, mean_flat):
    B = output_nchw.shape[0]
    tgt = target_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    out = output_nchw.transpose(0, 2, 3, 1).reshape(B, N)
    a_true = (tgt - mean_flat[np.newaxis, :]) @ pod_basis_flat
    a_pred = (out - mean_flat[np.newaxis, :]) @ pod_basis_flat
    eps = 1e-12
    numer = np.sum((a_pred - a_true) ** 2, axis=0)
    denom = np.sum(a_true ** 2, axis=0) + eps
    return np.sqrt(numer / denom)


def load_nrmse(model: str) -> list:
    """Per-mode NRMSE of one estimator at (M, sigma) = (20, 0)."""
    records = json.loads(
        (_ROOT / "artifacts" / "statistics" / "modal_coefficient_error.json").read_text()
    )["records"]
    record = next(r for r in records
                  if r["model"] == model and r["sensor_count"] == 20 and r["noise_sigma"] == 0.0)
    return np.asarray(record["nrmse_per_mode"], dtype=np.float64), record["spearman_ci_95"]


def main() -> int:
    ps.apply()
    from scipy.stats import spearmanr

    t0 = time.time()
    pod = np.load(str(_ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"))
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)
    mode_energy = np.var(full_coeffs, axis=0)
    mode_energy_norm = mode_energy / mode_energy[0]

    nrmse_mlp, ci_mlp = load_nrmse("mlp")
    nrmse_ridge, ci_ridge = load_nrmse("ridge")
    nrmse_vcnn, ci_vcnn = load_nrmse("vcnn")

    # ── full-width 1x3 figure ───────────────────────────────────
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35),
                             gridspec_kw=dict(wspace=0.32))
    models = [("MLP", nrmse_mlp, ci_mlp, ps.MODEL_COLORS["MLP"]),
              ("Ridge", nrmse_ridge, ci_ridge, ps.MODEL_COLORS["Ridge"]),
              ("VCNN", nrmse_vcnn, ci_vcnn, ps.MODEL_COLORS["VCNN"])]

    for idx, (name, nrmse, ci, color) in enumerate(models):
        ax = axes[idx]
        ax.scatter(mode_energy_norm, nrmse, c=color, alpha=0.5, s=8,
                   edgecolors="none", rasterized=True)
        r_s, _ = spearmanr(mode_energy_norm, nrmse)
        ci = (float(ci[0]), float(ci[1]))
        # raise the y limit so the text box sits in the blank area above the points
        ymax = float(np.max(nrmse))
        ax.set_ylim(ymin=float(np.min(nrmse)) * 0.5, ymax=ymax * 6.0)
        ax.text(0.05, 0.97, f"$\\rho$ = {r_s:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.6", pad=2.5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda_j/\\lambda_1$", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("NRMSE $e_j^{\\mathrm{NRMSE}}$", fontsize=8.5)
        else:
            ax.set_ylabel("")
        ax.set_title(name, fontsize=ps.TITLE_FONT)
        ax.tick_params(labelsize=7.5)
        ps.panel_label(ax, chr(ord("a") + idx))
    fig.tight_layout()
    ps.save(fig, OUT_DIR, "fig10_energy_vs_nrmse")
    print(f"  [Fig10] done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_counterexample.py — supplementary figure S5: low-error counterexamples.

  (a) Type A: Ridge against VCNN per-band direct error (M=30, sigma=0), red
      circle on the first failed band
  (b) Type B: MLP direct error against the POD-dominant error (M=50,
      sigma=0.01), red circle on the first failed band

The submitted supplementary figure uses panel (b); panel (a) is an alternative rendering of the example shown in main figure 2a. Conditions and conclusions go in the caption.

Data
  - Type A: artifacts/statistics/band_error_decomposition.json
            (M=30, sigma=0, snapshot 49: closed-form Ridge and VCNN on the same snapshot)
  - Type B: MLP M=50 sigma=0.01 seed=0 sample_idx=90 (band-POD protocol)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from luna.core.constants import BANDS_CF, TAU_DEFAULT, DEFAULT_LEVEL, DEFAULT_MODE
from luna.pod.band_pod import fit_band_pod
from luna.wavelet.metrics import band_errors_all, compute_S_full, compute_S_coh, rel_l2
from luna.wavelet.transform import decompose_field_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
TAU = TAU_DEFAULT


def plot_type_a(rows):
    """Type A: Ridge vs VCNN per-band E_direct (M=30, σ=0, same sample)."""
    def pick(model, mask, sigma, sample, seed):
        for r in rows:
            if (r["model"] == model and r["sensor_count"] == mask
                    and r["noise_sigma"] == sigma and r["snapshot_index"] == sample
                    and r["training_seed"] == seed):
                return r
        return None

    ridge = pick("ridge", 30, 0.0, 49, 0)
    vcnn = pick("vcnn", 30, 0.0, 49, 202)
    if ridge is None or vcnn is None:
        print("[warn] Type A records not found")
        return None

    rim = [ridge["band_errors"][b]["total"] for b in BANDS_CF]
    vim = [vcnn["band_errors"][b]["total"] for b in BANDS_CF]

    fig, ax = ps.figure(6.6, 3.4, 1, 1)
    x = np.arange(len(BANDS_CF))
    w = 0.36
    ax.bar(x - w / 2, rim, w, color=ps.MODEL_COLORS["Ridge"], alpha=0.92,
           label="Ridge")
    ax.bar(x + w / 2, vim, w, color=ps.MODEL_COLORS["VCNN"], alpha=0.92,
           label="VCNN")
    ax.axhline(TAU, color=ps.TAU_COLOR, ls="--", lw=1.1)
    ax.axhspan(TAU, max(max(rim), max(vim)) * 1.4, color=ps.TAU_COLOR, alpha=0.05,
               zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, max(max(rim), max(vim)) * 3)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_CF)
    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("$E_{\\mathrm{direct}}(b)$")

    # first-failed-band red circle (the only emphasis)
    for off, errs, name in [(-w / 2, rim, "Ridge"), (w / 2, vim, "VCNN")]:
        fails = [k for k in range(len(BANDS_CF)) if errs[k] > TAU]
        if fails:
            k = fails[0]
            kwargs = dict(width=0.55, height=errs[k] * 0.9 + 0.015, fill=False,
                          edgecolor=ps.ACCENT, lw=1.6, zorder=5)
            ax.add_patch(__import__("matplotlib").patches.Ellipse(
                (x[k] + off, errs[k]), **kwargs))
            if name == "Ridge":
                # text above the failed band, ha=left keeps it inside the axes
                ax.annotate(f"{name} fails first at {BANDS_CF[k]}",
                            xy=(x[k] + off, errs[k]),
                            xytext=(x[k] + off - 0.55, errs[k] * 2.2),
                            ha="left", fontsize=7.5, color=ps.TAU_COLOR,
                            arrowprops=dict(arrowstyle="->", color=ps.TAU_COLOR,
                                            lw=0.9))
    ax.legend(loc="upper left", fontsize=7.5)
    ps.save(fig, OUT_DIR, "fig03_counterexample_a")
    print(f"  [TypeA] Ridge S_full={ridge['s_full']} GER={ridge['global_error']:.5f}; "
          f"VCNN S_full={vcnn['s_full']} GER={vcnn['global_error']:.5f}")


def plot_type_b():
    """Type B: MLP M=50 σ=0.01 seed 0 sample 90 — E_direct vs E_coh."""
    # band-POD (same protocol as the analytical benchmark figure)
    ref = np.load(_ROOT / "artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz")
    test_idx = sorted(set(ref["test_indices"].tolist()))
    all_fields = np.load(_ROOT / "data/cylinder2d_q1.npy")
    train_idx = sorted(set(range(all_fields.shape[0])) - set(test_idx))
    rng = np.random.RandomState(7)
    sub = sorted(rng.choice(train_idx, min(400, len(train_idx)), replace=False))
    band_pod = fit_band_pod(all_fields[sub][:, :, :, 0].astype(np.float64),
                            pod_energy_threshold=0.99, wavelet="db2",
                            level=DEFAULT_LEVEL, mode=DEFAULT_MODE)

    d = np.load(_ROOT / "artifacts/pod_model_sweep_nc/mlp_n0050/seed000/tests/s0100/test_raw.npz")
    t = d["target_nchw"][90, 0, :, :].astype(np.float64)
    r = d["output_nchw"][90, 0, :, :].astype(np.float64)

    ger = rel_l2(r, t)
    ed = band_errors_all(t, r, "db2", DEFAULT_LEVEL, DEFAULT_MODE)
    sfull = compute_S_full(t, r, TAU)
    scoh = compute_S_coh(t, r, band_pod, TAU)

    ec = {}
    for b in BANDS_CF:
        tb = decompose_field_2d(t)[b].ravel()
        rb = decompose_field_2d(r)[b].ravel()
        mu = np.asarray(band_pod[b]["mean"]).ravel()
        U = np.asarray(band_pod[b]["basis"])
        yc, tc = rb - mu, tb - mu
        pt = np.linalg.norm(tc @ U.T)
        ec[b] = float(np.linalg.norm((yc - tc) @ U.T) / (pt + 1e-12)) if pt > 0 else 1.0

    fig, ax = ps.figure(6.6, 3.4, 1, 1)
    x = np.arange(len(BANDS_CF))
    w = 0.36
    ax.bar(x - w / 2, [ed[b] for b in BANDS_CF], w, color=ps.MODEL_COLORS["MLP"],
           alpha=0.92, label="$E_{\\mathrm{direct}}(b)$")
    ax.bar(x + w / 2, [ec[b] for b in BANDS_CF], w, color=ps.MODEL_COLORS["Gappy"],
           alpha=0.92, label="$E_{\\mathrm{coh}}(b)$")
    ax.axhline(TAU, color=ps.TAU_COLOR, ls="--", lw=1.1)
    ax.axhspan(TAU, 0.4, color=ps.TAU_COLOR, alpha=0.05, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_CF)
    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("Relative band error")

    # first-failed-band red circle (E_direct only)
    fails = [k for k in range(len(BANDS_CF)) if ed[BANDS_CF[k]] > TAU]
    for k in fails:
        ax.add_patch(__import__("matplotlib").patches.Ellipse(
            (x[k] - w / 2, ed[BANDS_CF[k]]), width=0.55,
            height=ed[BANDS_CF[k]] * 0.9 + 0.015, fill=False,
            edgecolor=ps.ACCENT, lw=1.6, zorder=5))
    # as in panel (a): label above the failure circles, one short arrow each
    if fails:
        fx = [x[k] - w / 2 for k in fails]
        tx = float(np.mean(fx))
        y_text = max(ed[BANDS_CF[k]] for k in fails) * 2.2
        ax.text(tx, y_text, f"fails first at {BANDS_CF[fails[0]]}",
                ha="center", va="bottom", fontsize=7.5, color=ps.TAU_COLOR)
        for k in fails:
            ax.annotate("", xy=(x[k] - w / 2, ed[BANDS_CF[k]]),
                        xytext=(tx, y_text - 0.012), ha="center",
                        arrowprops=dict(arrowstyle="->", color=ps.TAU_COLOR,
                                        lw=0.9, shrinkA=2, shrinkB=2))
    ax.legend(loc="upper left", fontsize=7.5)
    ps.save(fig, OUT_DIR, "fig03_counterexample_b")
    print(f"  [TypeB] GER={ger:.4f} S_full={sfull} S_coh={scoh} "
          f"E_direct={[round(ed[b],4) for b in BANDS_CF]}")


def main() -> int:
    ps.apply()
    print("== Fig. 3 counterexamples ==")
    records = json.loads(
        (_ROOT / "artifacts" / "statistics" / "band_error_decomposition.json").read_text(encoding="utf-8")
    )["records"]
    plot_type_a(records)
    plot_type_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())

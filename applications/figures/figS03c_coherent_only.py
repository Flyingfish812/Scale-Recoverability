#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS03c_coherent_only.py — S-Fig 3(c) coherent-only POD component example
(v3-3, 2026-08-31)

Single VCNN sample (figS03_coherent_only_sample.pdf, M=20, σ=0, seed0):
  sample = the first test snapshot with S_full <= 3 in
  vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz
  (sample_idx=1, S_full=3, first failed band W2).
  S_coh=5 was confirmed by recomputing with NC train-split band-POD bases
  (channel 0 and the two-component bundle); a coherent-only example.
  Row1: Target / VCNN output / Direct residual (S_full)
  Row2: failed band target / output / band error (evidence that S_coh > S_full)
Output: figS03_coherent_only_sample.pdf under artifacts/figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
BANDS = ps.BANDS
TAU = ps.TAU


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    npz_path = (_ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000"
                "/vcnn_n0020_seed000_custom/tests/s0000/test_raw.npz")
    if not npz_path.exists():
        print(f"  [S3c] missing {npz_path}")
        return 1

    import torch
    from luna.wavelet.transform import decompose_field_2d

    def comps(field):
        return {b: decompose_field_2d(field)[b] for b in BANDS}

    with np.load(npz_path) as z:
        out_nchw = np.asarray(z["output_nchw"], dtype=np.float64)
        tgt_nchw = np.asarray(z["target_nchw"], dtype=np.float64)
    ckpt_path = npz_path.parent.parent.parent / "vcnn_best.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mean_c = np.asarray(ckpt.get("norm_mean_c", [0.0, 0.0]), dtype=np.float64)
    std_c = np.asarray(ckpt.get("norm_std_c", [1.0, 1.0]), dtype=np.float64)
    tgt_phys = tgt_nchw * std_c[None, :, None, None] + mean_c[None, :, None, None]
    out_phys = out_nchw * std_c[None, :, None, None] + mean_c[None, :, None, None]

    def rel_l2(a, b):
        return np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(b.ravel()) + 1e-12)

    def s_from_errs(errs):
        s = 0
        for v in errs:
            if v <= TAU:
                s += 1
            else:
                break
        return s

    sample_idx = None
    for si in range(min(out_phys.shape[0], 100)):
        tc = comps(tgt_phys[si, 0])
        oc = comps(out_phys[si, 0])
        errs = [rel_l2(oc[b], tc[b]) for b in BANDS]
        if s_from_errs(errs) <= 3:
            sample_idx = si
            break
    if sample_idx is None:
        sample_idx = 0

    tgt = tgt_phys[sample_idx, 0]
    out = out_phys[sample_idx, 0]
    tc = comps(tgt)
    oc = comps(out)
    errs = [rel_l2(oc[b], tc[b]) for b in BANDS]
    s_val = s_from_errs(errs)
    fail_band = BANDS[s_val] if s_val < len(BANDS) else "W1"

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0),
                             gridspec_kw=dict(wspace=0.16, hspace=0.22))

    def show(ax, arr, title, lo=None, hi=None, cb=False):
        if lo is None:
            lo, hi = -float(np.abs(arr).max()), float(np.abs(arr).max())
        im = ax.imshow(arr, cmap="RdBu_r", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=ps.TITLE_FONT)
        ax.set_xticks([]); ax.set_yticks([])
        if cb:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        return im

    show(axes[0, 0], tgt, "Target (sample {})".format(sample_idx))
    show(axes[0, 1], out, "VCNN output")
    show(axes[0, 2], out - tgt,
         "Direct residual ($S_{{\\mathrm{{full}}}}$={})".format(s_val), -2, 2)
    show(axes[1, 0], tc[fail_band], "Target: {} band".format(fail_band))
    show(axes[1, 1], oc[fail_band], "Output: {} band".format(fail_band))
    show(axes[1, 2], oc[fail_band] - tc[fail_band],
         "Band error ({})\nRelL2={:.3f}".format(
             fail_band, errs[BANDS.index(fail_band)]), cb=True)

    fig.tight_layout()
    ps.save(fig, OUT_DIR, "figS03_coherent_only_sample")
    print(f"  [S3c] sample={sample_idx}, S_full={s_val}, fail_band={fail_band}, "
          f"errs={[round(e,3) for e in errs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sensitivity of the band errors to the choice of wavelet basis.

Every scale-recoverability number is computed in a db2 wavelet basis. This module
repeats the decomposition with the other bases a reader might reach for — sym2,
which is a phase-shifted db2, and haar, the piecewise-constant limit — and reports
the mean direct band error of the same reconstruction. Sym2 reproduces db2 by
construction; haar trades accuracy in the detail bands for a coarser basis, which
is the effect the table quantifies.

Output
------
artifacts/statistics/wavelet_family_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.metrics.sample_metrics import BANDS  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402
from luna.wavelet.metrics import band_errors_all  # noqa: E402

WAVELETS = ["db2", "sym2", "haar"]
MODEL = "mlp"
SENSORS = 20
SIGMA = 0.0
SEED = 0


def main() -> int:
    print("== wavelet family sensitivity")
    path = run_path(MODEL, SENSORS, SIGMA, SEED)
    if path is None:
        print(f"   [SKIP] no run for {MODEL} M={SENSORS} sigma={SIGMA} seed={SEED}")
        return 1

    target, recon = load_run(path)
    n_samples = target.shape[0]
    print(f"   {MODEL} M={SENSORS} sigma={SIGMA} seed={SEED}, {n_samples} snapshots")

    rows: dict[str, dict[str, float]] = {}
    for wavelet in WAVELETS:
        accumulated = {band: [] for band in BANDS}
        for i in range(n_samples):
            errors = band_errors_all(target[i, 0], recon[i, 0], wavelet=wavelet)
            for band in BANDS:
                accumulated[band].append(errors[band])
        rows[wavelet] = {band: round(float(np.mean(accumulated[band])), 4)
                         for band in BANDS}

    global_error = float(np.mean(
        [np.linalg.norm(recon[i] - target[i]) / np.linalg.norm(target[i])
         for i in range(n_samples)]))
    for wavelet in WAVELETS:
        rows[wavelet]["GER"] = round(global_error, 4)

    result = {
        "description": (f"Mean direct band error of the {MODEL.upper()} estimator "
                        f"(M={SENSORS}, sigma={SIGMA}, seed={SEED}, {n_samples} test "
                        "snapshots) in three wavelet bases"),
        "wavelets": WAVELETS,
        "n_samples": n_samples,
        "rows": {band: [rows[w][band] for w in WAVELETS]
                 for band in ["GER"] + BANDS},
        "per_wavelet": rows,
    }

    target_path = ROOT / "artifacts" / "statistics" / "wavelet_family_sensitivity.json"
    target_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for band in ["GER"] + BANDS:
        print(f"   {band:4s} " + "  ".join(f"{w} {rows[w][band]:.4f}"
                                           for w in WAVELETS))
    print(f"[OK] {target_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

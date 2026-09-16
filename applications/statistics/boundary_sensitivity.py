"""Sensitivity of the scale count to the boundary handling of the transform.

The scale count is quoted for a periodized DWT. This module repeats the count with
the symmetric extension on two representative configurations, so that the reported
numbers can be read as independent of that convention.

Output
------
artifacts/statistics/boundary_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402
from luna.wavelet.metrics import compute_S_full  # noqa: E402

#: (label, estimator, sensor count, noise level, training seed)
CONFIGURATIONS = [
    ("mlp_m20_s0", "mlp", 20, 0.0, 0),
    ("mlp_m30_s001", "mlp", 30, 0.001, 0),
    ("mlp_m30_s01", "mlp", 30, 0.01, 0),
]
MODES = ["periodization", "symmetric"]


def main() -> int:
    print("== boundary sensitivity of the scale count")
    config = get_config()
    results = {}
    for label, model, sensors, sigma, seed in CONFIGURATIONS:
        path = run_path(model, sensors, sigma, seed)
        if path is None:
            print(f"   [SKIP] {label}: no run")
            continue
        target, recon = load_run(path)
        entry = {}
        for mode in MODES:
            counts = [compute_S_full(target[i, 0], recon[i, 0], config.tau, mode=mode)
                      for i in range(target.shape[0])]
            entry[f"{mode}_sfull"] = round(float(np.mean(counts)), 3)
        entry["delta"] = round(entry["periodization_sfull"] - entry["symmetric_sfull"], 3)
        entry["n_samples"] = int(target.shape[0])
        results[label] = entry
        print(f"   {label}: periodization {entry['periodization_sfull']}, "
              f"symmetric {entry['symmetric_sfull']}, delta {entry['delta']}")

    payload = {
        "description": ("Mean scale count at tau under two boundary conventions of "
                        "the discrete wavelet transform"),
        "tau": config.tau,
        "configs": results,
    }
    path = ROOT / "artifacts" / "statistics" / "boundary_sensitivity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

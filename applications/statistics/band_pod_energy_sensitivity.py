"""How much does the band-POD energy threshold move the coherent-only count?

The coherent-only scale count projects the fields onto per-band POD bases whose size follows an energy threshold eta. This module repeats the count at eta = 0.99 (the value used everywhere else) and at the ends of the range the paper reports, so that the threshold can be seen not to carry the result.

Output
------
artifacts/statistics/band_pod_energy_sensitivity.json
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
from luna.core.constants import DEFAULT_LEVEL, DEFAULT_MODE  # noqa: E402
from luna.pod.band_pod import fit_band_pod  # noqa: E402
from luna.wavelet.metrics import compute_S_coh, compute_S_full  # noqa: E402

RAW_SEQUENCE = ROOT / "data" / "cylinder2d_q1.npy"
# : Fields used to fit the per-band bases, drawn from the training split.
BAND_POD_FIELDS = 400
ETA_VALUES = [0.95, 0.99, 0.999]
REFERENCE_ETA = 0.99
MODEL, SENSORS, SIGMA, SEED = "mlp", 30, 0.0, 0


def training_fields() -> np.ndarray:
    """Training snapshots of the reference run, streamwise component."""
    data = np.load(run_path(MODEL, SENSORS, SIGMA, SEED), allow_pickle=True)
    test_indices = set(int(i) for i in data["test_indices"])
    sequence = np.load(RAW_SEQUENCE, mmap_mode="r")
    train_indices = sorted(set(range(sequence.shape[0])) - test_indices)
    rng = np.random.RandomState(7)
    subset = sorted(rng.choice(train_indices,
                              min(BAND_POD_FIELDS, len(train_indices)),
                              replace=False))
    return np.asarray(sequence[subset])[:, :, :, 0].astype(np.float64)


def main() -> int:
    print("== sensitivity to the band-POD energy threshold")
    config = get_config()
    target, recon = load_run(run_path(MODEL, SENSORS, SIGMA, SEED))
    fields = training_fields()

    entries = {}
    for eta in ETA_VALUES:
        band_pod = fit_band_pod(fields, pod_energy_threshold=eta)
        s_coh = [compute_S_coh(target[i, 0], recon[i, 0], band_pod, config.tau)
                 for i in range(target.shape[0])]
        s_full = [compute_S_full(target[i, 0], recon[i, 0], config.tau)
                  for i in range(target.shape[0])]
        entries[f"eta_{eta}"] = {
            "eta": eta,
            "mean_S_coh": round(float(np.mean(s_coh)), 3),
            "mean_S_full": round(float(np.mean(s_full)), 3),
        }
        print(f"   eta={eta}: mean S_coh {entries[f'eta_{eta}']['mean_S_coh']}, "
              f"mean S_full {entries[f'eta_{eta}']['mean_S_full']}")

    reference = entries[f"eta_{REFERENCE_ETA}"]["mean_S_coh"]
    changes = {key: round(abs(value["mean_S_coh"] - reference), 3)
               for key, value in entries.items()}
    payload = {
        "description": ("Mean coherent-only and full-band scale counts of one "
                        "configuration as a function of the band-POD energy "
                        "threshold, over the range the paper reports"),
        "config": {"model": MODEL, "sensor_count": SENSORS, "noise_sigma": SIGMA,
                   "training_seed": SEED, "n_samples": int(target.shape[0]),
                   "tau": config.tau, "reference_eta": REFERENCE_ETA},
        "configs": {f"{MODEL}_m{SENSORS}_s{SIGMA:g}": entries},
        "max_scoh_change": max(changes.values()),
        "changes_vs_reference": changes,
    }

    path = ROOT / "artifacts" / "statistics" / "band_pod_energy_sensitivity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"   largest change of the coherent-only count: {payload['max_scoh_change']}")
    print(f"[OK] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

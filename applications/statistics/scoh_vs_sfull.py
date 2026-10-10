"""Does the coherent-only reading of the scale count agree with the total-error one?

``S_full`` counts the bands whose total error, truncation plus estimator error, is below τ. ``S_coh`` repeats the count after a per-band POD has absorbed the part of the error that lies inside the band, so it answers a different question: is the band present, rather than is it present in every coefficient. This module compares the two counts configuration by configuration and reports how often they differ.

Output
------
artifacts/statistics/scoh_vs_sfull.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402
from features.training.pod_sweep import split_indices  # noqa: E402
from luna.core.constants import DEFAULT_LEVEL, DEFAULT_MODE  # noqa: E402
from luna.pod.band_pod import fit_band_pod  # noqa: E402
from luna.wavelet.metrics import compute_S_coh, compute_S_full  # noqa: E402

RAW_SEQUENCE = ROOT / "data" / "cylinder2d_q1.npy"
BAND_POD_FIELDS = 400
BAND_POD_SEED = 7
# : Training seed whose random split defines the fitting pool of the band-POD
# : basis. The same seed reproduces the run's test set exactly (asserted below),
# : so the basis is fitted on the training split alone.
BAND_POD_TRAINING_SEED = 0
N_RESAMPLES = 10_000
CI_LEVEL = 0.95


_BAND_POD_CACHE: dict[int, dict] = {}


def band_pod_models_for_seed(training_seed: int) -> dict:
    """Per-band POD models fitted on the training split of one training seed.

    The split is the ``torch`` random split of the run with this training seed
    (``features.training.pod_sweep.split_indices``), so the fitting pool
    excludes the validation and test snapshots. A fixed random draw of
    ``BAND_POD_FIELDS`` training snapshots keeps the fit deterministic.
    """
    if training_seed in _BAND_POD_CACHE:
        return _BAND_POD_CACHE[training_seed]

    sequence = np.load(RAW_SEQUENCE, mmap_mode="r")
    split = split_indices(sequence.shape[0], training_seed)
    train_indices = np.sort(np.asarray(split["train"], dtype=np.int64))

    rng = np.random.RandomState(BAND_POD_SEED)
    subset = sorted(rng.choice(train_indices, min(BAND_POD_FIELDS, len(train_indices)),
                               replace=False))
    fields = np.asarray(sequence[subset])[:, :, :, 0].astype(np.float64)
    _BAND_POD_CACHE[training_seed] = fit_band_pod(
        fields, pod_energy_threshold=0.99, wavelet="db2",
        level=DEFAULT_LEVEL, mode=DEFAULT_MODE)
    return _BAND_POD_CACHE[training_seed]


def band_pod_models(reference_run: Path) -> dict:
    """Per-band POD models fitted on the training split of the reference run."""
    data = np.load(reference_run, allow_pickle=True)
    test_indices = set(int(i) for i in data["test_indices"])
    sequence = np.load(RAW_SEQUENCE, mmap_mode="r")
    split = split_indices(sequence.shape[0], BAND_POD_TRAINING_SEED)
    if set(int(i) for i in split["test"]) != test_indices:
        raise RuntimeError(
            "the reference run does not use the expected seed-0 random split; "
            "check the training seed before fitting the band-POD basis")
    return band_pod_models_for_seed(BAND_POD_TRAINING_SEED)


def compare_configuration(job: tuple[str, int, float]) -> dict | None:
    """Mean scale count of one configuration under both readings."""
    config = get_config()
    model, sensors, sigma = job
    path = run_path(model, sensors, sigma, 0)
    if path is None:
        return None

    target, recon = load_run(path)
    band_pod = band_pod_models(run_path("mlp", 20, 0.0, 0))

    s_full, s_coh = [], []
    for i in range(target.shape[0]):
        s_full.append(compute_S_full(target[i, 0], recon[i, 0], config.tau))
        s_coh.append(compute_S_coh(target[i, 0], recon[i, 0], band_pod, config.tau))

    return {
        "model": model,
        "sensor_count": sensors,
        "noise_sigma": sigma,
        "n_samples": len(s_full),
        "mean_S_full": round(float(np.mean(s_full)), 3),
        "mean_S_coh": round(float(np.mean(s_coh)), 3),
        "mean_diff": round(float(np.mean(s_coh)) - float(np.mean(s_full)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = parser.parse_args()

    print("== coherent-only versus total-error scale count")
    config = get_config()
    jobs = [(model, sensors, sigma)
            for model in ("mlp", "vcnn", "ridge")
            for sensors in config.M_values
            for sigma in config.sigma_values]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            configurations = [c for c in pool.map(compare_configuration, jobs, chunksize=1)
                              if c is not None]
    else:
        configurations = [c for c in map(compare_configuration, jobs) if c is not None]

    diffs = np.asarray([c["mean_diff"] for c in configurations], dtype=float)
    higher = int(np.sum(diffs > 1e-9))
    equal = int(np.sum(np.abs(diffs) <= 1e-9))
    lower = int(np.sum(diffs < -1e-9))

    rng = np.random.default_rng(config.block_bootstrap.get("seed", 0))
    draws = rng.integers(0, diffs.size, size=(N_RESAMPLES, diffs.size))
    means = diffs[draws].mean(axis=1)
    ci = [round(float(np.quantile(means, (1 - CI_LEVEL) / 2)), 2),
          round(float(np.quantile(means, 1 - (1 - CI_LEVEL) / 2)), 2)]

    n = len(configurations)
    result = {
        "description": ("Mean scale count of each configuration under the total-error "
                        "reading (S_full) and the coherent-only reading (S_coh); "
                        f"{CI_LEVEL:.0%} bootstrap interval of the mean difference over "
                        "configurations"),
        "n_configurations": n,
        "ci_level": CI_LEVEL,
        "n_configs_scoh_gt": higher,
        "n_configs_equal": equal,
        "n_configs_scoh_lt": lower,
        "pct_scoh_gt": round(100.0 * higher / n, 1),
        "pct_equal": round(100.0 * equal / n, 1),
        "pct_scoh_lt": round(100.0 * lower / n, 1),
        "mean_diff": round(float(diffs.mean()), 2),
        "ci": ci,
        "per_configuration": configurations,
    }

    path = ROOT / "artifacts" / "statistics" / "scoh_vs_sfull.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   {n} configurations: S_coh > S_full {higher}, equal {equal}, lower {lower}")
    print(f"   mean difference {result['mean_diff']} (CI {ci})")
    print(f"[OK] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

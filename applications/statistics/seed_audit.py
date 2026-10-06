"""How much do the reported scale counts and global errors move with the seed sample?

Every configuration of the sweep is trained with three seeds (0, 101, 202). This module asks what a wider seed sample would change: two further seeds (303, 404) were trained for the sensor counts of three representative conditions, and the per-run scale count and global error of the three-seed and the five-seed sample are compared on the same test snapshots.

    condition     sensor count M   noise level sigma
    clean         20               0
    transition    30               0.01
    high noise    20               0.1

Each run contributes the mean global error over its test snapshots and the mean, standard deviation and mode of the scale count, together with the coherent-subspace index and the direct band errors of the streamwise component. The global error is the relative L2 error of the full two-component state and the scale indices are evaluated on the streamwise component, both in physical units, the convention of the rest of the layer.

The convolutional estimator has runs for the three main seeds only, so its five-seed block repeats its three-seed block; the artifact keeps the same comparison keys for both estimators so that the two can be read side by side.

Inputs
    data/cylinder2d_q1.npy        raw snapshots, for the band-POD reference basis
    artifacts/<estimator runs>/   one test_raw.npz per (estimator, M, sigma, seed)
Output
    artifacts/statistics/seed_audit.json

Usage
    python -m applications.statistics.seed_audit
    python -m applications.statistics.seed_audit --verify

``--verify`` compares the artifact with the reference values of the paper. The reference evaluated the convolutional estimator on its stored normalised fields and every global error on the streamwise component alone, so only the MLP scale indices are expected to reproduce exactly; the remaining fields are compared for information and the comparison still exits non-zero only when a reproduced field disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.training.estimator_runs import (  # noqa: E402
    load_run,
    run_path,
    snapshot_indices,
)
from luna.core.constants import DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET  # noqa: E402
from luna.pod.band_pod import fit_band_pod  # noqa: E402
from luna.wavelet.metrics import (  # noqa: E402
    band_errors_all,
    compute_S_coh,
    compute_S_full,
    global_error,
)

DATA_ARRAY = ROOT / "data" / "cylinder2d_q1.npy"
OUTPUT = ROOT / "artifacts" / "statistics" / "seed_audit.json"
# : Frozen artifact of the earlier analysis of the same quantity, read by --verify.
REFERENCE = (ROOT / "artifacts" / "derived" / "main" / "statistics"
             / "seed_audit_3v5.json")

# : Conditions of the audit: label, sensor count, noise level.
CONDITIONS = [
    ("clean", 20, 0.0),
    ("transition", 30, 0.01),
    ("noisy", 20, 0.1),
]
# : Estimators of the audit, in the order of the artifact.
MODELS = ("mlp", "vcnn")
# : Seeds of the main sweep and of the extended sample.
SEEDS_MAIN = [0, 101, 202]
SEEDS_EXTENDED = [0, 101, 202, 303, 404]
# : Snapshots of the raw sequence used to fit the band-POD reference basis, and : the seed of the draw. The draw is the one of the main sensitivity analysis.
N_TRAIN_POD = 400
BAND_POD_SEED = 7
# : Sensor count and noise level whose test split defines the training snapshots.
SPLIT_REFERENCE = ("mlp", 20, 0.0)
# : Fields the paper reads from every block of the artifact.
REPRODUCED_FIELDS = (
    "S_full_mean_3sd",
    "S_full_sd_across_seeds_3",
    "S_full_mean_5sd",
    "S_full_sd_across_seeds_5",
)
# : Tolerance of --verify on a reproduced field.
VERIFY_TOLERANCE = 1e-9


def band_pod_reference() -> dict:
    """Band-POD basis of the coherent subspace, on the sweep's training split.

    The coherent index needs a POD subspace per band. It is fitted once on the raw snapshots, on the complement of the test split of the reference configuration, and shared by every run of the audit.
    """
    path = run_path(*SPLIT_REFERENCE, seed=0)
    if path is None:
        raise SystemExit(f"missing reference run: {SPLIT_REFERENCE}")
    test_indices = set(snapshot_indices(path).tolist())
    fields = np.load(DATA_ARRAY, mmap_mode="r")
    train_indices = sorted(set(range(fields.shape[0])) - test_indices)
    rng = np.random.RandomState(BAND_POD_SEED)
    subset = sorted(
        rng.choice(train_indices, min(N_TRAIN_POD, len(train_indices)), replace=False)
    )
    return fit_band_pod(
        np.asarray(fields[subset])[:, :, :, 0].astype(np.float64),
        pod_energy_threshold=get_config().eta,
        wavelet=DEFAULT_WAVELET,
        level=DEFAULT_LEVEL,
        mode=DEFAULT_MODE,
    )


def per_run_metrics(path: Path, band_pod: dict, cfg) -> dict:
    """Scale counts and errors of one training run over its test snapshots."""
    target, reconstruction = load_run(path)
    truth = target[:, 0]
    estimate = reconstruction[:, 0]

    ger = []
    s_full = []
    s_coh = []
    direct = {band: [] for band in cfg.bands}
    for i in range(target.shape[0]):
        ger.append(global_error(target[i], reconstruction[i]))
        errors = band_errors_all(
            truth[i], estimate[i], cfg.wavelet_family, cfg.wavelet_level, cfg.wavelet_mode
        )
        for band in cfg.bands:
            direct[band].append(errors[band])
        s_full.append(int(compute_S_full(
            truth[i], estimate[i], cfg.tau,
            cfg.wavelet_family, cfg.wavelet_level, cfg.wavelet_mode,
        )))
        s_coh.append(int(compute_S_coh(
            truth[i], estimate[i], band_pod, cfg.tau,
            cfg.wavelet_family, cfg.wavelet_level, cfg.wavelet_mode,
        )))

    s_full_array = np.asarray(s_full, dtype=int)
    s_coh_array = np.asarray(s_coh, dtype=int)
    return {
        "GER_mean": float(np.mean(ger)),
        "S_full_mean": float(np.mean(s_full_array)),
        "S_full_std": float(np.std(s_full_array)),
        "S_full_mode": int(np.bincount(s_full_array).argmax()),
        "S_coh_mean": float(np.mean(s_coh_array)),
        "S_coh_mode": int(np.bincount(s_coh_array).argmax()),
        "E_direct_mean": {band: float(np.mean(direct[band])) for band in cfg.bands},
    }


def seed_aggregate(per_seed: dict[int, dict], seeds: list[int], field: str) -> tuple[float, float]:
    """Mean and cross-run standard deviation of one field over the given seeds."""
    values = [per_seed[seed][field] for seed in seeds if seed in per_seed]
    mean = float(np.mean(values))
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, spread


def audit_block(per_seed: dict[int, dict]) -> dict:
    """Three-seed against five-seed comparison of one estimator and condition."""
    s_full_3, sd_3 = seed_aggregate(per_seed, SEEDS_MAIN, "S_full_mean")
    s_full_5, sd_5 = seed_aggregate(per_seed, SEEDS_EXTENDED, "S_full_mean")
    ger_3, ger_sd_3 = seed_aggregate(per_seed, SEEDS_MAIN, "GER_mean")
    ger_5, ger_sd_5 = seed_aggregate(per_seed, SEEDS_EXTENDED, "GER_mean")
    return {
        "n_seeds_old": sum(1 for seed in SEEDS_MAIN if seed in per_seed),
        "n_seeds_all": len(per_seed),
        "S_full_mean_3sd": s_full_3,
        "S_full_sd_across_seeds_3": sd_3,
        "S_full_mean_5sd": s_full_5,
        "S_full_sd_across_seeds_5": sd_5,
        "GER_mean_3sd": ger_3,
        "GER_sd_across_seeds_3": ger_sd_3,
        "GER_mean_5sd": ger_5,
        "GER_sd_across_seeds_5": ger_sd_5,
        "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
    }


def build_audit() -> dict:
    """Per-condition, per-estimator three-seed against five-seed comparison."""
    cfg = get_config()
    band_pod = band_pod_reference()
    audit: dict[str, dict] = {}
    for label, sensor_count, sigma in CONDITIONS:
        audit[label] = {}
        for model in MODELS:
            per_seed: dict[int, dict] = {}
            for seed in SEEDS_EXTENDED:
                path = run_path(model, sensor_count, sigma, seed)
                if path is None:
                    print(f"   [skip] {model} M={sensor_count} sigma={sigma} "
                          f"seed={seed}: no run on disk")
                    continue
                per_seed[seed] = per_run_metrics(path, band_pod, cfg)
            if not per_seed:
                continue
            audit[label][model] = audit_block(per_seed)
            block = audit[label][model]
            print(f"   {label:10s} {model:5s} | S_full 3-seed "
                  f"{block['S_full_mean_3sd']:.3f} +- {block['S_full_sd_across_seeds_3']:.3f} "
                  f"-> 5-seed {block['S_full_mean_5sd']:.3f} "
                  f"+- {block['S_full_sd_across_seeds_5']:.3f} | GER 3-seed "
                  f"{block['GER_mean_3sd']:.5f} -> 5-seed {block['GER_mean_5sd']:.5f}")

    # The two estimators of a condition are ranked by their three-seed mean scale count, which is the ordering the paper reports.
    for label in audit:
        means = {model: audit[label][model]["S_full_mean_3sd"]
                 for model in audit[label]}
        if len(means) == 2:
            ranking = "MLP>VCNN" if means["mlp"] >= means["vcnn"] else "VCNN>MLP"
        else:
            ranking = "single-estimator"
        for model in audit[label]:
            audit[label][model]["ranking_3sd"] = ranking
    return audit


def verify(audit: dict) -> int:
    """Compare the artifact with the reference values of the paper."""
    if not REFERENCE.exists():
        print(f"   [skip] no reference values at {REFERENCE.relative_to(ROOT)}")
        return 0
    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
    failures = 0
    for label, _, _ in CONDITIONS:
        for model in MODELS:
            if label not in audit or model not in audit[label]:
                continue
            block = audit[label][model]
            for field in REPRODUCED_FIELDS:
                if model == "vcnn":
                    print(f"   [info] {label} {model} {field}: "
                          f"{block[field]:.6f} (frozen {frozen[label][model][field]:.6f}, "
                          f"convention differs)")
                    continue
                difference = abs(block[field] - frozen[label][model][field])
                status = "ok" if difference <= VERIFY_TOLERANCE else "FAIL"
                if status == "FAIL":
                    failures += 1
                print(f"   [{status}] {label} {model} {field}: "
                      f"{block[field]:.9f} vs frozen "
                      f"{frozen[label][model][field]:.9f} (diff {difference:.2e})")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="compare the artifact with the reference values")
    args = parser.parse_args()

    start = time.time()
    print("== seed audit: three main seeds against five seeds")
    audit = build_audit()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2, default=float), encoding="utf-8")
    print(f"[OK] {OUTPUT.relative_to(ROOT)} ({time.time() - start:.1f}s)")

    if args.verify:
        return verify(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

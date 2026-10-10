"""POD-dominant records and their band capture ratios in physical units.

For every reconstruction record of the main study this module compares the
per-band direct error with the POD-dominant band error of ``luna``:

    E_direct(b) = ||W_b u - W_b uhat||_2 / ||W_b u||_2,
    E_coh(b)    = ||P_b W_b (u - uhat)||_2 / ||Pi_b(W_b u)||_2,
    Pi_b(w)     = mu_b + P_b (w - mu_b),

with ``P_b`` the band-POD projector of the record's training seed (fitted to
that split's training snapshots alone, see ``scoh_vs_sfull.band_pod_models_for_seed``)
and ``mu_b`` the band mean it was centred on.

Each record is classified as

    full                every E_direct(b) <= tau,
    POD-dominant only   some E_direct > tau but every E_coh <= tau,
    failed              otherwise.

For the POD-dominant records the module reports the per-band capture ratio

    gamma_b(u) = ||Pi_b(W_b u)||_2^2 / ||W_b u||_2^2,

the fraction of the band energy that the projected target retains, i.e. the
energy fraction of the denominator reference of ``E_coh``. It is quoted by the
paper next to the fitting-time energy target eta, so that the reader can check
the coherent comparison is not made against a small reference.

The estimate ``c1_b = ||P_b (W_b u - mu_b)||_2 / ||W_b u||_2`` (centred
projection, unsquared) is stored alongside for continuity with the legacy
coherent-only analysis; ``gamma_b`` is the quantity defined in the paper.

Output
------
artifacts/statistics/co_energy_records.json
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
from applications.statistics.scoh_vs_sfull import band_pod_models_for_seed  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402
from luna.core.constants import (  # noqa: E402
    BANDS_CF,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    DEFAULT_WAVELET,
)
from luna.wavelet.metrics import compute_S_coh, compute_S_full  # noqa: E402
from luna.wavelet.transform import decompose_field_2d  # noqa: E402

OUTPUT = ROOT / "artifacts" / "statistics" / "co_energy_records.json"
# : Learned-model seeds of the main sweep; Ridge is deterministic.
SEEDS = (0, 101, 202)
MODELS = ("mlp", "vcnn", "ridge")
# : Energy threshold of the band-POD fit, matching the paper's eta.
BAND_POD_ETA = 0.99


def band_captures(
    field: np.ndarray,
    band_pod: dict,
    wavelet: str,
    level: int,
    mode: str,
) -> dict[str, float]:
    """Per-band capture ratios of one target field (streamwise component)."""
    out: dict[str, float] = {}
    bands = decompose_field_2d(field, wavelet, level, mode)
    for band in BANDS_CF:
        w = bands[band].ravel()
        mu = np.asarray(band_pod[band]["mean"], dtype=np.float64).ravel()
        basis = np.asarray(band_pod[band]["basis"], dtype=np.float64)
        w_norm2 = float(w @ w)
        coeff = basis @ (w - mu)
        projected = basis.T @ coeff + mu
        out[band] = float((projected @ projected) / w_norm2)  # gamma_b
        out[f"c1_{band}"] = float(np.linalg.norm(coeff) / np.sqrt(w_norm2))
    return out


def analyse_configuration(job: tuple) -> dict | None:
    """Classify every record of one configuration and collect its captures."""
    model, sensors, sigma, seed = job
    path = run_path(model, sensors, sigma, seed)
    if path is None:
        return None

    cfg = get_config()
    tau = cfg.tau
    wavelet, level, mode = cfg.wavelet_family, cfg.wavelet_level, cfg.wavelet_mode
    band_pod = band_pod_models_for_seed(seed)

    target, recon = load_run(path)
    n_full = n_pod_only = n_failed = 0
    captures: dict[str, list[float]] = {b: [] for b in BANDS_CF}
    captures_c1: dict[str, list[float]] = {b: [] for b in BANDS_CF}
    pod_only_gamma_record: list[float] = []

    for i in range(target.shape[0]):
        u = target[i, 0]
        uh = recon[i, 0]
        s_full = compute_S_full(u, uh, tau, wavelet, level, mode)
        if s_full == 5:
            n_full += 1
            continue
        s_coh = compute_S_coh(u, uh, band_pod, tau, wavelet, level, mode)
        if s_coh < 5:
            n_failed += 1
            continue
        n_pod_only += 1
        values = band_captures(u, band_pod, wavelet, level, mode)
        for b in BANDS_CF:
            captures[b].append(values[b])
            captures_c1[b].append(values[f"c1_{b}"])
        pod_only_gamma_record.append(
            float(np.median([values[b] for b in BANDS_CF])))

    return {
        "model": model,
        "sensor_count": sensors,
        "noise_sigma": sigma,
        "training_seed": seed,
        "n_records": int(target.shape[0]),
        "n_full": n_full,
        "n_pod_only": n_pod_only,
        "n_failed": n_failed,
        "gamma_by_band": {b: captures[b] for b in BANDS_CF},
        "c1_by_band": {b: captures_c1[b] for b in BANDS_CF},
        "gamma_record_median": pod_only_gamma_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = parser.parse_args()

    print("== POD-dominant records and band capture ratios")
    cfg = get_config()
    jobs = []
    for model in MODELS:
        # Ridge is deterministic: one record per (sensor, noise) condition.
        seeds = (0,) if model == "ridge" else SEEDS
        for sensors in cfg.M_values:
            for sigma in cfg.sigma_values:
                for seed in seeds:
                    jobs.append((model, sensors, sigma, seed))
    assert cfg.bands == list(BANDS_CF), "band order changed; update BANDS_CF usage"

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            configurations = [c for c in pool.map(analyse_configuration, jobs, chunksize=1)
                              if c is not None]
    else:
        configurations = [c for c in map(analyse_configuration, jobs) if c is not None]
    if not configurations:
        raise SystemExit("no runs found")

    n_records = sum(c["n_records"] for c in configurations)
    n_full = sum(c["n_full"] for c in configurations)
    n_pod_only = sum(c["n_pod_only"] for c in configurations)
    n_failed = sum(c["n_failed"] for c in configurations)
    if not n_records:
        raise SystemExit("no records processed")

    by_band_gamma = {
        b: [v for c in configurations for v in c["gamma_by_band"][b]]
        for b in BANDS_CF
    }
    by_band_c1 = {
        b: [v for c in configurations for v in c["c1_by_band"][b]]
        for b in BANDS_CF
    }
    record_medians = np.asarray(
        [v for c in configurations for v in c["gamma_record_median"]], dtype=float)

    def summary(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            "median": round(float(np.median(arr)), 4),
            "iqr": [round(float(np.percentile(arr, 25)), 4),
                    round(float(np.percentile(arr, 75)), 4)],
        }

    overall_pooled = [v for c in configurations
                      for b in BANDS_CF for v in c["gamma_by_band"][b]]
    by_model: dict[str, dict] = {}
    for c in configurations:
        entry = by_model.setdefault(c["model"], {"n_records": 0, "n_pod_only": 0})
        entry["n_records"] += c["n_records"]
        entry["n_pod_only"] += c["n_pod_only"]
    for entry in by_model.values():
        entry["pct_pod_only"] = round(
            100.0 * entry["n_pod_only"] / entry["n_records"], 1)
    result = {
        "description": (
            "POD-dominant (coherent-only) records of the main study and the "
            "band capture ratio gamma_b = ||Pi_b(W_b u)||^2 / ||W_b u||^2 of "
            "their targets, with the band-POD basis fitted to each training "
            "seed's training split alone"),
        "tau": cfg.tau,
        "band_pod_eta": BAND_POD_ETA,
        "n_configurations": len(configurations),
        "n_records": n_records,
        "n_full": n_full,
        "n_pod_only": n_pod_only,
        "n_failed": n_failed,
        "pct_pod_only_of_records": round(100.0 * n_pod_only / n_records, 1),
        "by_model": by_model,
        "gamma_median_pooled": round(float(np.median(overall_pooled)), 4),
        "gamma_iqr_pooled": [
            round(float(np.percentile(overall_pooled, 25)), 4),
            round(float(np.percentile(overall_pooled, 75)), 4),
        ],
        "gamma_record_median": {
            "median": round(float(np.median(record_medians)), 4),
            "iqr": [round(float(np.percentile(record_medians, 25)), 4),
                    round(float(np.percentile(record_medians, 75)), 4)],
        },
        "gamma_by_band": {b: summary(by_band_gamma[b]) for b in BANDS_CF},
        "gamma_a4_median": round(float(np.median(by_band_gamma["A4"])), 4),
        "gamma_w1_median": round(float(np.median(by_band_gamma["W1"])), 4),
        "c1_by_band": {b: summary(by_band_c1[b]) for b in BANDS_CF},
        "conventions": {
            "units": "physical fields for every estimator (load_run)",
            "band_basis": "per training seed, training split, 400 fields",
            "capture": "gamma_b = ||mu_b + P_b(W_b u - mu_b)||^2 / ||W_b u||^2",
            "c1": "legacy-style centred projection ||P_b(W_b u - mu_b)|| / ||W_b u||",
        },
        "configurations": [
            {k: v for k, v in c.items()
             if k not in ("gamma_by_band", "c1_by_band", "gamma_record_median")}
            for c in configurations
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   records {n_records}: full {n_full}, POD-dominant only {n_pod_only}, "
          f"failed {n_failed}")
    print(f"   gamma median (pooled) {result['gamma_median_pooled']:.4f}, "
          f"record medians {result['gamma_record_median']['median']:.4f} "
          f"(IQR {result['gamma_record_median']['iqr']})")
    print(f"   A4 {result['gamma_a4_median']:.4f}, W1 {result['gamma_w1_median']:.4f}")
    print(f"   written to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

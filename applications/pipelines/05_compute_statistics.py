"""Produce every statistic the paper reports.

Each producer in ``applications/statistics`` writes one artifact under
``artifacts/statistics``; this module runs them in dependency order so that the
whole statistics layer can be rebuilt with one command. Producers are
independent Python programs, so a failure is reported but does not stop the
remaining steps, and the exit status of the whole run is non-zero if any of them
failed.

Prerequisites
    trained runs under artifacts/ (see applications/pipelines/03_train_estimators.py)
    the POD bases under artifacts/pod_bases/

Pipeline
    previous  applications/pipelines/03_train_estimators.py
    this step writes artifacts/statistics/
    next      applications/figures/make_all_figures.py

Outputs
    artifacts/statistics/*.json  (one file per producer, see the docstrings)
    artifacts/statistics/sensor_family/*.csv

Usage
    python applications/pipelines/05_compute_statistics.py
    python applications/pipelines/05_compute_statistics.py --only band_error_decomposition
    python applications/pipelines/05_compute_statistics.py --list
    python applications/pipelines/05_compute_statistics.py --sequential
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Producers in dependency order: a producer only reads artifacts written by the
#: steps above it (or by the training pipeline).
PRODUCERS = [
    "applications.statistics.truncation_reference_audit",
    "applications.statistics.analytical_benchmark",
    "applications.statistics.band_error_decomposition",
    "applications.statistics.band_error_records",
    "applications.statistics.low_ger_analysis",
    "applications.statistics.per_mode_nrmse",
    "applications.statistics.excess_error_recompute",
    "applications.statistics.sensor_noise_phase",
    "applications.statistics.mode_scale_energy",
    "applications.statistics.level_sensitivity",
    "applications.statistics.modal_coefficient_error",
    "applications.statistics.noise_propagation",
    "applications.statistics.band_denominator_check",
    "applications.statistics.temporal_dependence",
    "applications.statistics.paired_model_comparison",
    "applications.statistics.sensor_family_summary",
    "applications.statistics.gappy_pod_baseline",
    "applications.statistics.gappy_band_errors",
    "applications.statistics.fourier_band_baseline",
    "applications.statistics.transform_symmetry_check",
    "applications.statistics.threshold_sensitivity",
    "applications.statistics.wavelet_sensitivity",
    "applications.statistics.equal_ger_pairs",
    "applications.statistics.ger_band_correlation",
    "applications.statistics.within_config_physics_bootstrap",
    "applications.statistics.seed_stability",
    "applications.statistics.seed_audit",
    "applications.statistics.boundary_sensitivity",
    "applications.statistics.band_pod_energy_sensitivity",
    "applications.statistics.tau_pairwise_checks",
    "applications.statistics.wavelet_family_sensitivity",
    "applications.statistics.scoh_vs_sfull",
]

#: Producers that evaluate one configuration per worker accept a worker count;
#: passing it keeps the reproduction within minutes instead of hours.
WORKER_FLAG = {
    "applications.statistics.paired_model_comparison": "--jobs",
    "applications.statistics.scoh_vs_sfull": "--jobs",
}


def run(module: str, workers: int) -> tuple[str, int, float]:
    """Run one producer in its own process and report its status."""
    command = [sys.executable, "-u", "-m", module]
    if module in WORKER_FLAG:
        command += [WORKER_FLAG[module], str(workers)]
    start = time.time()
    print(f"\n{'=' * 70}\n  {module}\n{'=' * 70}", flush=True)
    completed = subprocess.run(
        command, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
    )
    return module, completed.returncode, time.time() - start



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only the producers whose module name contains one of these")
    parser.add_argument("--list", action="store_true", help="list the producers and exit")
    parser.add_argument("--sequential", action="store_true",
                        help="run one producer at a time (default: two at a time)")
    parser.add_argument("--jobs", type=int, default=2, help="concurrent producers")
    parser.add_argument("--workers", type=int, default=0,
                        help="workers inside a producer (0 = a quarter of the cores)")
    args = parser.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 4) // 4)

    if args.list:
        for module in PRODUCERS:
            print(module)
        return 0

    selected = PRODUCERS
    if args.only:
        selected = [m for m in PRODUCERS
                    if any(name in m.rsplit(".", 1)[-1] for name in args.only)]
        if not selected:
            raise SystemExit(f"no producer matches {args.only}")

    start = time.time()
    print(f"== computing statistics: {len(selected)} producers, "
          f"{workers} workers each")
    if args.sequential or args.jobs <= 1:
        results = [run(module, workers) for module in selected]
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda m: run(m, workers), selected))

    failed = [module for module, code, _ in results if code != 0]
    print(f"\n[{'OK' if not failed else 'FAILED'}] {len(results) - len(failed)}"
          f"/{len(results)} producers succeeded ({time.time() - start:.1f}s)")
    for module, _, duration in results:
        print(f"   {duration:7.1f}s  {module}")
    for module in failed:
        print(f"   failed: {module}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

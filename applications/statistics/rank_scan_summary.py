"""Aggregate the retained-rank scan into the numbers the sensitivity table prints.

The main study holds the retained rank of the POD-based estimators at
``r = 128``. ``applications/statistics/rank_sensitivity.py`` re-runs those
estimators with the same split and the same validation-selected Gappy POD rule at
``r = 100`` and ``r = 150``; this module reads the resulting runs and evaluates
each one with the paper's metrics (global relative error over both velocity
components in physical units, and per-band errors and scale count on the
streamwise component, ``tau = 0.05``).

The ``r = 128`` column of the table is the main study itself, and the scan
re-derives it from the same canonical bundle: the summary records how far the
scan's own ``r = 128`` arm is from the main-study run of the same split and
seed (``main_study_check``). The convolutional estimator does not appear: it
reconstructs the field without a POD basis, so it has no retained rank to vary.

Output
------
artifacts/statistics/rank_scan_summary.json
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

from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.training.estimator_runs import load_run, run_path  # noqa: E402

RANKS = (100, 128, 150)
SCANNED_RANKS = {k: RANKS for k in ("ridge", "gappy", "mlp")}
MTEN = (20, 30, 50)
MLP_M = (20, 30)
SIGMAS = (0.0, 0.001, 0.01, 0.1)
#: The reference conditions of the main-study comparison: every scanned mask at
#: clean and medium noise.
CHECK_SIGMAS = (0.0, 0.01)
CHECK_MASKS = {"ridge": MTEN, "gappy": MTEN, "mlp": MLP_M}
TAU = 0.05


def relative(path: Path) -> str:
    """Path relative to the repository when possible, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: str) -> Path:
    """Absolute path of a recorded artifact path (relative paths resolve to the
    repository root)."""
    given = Path(path)
    return given if given.is_absolute() else (ROOT / given)


def find_scan_root(explicit: str | None) -> Path:
    """Scan root: the given directory, the published default, or any scan batch."""
    candidates: list[Path] = []
    if explicit:
        given = Path(explicit)
        candidates.append(given if given.is_absolute() else ROOT / given)
    candidates.append(ROOT / "results" / "rank_scan")
    candidates.extend(sorted((ROOT / "results").glob("*rank_scan*")))
    for path in candidates:
        if (path / "rank_sensitivity.json").exists():
            return path.resolve()
    raise FileNotFoundError("no rank_sensitivity.json found under results/")


def run_metrics(npz: Path) -> dict:
    """Mean GER, scale count and finest-band error of one run."""
    target, recon = load_run(npz)
    ger, sfull, w1 = [], [], []
    for i in range(len(target)):
        m = compute_sample_metrics(recon, target, i, tau=TAU)
        ger.append(m["GER"])
        sfull.append(m["S_full"])
        w1.append(m["E_W1"])
    return {
        "ger": round(float(np.mean(ger)), 8),
        "s_full": round(float(np.mean(sfull)), 3),
        "e_direct_w1": round(float(np.mean(w1)), 8),
        "n_snapshots": int(len(target)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", default=None,
                        help="directory holding rank_sensitivity.json "
                             "(default: results/rank_scan, else any scan batch)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scan_root = find_scan_root(args.scan_root)
    scan = json.loads((scan_root / "rank_sensitivity.json").read_text(encoding="utf-8"))
    by_rank = {(int(r["r"]), r["estimator"], int(r["M"]), float(r["sigma"])): r
               for r in scan["results"] if r.get("npz")}
    by_rank = {k: {**v, "npz": resolve(v["npz"])} for k, v in by_rank.items()}
    cells: list[dict] = []
    for estimator, ranks in SCANNED_RANKS.items():
        masks = MLP_M if estimator == "mlp" else MTEN
        for M in masks:
            for sigma in SIGMAS:
                for rank in ranks:
                    entry = by_rank.get((rank, estimator, M, sigma))
                    if entry is None or not Path(entry["npz"]).exists():
                        continue
                    source, npz = "rank_scan", Path(entry["npz"])
                    cell = {"estimator": estimator, "M": M, "sigma": sigma,
                            "r": rank, "source": source, "npz": relative(npz),
                            "selected_rank": entry.get("selected_rank")}
                    cell.update(run_metrics(npz))
                    cells.append(cell)

    # The r = 128 arm of the scan should reproduce the main study: the bundle is
    # the canonical one, so any residual difference is numerical.
    main_study_check: list[dict] = []
    for estimator in SCANNED_RANKS:
        for M in CHECK_MASKS[estimator]:
            for sigma in CHECK_SIGMAS:
                reference = run_path(estimator, M, sigma, 0)
                scan_cell = next((c for c in cells if c["estimator"] == estimator
                                  and c["M"] == M and c["sigma"] == sigma
                                  and c["r"] == 128), None)
                if reference is None or scan_cell is None:
                    continue
                ref = run_metrics(Path(reference))
                delta = (abs(scan_cell["ger"] - ref["ger"]) / ref["ger"] * 100.0
                         if ref["ger"] else float("nan"))
                main_study_check.append({
                    "estimator": estimator, "M": M, "sigma": sigma, "r": 128,
                    "scan_ger": scan_cell["ger"], "main_ger": ref["ger"],
                    "delta_ger_pct": round(delta, 4),
                    "scan_s_full": scan_cell["s_full"], "main_s_full": ref["s_full"],
                    "main_npz": relative(Path(reference)),
                })

    result = {
        "schema": "luna.rank_scan_summary.v1",
        "task": "retained-rank sensitivity",
        "description": (
            "POD-based estimators re-run at retained ranks 100, 128 and 150 on "
            "the standard split; r = 128 is the main study. Mean global relative "
            "error (both components, physical units) and mean scale count "
            "(streamwise), tau = 0.05."),
        "scan_root": relative(scan_root),
        "scan_ranks": list(RANKS),
        "scanned_ranks": {k: list(v) for k, v in SCANNED_RANKS.items()},
        "sigmas": list(SIGMAS),
        "rank_selection": "validation-selected on clean data, capped by the number "
                          "of scalar observations (r <= 2M) for Gappy POD",
        "main_study_check": main_study_check,
        "cells": cells,
    }

    path = ROOT / "artifacts" / "statistics" / "rank_scan_summary.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        for estimator in SCANNED_RANKS:
            for sigma in (0.0, 0.01):
                row = [c for c in cells if c["estimator"] == estimator
                       and c["M"] == 20 and c["sigma"] == sigma]
                if row:
                    text = "  ".join(f"r={c['r']}: {c['ger']:.5f}/{c['s_full']:.2f}"
                                     for c in sorted(row, key=lambda c: c["r"]))
                    print(f"{estimator:6s} M=20 sigma={sigma:<5} {text}")
        for check in main_study_check:
            print(f"check {check['estimator']:6s} M={check['M']:2d} sigma={check['sigma']:<5} "
                  f"r=128 scan {check['scan_ger']:.5f} vs main study "
                  f"{check['main_ger']:.5f}  ({check['delta_ger_pct']:.3f} %)")
    print(f"[OK] {relative(path)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stability of the band norms used as denominators of the band errors.

Every band error in the paper is a relative error, so a band whose coefficient norm is close to zero would inflate the ratio. This module audits the denominators of the 300 test snapshots: it reports the distribution of the band norms, their share of the total energy, and how many snapshots fall below an absolute or relative threshold. The same statistics are reported for the projected norms that form the denominators of E_coh, `||Pi_b(W_b u)||_2`, with the affine projection `Pi_b` onto the band's POD subspace of the reference training split.

The quantities are evaluated in the wavelet coefficient domain, which is the domain in which the band errors of the paper are defined.

Inputs
    artifacts/<estimator runs>/    the target fields of one test split
Output
    artifacts/statistics/band_denominator_check.json
    artifacts/statistics/band_denominator_check.csv
    artifacts/statistics/tables/band_denominator_check.tex

Usage
    python -m applications.statistics.band_denominator_check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from applications.statistics.scoh_vs_sfull import band_pod_models  # noqa: E402
from features.metrics.band_error.denominator_audit import (  # noqa: E402
    audit_band_denominators,
    projected_denominators,
)
from features.training.estimator_runs import load_run, run_path  # noqa: E402

# : Any run of the test split provides the same target fields.
REFERENCE_RUN = ("mlp", 10, 0.0, 0)
OUTPUT = ROOT / "artifacts" / "statistics" / "band_denominator_check.json"
OUTPUT_CSV = ROOT / "artifacts" / "statistics" / "band_denominator_check.csv"
OUTPUT_TEX = ROOT / "artifacts" / "statistics" / "tables" / "band_denominator_check.tex"


def reference_run() -> Path:
    """Run whose test split provides the audited target fields."""
    path = run_path(*REFERENCE_RUN)
    if path is None:
        raise SystemExit(f"missing reference run {REFERENCE_RUN}")
    return path


def test_targets(path: Path) -> np.ndarray:
    """Streamwise-component test targets, shape (n_snapshots, 80, 160)."""
    target, _ = load_run(path)
    return target[:, 0]


def format_table(report: dict, bands: list[str]) -> str:
    lines = [
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Band & min & 1\% & 5\% & 25\% & median & 75\% & 95\% & max & near zero \\",
        r"\midrule",
    ]
    for band in bands:
        stats = report[band]["abs_norm"]
        near_zero = report[band]["near_zero"]
        lines.append(
            f"{band} & {stats['min']:.3e} & {stats['1%']:.3e} & {stats['5%']:.3e} "
            f"& {stats['25%']:.3e} & {stats['median']:.3e} & {stats['75%']:.3e} "
            f"& {stats['95%']:.3e} & {stats['max']:.3e} & {near_zero['any_count']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="compare with the submitted audit")
    args = parser.parse_args()

    cfg = get_config()
    bands = cfg.bands
    eps_abs, eps_rel = cfg.eps_abs, cfg.eps_rel

    start = time.time()
    print(f"== band denominator stability (bands={bands}, eps_abs={eps_abs}, "
          f"eps_rel={eps_rel})")
    targets = test_targets(reference_run())
    print(f"   targets: {targets.shape}")

    report = {
        key: value
        for key, value in audit_band_denominators(
            targets, bands=bands, eps_abs=eps_abs, eps_rel=eps_rel
        ).items()
        if not key.startswith("_")
    }

    projected = projected_denominators(
        targets, band_pod_models(reference_run()), bands=bands,
        wavelet=cfg.wavelet_family, level=cfg.wavelet_level, mode=cfg.wavelet_mode,
        eps_abs=eps_abs, eps_rel=eps_rel,
    )
    worst_ratio = min(projected[b]["ratio"]["min"] for b in bands)
    projected_near_zero = sum(projected[b]["near_zero"]["any_count"] for b in bands)
    print(f"   projected norms: min squared-norm ratio {worst_ratio:.3f}, "
          f"near-zero cases {projected_near_zero}")

    if args.verify:
        return _verify(report)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for band in bands:
        absolute = report[band]["abs_norm"]
        energy = report[band]["energy_fraction"]
        near_zero = report[band]["near_zero"]
        rows.append({
            "band": band,
            "abs_min": absolute["min"], "abs_1pct": absolute["1%"],
            "abs_5pct": absolute["5%"], "abs_25pct": absolute["25%"],
            "abs_median": absolute["median"], "abs_75pct": absolute["75%"],
            "abs_95pct": absolute["95%"], "abs_max": absolute["max"],
            "abs_mean": absolute["mean"],
            "energy_min": energy["min"], "energy_median": energy["median"],
            "energy_max": energy["max"], "energy_mean": energy["mean"],
            "near_zero_abs": near_zero["abs_count"],
            "near_zero_rel": near_zero["rel_count"],
            "near_zero_any": near_zero["any_count"],
            "n": absolute["n"],
        })
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT.write_text(
        json.dumps({
            "quantity": "band norms used as error denominators",
            "config": {
                "bands": bands, "eps_abs": eps_abs, "eps_rel": eps_rel,
                "wavelet": cfg.wavelet_family, "level": cfg.wavelet_level,
                "mode": cfg.wavelet_mode, "n_snapshots": int(targets.shape[0]),
                "domain": "wavelet coefficients",
            },
            "report": report,
            "projected": projected,
            "runtime_s": round(time.time() - start, 2),
        }, indent=2),
        encoding="utf-8",
    )

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(format_table(report, bands), encoding="utf-8")

    print(f"[OK] {OUTPUT.name} / {OUTPUT_CSV.name} / {OUTPUT_TEX.name} "
          f"({time.time() - start:.1f}s)")
    near_zero_total = 0
    for band in bands:
        absolute = report[band]["abs_norm"]
        energy = report[band]["energy_fraction"]
        near_zero = report[band]["near_zero"]
        near_zero_total += near_zero["any_count"]
        print(f"   {band}: norm median={absolute['median']:.3e} min={absolute['min']:.3e} | "
              f"energy share median={energy['median']:.3e} | near zero={near_zero}")
    print(f"   snapshots with a negligible band norm: {near_zero_total}")
    return 0


def _verify(report: dict) -> int:
    """Compare with the audit reported in the paper."""
    submitted = ROOT / "artifacts" / "derived" / "supplementary" / "band_denominator_audit.json"
    if not submitted.exists():
        print("   submitted audit not available; nothing to compare")
        return 1
    old = json.loads(submitted.read_text(encoding="utf-8"))["report"]
    print("== verification against the submitted audit")
    failures = 0
    for band in report:
        for section in ("abs_norm", "energy_fraction"):
            for key, value in report[band][section].items():
                reference = old[band][section][key]
                tolerance = 1e-12 * max(abs(reference), 1.0)
                if abs(value - reference) > tolerance:
                    print(f"   [FAIL] {band}.{section}.{key}: {value} != {reference}")
                    failures += 1
        if report[band]["near_zero"] != old[band]["near_zero"]:
            print(f"   [FAIL] {band}.near_zero: {report[band]['near_zero']} "
                  f"!= {old[band]['near_zero']}")
            failures += 1
    print("   all checks passed" if not failures else f"   {failures} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

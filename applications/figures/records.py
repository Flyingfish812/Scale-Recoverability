"""Views of the band-error records used by the figure scripts.

``applications/statistics/band_error_records.py`` stores one nested record per
(snapshot, estimator, sensor count, noise level, training seed) under
``artifacts/statistics/band_error_records.json``. The figures were written
against an older flat layout, so this module exposes the fields they need without
duplicating the file: a flat record view and, for the closed-form estimator, the
per-band means of one configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "artifacts" / "statistics" / "band_error_records.json"
BANDS = ["A4", "W4", "W3", "W2", "W1"]

_CACHE: list[dict] | None = None


def band_records(path: Path | None = None) -> list[dict]:
    """All records in the flat layout the figure scripts expect.

    Each entry carries the configuration, the global error, the scale count and
    the per-band total and reference band errors of one reconstruction.
    """
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE

    data = json.loads((path or RECORDS).read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data

    flat = []
    for record in records:
        entry = {
            "model_type": record["model"],
            "mask_num": record["sensor_count"],
            "seed": record["training_seed"],
            "noise_sigma": record["noise_sigma"],
            "sample_idx": record["snapshot_index"],
            "GER": record["global_error"],
            "S_full_total": record["s_full"],
            "S_coh_total": record.get("s_coh", -1),
        }
        for band in BANDS:
            entry[f"E_total_{band}"] = record["band_errors"][band]["total"]
            entry[f"E_trunc_{band}"] = record["band_errors"][band]["truncation"]
            entry[f"E_pred_{band}"] = record["band_errors"][band]["prediction"]
        flat.append(entry)

    if path is None:
        _CACHE = flat
    return flat


def per_band_means(model: str, sensors: int, sigma: float,
                   path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of the band error of one configuration."""
    selected = [r for r in band_records(path)
                if r["model_type"] == model and r["mask_num"] == sensors
                and r["noise_sigma"] == sigma]
    summary = {}
    for band in BANDS:
        values = [r[f"E_total_{band}"] for r in selected]
        summary[band] = ((sum(values) / len(values), np_std(values))
                         if values else (0.0, 0.0))
    return summary


def config_summary(model: str, sensors: int, sigma: float,
                   path: Path | None = None) -> dict:
    """Headline means of one configuration, as reported in the tables."""
    selected = [r for r in band_records(path)
                if r["model_type"] == model and r["mask_num"] == sensors
                and r["noise_sigma"] == sigma]
    if not selected:
        return {"n": 0, "GER_mean": 0.0, "S_full_mean": 0.0,
                "per_band_mean": dict.fromkeys(BANDS, 0.0),
                "per_band_std": dict.fromkeys(BANDS, 0.0)}

    bands = per_band_means(model, sensors, sigma, path)
    return {
        "n": len(selected),
        "GER_mean": sum(r["GER"] for r in selected) / len(selected),
        "S_full_mean": sum(r["S_full_total"] for r in selected) / len(selected),
        "per_band_mean": {b: v[0] for b, v in bands.items()},
        "per_band_std": {b: v[1] for b, v in bands.items()},
    }


def np_std(values: list[float]) -> float:
    """Standard deviation of a list, without importing numpy for one call."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def truncation_global_error() -> float:
    """Mean global error of the rank-128 POD truncation over the test snapshots."""
    raw = json.loads(RECORDS.read_text(encoding="utf-8"))["records"]
    per_snapshot = {r["snapshot_index"]: r["truncation_global_error"] for r in raw}
    values = list(per_snapshot.values())
    return float(sum(values) / len(values)) if values else 0.0


def truncation_scale_count() -> float:
    """Scale count of the rank-128 POD truncation, read from the audit table."""
    audit = json.loads((ROOT / "artifacts" / "statistics"
                        / "truncation_reference_audit.json").read_text(encoding="utf-8"))
    nc = next(s for s in audit["summaries"] if s["dataset"] == "nc")
    bands = nc["table"]["128"]["bands"]
    tau = audit["tau"]
    count = 0
    for band in BANDS:
        if float(bands[band]["mean"]) <= tau:
            count += 1
        else:
            break
    return float(count)

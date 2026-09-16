"""Distribution of POD modal energy across spatial scales.

Each POD mode is decomposed into the five wavelet bands, and the fraction of
band energy carried by the first r modes gives the cumulative coverage curve

    C_b(r) = Σ_{j<=r} λ_j E_b(φ_j) / Σ_{j<=R} λ_j E_b(φ_j)

where E_b(φ_j) is the energy of mode j inside band b. Steep curves mean that a
band is described by a few modes; shallow curves mean that it is spread over many
weak modes. The number of modes needed for 90% and 99% coverage of each band is
reported explicitly, since that is the quantity quoted in the paper.

Inputs
    artifacts/pod_bases/...    rank-128 POD basis (both components)
Output
    artifacts/statistics/mode_scale_energy.json

Usage
    python -m applications.statistics.mode_scale_energy
    python -m applications.statistics.mode_scale_energy --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.core.constants import (  # noqa: E402
    BANDS_CF,
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    DEFAULT_WAVELET,
)
from luna.data.io import load_npz  # noqa: E402

POD_BUNDLE = ROOT / "artifacts" / "pod_bases" / "cylinder2d_q1" / "pod_base_bundle.npz"
OUTPUT = ROOT / "artifacts" / "statistics" / "mode_scale_energy.json"
COVERAGE_THRESHOLDS = (0.5, 0.9, 0.99)

# Reference values quoted in the paper (A4 and W1 bands, NC basis).
EXPECTED = {"A4": {0.9: 12, 0.99: 35}, "W1": {0.9: 92, 0.99: 124}}


def _zero_like(coeffs):
    out = [np.zeros_like(coeffs[0])]
    for detail in coeffs[1:]:
        out.append(tuple(np.zeros_like(part) for part in detail))
    return out


def _band_components(field: np.ndarray) -> dict[str, np.ndarray]:
    """Five band-limited components of a 2D field (coarse to fine)."""
    coeffs = pywt.wavedec2(field, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                           mode=DEFAULT_MODE)
    height, width = field.shape
    components: dict[str, np.ndarray] = {}
    # wavedec2 returns [approximation, detail at the coarsest level, ..., finest]
    levels = {"A4": 0, "W4": 1, "W3": 2, "W2": 3, "W1": 4}
    for band, index in levels.items():
        kept = _zero_like(coeffs)
        kept[index] = coeffs[index]
        reconstructed = pywt.waverec2(kept, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE)
        components[band] = np.asarray(reconstructed[:height, :width], dtype=np.float64)
    return components


def band_energy_per_mode(basis: np.ndarray, singular_values: np.ndarray) -> dict[str, np.ndarray]:
    """Energy of every mode in every band, weighted by its singular value."""
    n_modes = basis.shape[0]
    energy = {band: np.zeros(n_modes) for band in BANDS_CF}
    for j in range(n_modes):
        for channel in range(basis.shape[-1]):
            components = _band_components(basis[j, :, :, channel])
            for band in BANDS_CF:
                energy[band][j] += singular_values[j] * float(np.sum(components[band] ** 2))
    return energy


def cumulative_coverage(energy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {band: np.cumsum(values) / np.sum(values) for band, values in energy.items()}


def modes_for_threshold(curve: np.ndarray, threshold: float) -> int:
    index = np.where(curve >= threshold)[0]
    return int(index[0]) + 1 if index.size else int(curve.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        return verify(payload["threshold_modes"])

    started = time.time()
    bundle = load_npz(str(POD_BUNDLE))
    basis = np.asarray(bundle["pod_basis"], dtype=np.float64)
    singular_values = np.asarray(bundle["singular_values"], dtype=np.float64)
    print(f"POD basis: {basis.shape}, rank {basis.shape[0]}")

    energy = band_energy_per_mode(basis, singular_values)
    coverage = cumulative_coverage(energy)
    thresholds = {
        band: {str(threshold): modes_for_threshold(coverage[band], threshold)
               for threshold in COVERAGE_THRESHOLDS}
        for band in BANDS_CF
    }
    for band in BANDS_CF:
        print(f"  {band}: " + ", ".join(
            f"{int(threshold * 100)}% -> {thresholds[band][str(threshold)]} modes"
            for threshold in COVERAGE_THRESHOLDS))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "cumulative POD mode energy per wavelet band",
        "config": {
            "dataset": "nc",
            "wavelet": DEFAULT_WAVELET,
            "level": DEFAULT_LEVEL,
            "mode": DEFAULT_MODE,
            "n_modes": int(basis.shape[0]),
            "weighting": "singular_value * band energy of the mode",
        },
        "band_energy_per_mode": {band: energy[band].tolist() for band in BANDS_CF},
        "cumulative_coverage": {band: coverage[band].tolist() for band in BANDS_CF},
        "singular_values": singular_values.tolist(),
        # lambda_j / lambda_1: the modal energy is the squared singular value
        "mode_energy_ratio": ((singular_values / singular_values[0]) ** 2).tolist(),
        "threshold_modes": thresholds,
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)} ({time.time() - started:.0f} s)")
    if args.verify:
        return verify(thresholds)
    return 0


def verify(thresholds: dict) -> int:
    print("\nverification against the values quoted in the paper")
    ok = True
    for band, expected in EXPECTED.items():
        for threshold, want in expected.items():
            got = thresholds[band][str(threshold)]
            flag = "OK" if got == want else "DIFFERS"
            print(f"  {band} {int(threshold * 100)}% coverage: {got} modes "
                  f"(paper: {want})  {flag}")
            ok &= got == want
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

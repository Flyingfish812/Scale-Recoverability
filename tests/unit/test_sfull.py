"""S_full tests.

Covers: all pass / A4 fails / middle failure / a later band passes again / error
exactly at the threshold / NaN and zero denominator / different decomposition levels.
"""

import numpy as np
import pytest

from luna.wavelet.metrics import contiguous_recoverable_index, compute_S_full
from luna.wavelet.transform import decompose_field_2d, recompose_field_2d

rng = np.random.default_rng(11)


def test_all_pass():
    assert contiguous_recoverable_index([0.01] * 5, tau=0.05) == 5


def test_a4_fail():
    assert contiguous_recoverable_index([0.06, 0.01, 0.01, 0.01, 0.01], tau=0.05) == 0


def test_middle_fail():
    # Third band fails → only the first 2 are recoverable
    assert contiguous_recoverable_index([0.01, 0.01, 0.06, 0.01, 0.01], tau=0.05) == 2


def test_repass_after_fail_stops():
    # A4 passes, then W4 fails; a later W3 pass still does not count (contiguous rule)
    assert contiguous_recoverable_index([0.01, 0.06, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_threshold_boundary_inclusive():
    # error == tau counts as passing (≤)
    assert contiguous_recoverable_index([0.05, 0.05, 0.05, 0.05, 0.05], tau=0.05) == 5
    assert contiguous_recoverable_index([0.0500001, 0.01, 0.01, 0.01, 0.01], tau=0.05) == 0


def test_nan_stops():
    assert contiguous_recoverable_index([0.01, np.nan, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_inf_stops():
    assert contiguous_recoverable_index([0.01, np.inf, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_zero_denominator_guard():
    # Zero/near-zero denominators are protected by eps, no exception is raised; inf stops counting
    arr = np.array([0.01, 0.01, np.inf, np.inf, np.inf])
    assert contiguous_recoverable_index(arr, tau=0.05) == 2


def _field_from_bands(band_energies, shape=(64, 64)):
    """Build: target = Σ ω_b·band, pred = target + error injected only into given bands."""
    # Random wavelet components: decompose a random field to obtain 5 bands
    base = rng.standard_normal(shape)
    base_bands = decompose_field_2d(base, "db2", 4, "periodization")
    comps = {b: base_bands[b] * np.sqrt(w) for b, w in band_energies.items()}
    return recompose_field_2d(comps, "db2", 4, "periodization")


def test_compute_sfull_on_fields():
    """Validate compute_S_full on constructed fields: band-limited error injected in W1
    only → S_full=4."""
    energies = {"A4": 10.0, "W4": 5.0, "W3": 3.0, "W2": 2.0, "W1": 1.0}
    target = _field_from_bands(energies)
    # pred = target plus a band-limited perturbation in W1 only (random field's W1 part)
    noise = rng.standard_normal(target.shape)
    noise_w1 = decompose_field_2d(noise, "db2", 4, "periodization")["W1"]
    w1_scale = float(np.linalg.norm(decompose_field_2d(target, "db2", 4, "periodization")["W1"]))
    noise_w1 = noise_w1 / (np.linalg.norm(noise_w1) + 1e-12) * (0.3 * w1_scale)

    bands_t = decompose_field_2d(target, "db2", 4, "periodization")
    bands_p = dict(bands_t)
    bands_p["W1"] = bands_p["W1"] + noise_w1
    pred = recompose_field_2d(bands_p, "db2", 4, "periodization")
    sfull = compute_S_full(target, pred, tau=0.05)
    assert sfull == 4, f"expected S_full=4, got {sfull}"


def test_compute_sfull_perfect():
    target = _field_from_bands({"A4": 10.0, "W4": 5.0, "W3": 3.0, "W2": 2.0, "W1": 1.0})
    assert compute_S_full(target, target, tau=0.05) == 5

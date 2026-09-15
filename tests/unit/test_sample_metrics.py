"""The shared sample metrics must reproduce the values used for the paper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from features.metrics.sample_metrics import (  # noqa: E402
    BANDS,
    band_errors,
    compute_sample_metrics,
    gradient_rmse,
    laplacian,
    sigma_to_code,
)



def test_sigma_to_code_matches_run_layout():
    assert sigma_to_code(0.0) == "s0000"
    assert sigma_to_code(0.001) == "s0010"
    assert sigma_to_code(0.01) == "s0100"
    assert sigma_to_code(0.1) == "s1000"


def test_band_errors_are_relative_and_bounded():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(2, 80, 160))
    pred = target + 0.01 * rng.normal(size=(2, 80, 160))
    res = band_errors(target[0], pred[0])
    assert set(res["band_errors"]) == set(BANDS)
    assert all(0.0 <= e < 0.5 for e in res["band_errors"].values())
    assert 0 <= res["n_bands_below_tau"] <= len(BANDS)


def test_perfect_prediction_has_zero_error():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(1, 2, 80, 160))
    metrics = compute_sample_metrics(target, target, 0)
    assert metrics["GER"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["S_full"] == len(BANDS)
    assert metrics["vorticity_RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["gradient_RMSE"] == pytest.approx(0.0, abs=1e-12)


def test_laplacian_matches_gradient_composition():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(80, 160))
    expected = np.gradient(np.gradient(field, axis=0), axis=0) + np.gradient(
        np.gradient(field, axis=1), axis=1
    )
    assert np.allclose(laplacian(field), expected)
    assert gradient_rmse(field, field) == pytest.approx(0.0, abs=1e-12)


GOLDEN = [
    {"GER": 0.0499084614, "S_full": 3, "E_W1": 0.0502094182, "E_A4": 0.0509949893,
     "vorticity_RMSE": 0.0563394153, "gradient_RMSE": 0.051145994},
    {"GER": 0.0500693121, "S_full": 3, "E_W1": 0.0502029468, "E_A4": 0.0459953428,
     "vorticity_RMSE": 0.0561481171, "gradient_RMSE": 0.051207088},
    {"GER": 0.0496520733, "S_full": 4, "E_W1": 0.0499406068, "E_A4": 0.0446997208,
     "vorticity_RMSE": 0.0565651641, "gradient_RMSE": 0.0512418468},
]


def test_metric_values_are_pinned():
    """The metric definitions must not drift: values are compared with a
    reference computed when the definitions were consolidated."""
    rng = np.random.default_rng(7)
    out = rng.normal(size=(3, 2, 80, 160))
    tgt = out + 0.05 * rng.normal(size=(3, 2, 80, 160))
    for index, expected in enumerate(GOLDEN):
        metrics = compute_sample_metrics(out, tgt, index)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, rel=1e-8), (index, key)

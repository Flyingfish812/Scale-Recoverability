"""Supplementary denominator audit tests (P1-1)."""

import numpy as np
import pywt

from features.metrics.band_error.denominator_audit import (
    band_coefficient_norms,
    energy_fractions,
    audit_band_denominators,
)

BANDS = ["A4", "W4", "W3", "W2", "W1"]
rng = np.random.default_rng(3)


def _random_fields(n=16, shape=(80, 160)):
    return rng.standard_normal((n, *shape))


def test_energy_fractions_sum_to_one():
    u = rng.standard_normal((80, 160))
    norms = band_coefficient_norms(u)
    q = energy_fractions(norms)
    assert abs(sum(q.values()) - 1.0) < 1e-6  # Parseval (系数域)


def test_band_norms_match_coefficient_domain():
    u = rng.standard_normal((80, 160))
    coeffs = pywt.wavedec2(u, "db2", level=4, mode="periodization")
    norms = band_coefficient_norms(u)
    assert abs(norms["A4"] - np.linalg.norm(coeffs[0])) < 1e-6
    # pywt: coeffs[1..4] = W4..W1 (coarsest->finest)
    w1 = np.sqrt(sum(np.sum(d ** 2) for d in coeffs[-1]))  # finest band
    assert abs(norms["W1"] - w1) < 1e-6
    w4 = np.sqrt(sum(np.sum(d ** 2) for d in coeffs[1]))
    assert abs(norms["W4"] - w4) < 1e-6


def test_audit_report_structure_and_no_nearzero_for_full_energy_fields():
    rep = audit_band_denominators(_random_fields(), bands=BANDS, eps_abs=1e-12, eps_rel=1e-12)
    for b in BANDS:
        assert rep[b]["abs_norm"]["n"] == 16
        assert rep[b]["abs_norm"]["near_zero_count"] == 0
        assert 0.0 < rep[b]["energy_fraction"]["median"] <= 1.0
    assert rep["_field"]["n"] == 16


def test_audit_detects_near_zero():
    fields = _random_fields()
    fields[:, :, :] = 0.0  # zero fields -> near-zero denominators
    rep = audit_band_denominators(fields, bands=BANDS, eps_abs=1e-8, eps_rel=1e-6)
    for b in BANDS:
        assert rep[b]["near_zero"]["abs_count"] == 16

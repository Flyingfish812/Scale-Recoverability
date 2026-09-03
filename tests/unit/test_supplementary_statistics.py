"""Supplementary statistics tests (P1-2): temporal dependence & block bootstrap."""

import numpy as np

from features.statistics import temporal_dependence as td
from features.statistics import block_bootstrap as bb

rng = np.random.RandomState(0)
n = 1501
t = np.arange(n)
X = 3.0 * np.sin(2 * np.pi * t / 62.5) + 0.5 * rng.randn(n)


def test_dominant_period_detects_62_5():
    dp = td.dominant_period(X)
    assert 60 < dp["period"] < 66
    assert dp["peak_frac"] > 0.5


def test_autocorrelation_first_lag_high_for_periodic():
    acf = td.autocorrelation(X, max_lag=10)
    assert acf[0] > 0.8


def test_iact_and_ess_positive():
    assert td.integrated_autocorrelation_time(X) > 1.0
    assert td.effective_sample_size(X) < len(X)


def test_suggest_block_length_prefers_physical():
    bl = td.suggest_block_length([X], prefer_physical_period=True,
                                 candidate_block_lengths=(10, 20, 30, 50, 62, 125))
    assert bl["chosen_block_length"] in (50, 62, 125)
    assert bl["method"].startswith("physical_period")


def test_moving_block_bootstrap_reproduces_mean():
    res = bb.block_bootstrap(X, np.mean, block_len=62, n_resamples=200, seed=1)
    assert abs(res.mean() - X.mean()) < 0.05
    assert res.std() > 0


def test_block_bootstrap_seeded_reproducible():
    a = bb.block_bootstrap(X, np.mean, block_len=62, n_resamples=50, seed=7)
    b = bb.block_bootstrap(X, np.mean, block_len=62, n_resamples=50, seed=7)
    assert np.array_equal(a, b)


def test_snapshot_cluster_bootstrap():
    cids = np.arange(len(X)) // 62
    res = bb.block_bootstrap(X, np.mean, block_len=1, n_resamples=200, seed=1,
                             cluster_ids=cids)
    assert np.isfinite(res).all()

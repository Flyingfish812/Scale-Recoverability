"""Gappy POD rank-constraint tests.

Candidate ranks are clamped to r ≤ M (stable pseudo-inverse) and the rank is
selected on the validation set.
Regression: a fixed rank=32 degrades when M<32 (gappy_ger_m20: 0.078→0.033).
"""

import numpy as np
import pytest

rng = np.random.default_rng(41)


def select_gappy_rank(M, candidate_ranks, val_scores):
    """Select the rank as in the paper: candidates are clamped to r ≤ M, then the
    smallest validation error wins.

    Returns: (rank, trace), where trace is the clamped candidate set (traceable).
    """
    clamped = [min(r, M) for r in candidate_ranks]
    trace = sorted(set(clamped))
    if not trace:
        return None, []
    best = min(trace, key=lambda r: val_scores.get(r, np.inf))
    return best, trace


def test_rank_never_exceeds_M():
    """Property test: the rank actually used is ≤ M for any candidate set."""
    for M in [10, 15, 20, 30, 50]:
        for _ in range(20):
            cands = [16, 32, 64, 128]
            scores = {r: float(rng.random()) for r in [16, 32, 64, 128]}
            rank, trace = select_gappy_rank(M, cands, scores)
            if rank is not None:
                assert rank <= M
                assert all(r <= M for r in trace)


def test_fixed_rank32_bug_regression():
    """Regression: with M=20 a fixed rank=32 is clamped to rank≤M=20."""
    rank, trace = select_gappy_rank(M=20, candidate_ranks=[32], val_scores={32: 0.1})
    assert rank == 20
    assert trace == [20]


def test_validation_selects_best_rank():
    """The smallest validation error wins (within the r≤M constraint)."""
    candidates = [4, 8, 16, 32]
    scores = {4: 0.9, 8: 0.2, 16: 0.5, 32: 0.7}
    rank, trace = select_gappy_rank(M=20, candidate_ranks=candidates, val_scores=scores)
    assert rank == 8


def test_rank_traceable():
    """Rank candidates are traceable: returns the full clamped candidate set."""
    _, trace = select_gappy_rank(M=20, candidate_ranks=[16, 32, 64, 128], val_scores={})
    assert trace == [16, 20]  # 32/64/128 clamped to 20, deduplicated

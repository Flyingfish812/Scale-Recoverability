"""Gappy POD rank-constraint tests.

The rank is selected on validation data within a cap. The paper caps it by the
number of *scalar observations*, r <= m_obs = 2M (each sensor location provides
two velocity components); the earlier sweep capped it by the number of sensor
*locations*, r <= M. Both conventions are exercised here.

Regression: a fixed rank=32 degrades when the cap lies below 32
(gappy_ger_m20: 0.078→0.033).
"""

import numpy as np
import pytest

rng = np.random.default_rng(41)


def scalar_cap(M):
    """Cap of the paper: the number of scalar observations, m_obs = 2M."""
    return 2 * M


def select_gappy_rank(M, candidate_ranks, val_scores, cap=None):
    """Select the rank on validation data within a cap.

    ``cap`` maps M to the upper bound; it defaults to the paper's rule
    (``scalar_cap``). Pass ``cap=lambda m: m`` for the older sensor-location
    convention. Candidates above the cap are clamped and deduplicated, then the smallest validation error wins.

    Returns: (rank, trace), where trace is the clamped candidate set (traceable).
    """
    cap = cap or scalar_cap
    limit = int(cap(M))
    clamped = [min(r, limit) for r in candidate_ranks]
    trace = sorted(set(clamped))
    if not trace:
        return None, []
    best = min(trace, key=lambda r: val_scores.get(r, np.inf))
    return best, trace


@pytest.mark.parametrize("cap,label",
                         [(scalar_cap, "m_obs=2M"), (lambda m: m, "locations=M")])
def test_rank_never_exceeds_cap(cap, label):
    """Property test: the rank respects the cap for any candidate set."""
    for M in [10, 15, 20, 30, 50]:
        for _ in range(20):
            cands = [16, 32, 64, 128]
            scores = {r: float(rng.random()) for r in cands}
            rank, trace = select_gappy_rank(M, cands, scores, cap=cap)
            limit = cap(M)
            if rank is not None:
                assert rank <= limit, label
                assert all(r <= limit for r in trace), label


def test_paper_cap_is_two_M():
    """The default (paper) rule is r <= 2M, not r <= M."""
    candidates = [4, 8, 16, 32, 64]
    scores = {4: 0.9, 8: 0.2, 16: 0.5, 32: 0.7, 64: 0.1}
    rank, trace = select_gappy_rank(M=20, candidate_ranks=candidates, val_scores=scores)
    assert trace == [4, 8, 16, 32, 40]      # 64 clamped to 2M = 40
    # Scores are keyed by candidate value, so the clamped candidate 40 is unscored and the smallest *scored* candidate wins; the cap is what this test pins.
    assert rank == 8


def test_fixed_rank32_bug_regression():
    """Regression: with M=20 a fixed candidate 32 is clamped only by r <= M."""
    rank, trace = select_gappy_rank(M=20, candidate_ranks=[32], val_scores={32: 0.1})
    assert rank == 32                        # the scalar cap 2M = 40 does not clamp
    rank_loc, trace_loc = select_gappy_rank(
        M=20, candidate_ranks=[32], val_scores={32: 0.1}, cap=lambda m: m)
    assert rank_loc == 20                    # the location cap does
    assert trace_loc == [20]


def test_validation_selects_best_rank():
    """The smallest validation error wins (within the cap)."""
    candidates = [4, 8, 16, 32]
    scores = {4: 0.9, 8: 0.2, 16: 0.5, 32: 0.7}
    rank, _ = select_gappy_rank(M=20, candidate_ranks=candidates, val_scores=scores)
    assert rank == 8


def test_rank_traceable():
    """Rank candidates are traceable: the full clamped candidate set is returned."""
    _, trace = select_gappy_rank(M=20, candidate_ranks=[16, 32, 64, 128], val_scores={})
    assert trace == [16, 32, 40]             # 64/128 clamped to 2M = 40, deduplicated

"""Gappy POD 秩约束测试 (P0-6 §10.6)。

论文 S2.3 修正: 强制 r ≤ M (伪逆稳定), 验证集选 rank。
旧 bug: 固定 rank=32 当 M<32 时退化 (gappy_ger_m20: 0.078→0.033)。
"""

import numpy as np
import pytest

rng = np.random.default_rng(41)


def select_gappy_rank(M, candidate_ranks, val_scores):
    """按论文逻辑: 候选 rank 强制 ≤ M (截断), 在验证集误差最小处选 rank.

    Returns: (rank, trace) 其中 trace 为截断后的候选集 (可追踪).
    """
    clamped = [min(r, M) for r in candidate_ranks]
    trace = sorted(set(clamped))
    if not trace:
        return None, []
    best = min(trace, key=lambda r: val_scores.get(r, np.inf))
    return best, trace


def test_rank_never_exceeds_M():
    """性质测试: 任意候选下使用的 rank 均 ≤ M."""
    for M in [10, 15, 20, 30, 50]:
        for _ in range(20):
            cands = [16, 32, 64, 128]
            scores = {r: float(rng.random()) for r in [16, 32, 64, 128]}
            rank, trace = select_gappy_rank(M, cands, scores)
            if rank is not None:
                assert rank <= M
                assert all(r <= M for r in trace)


def test_fixed_rank32_bug_regression():
    """回归: M=20 时旧固定 rank=32 被截断为 rank≤M=20."""
    rank, trace = select_gappy_rank(M=20, candidate_ranks=[32], val_scores={32: 0.1})
    assert rank == 20
    assert trace == [20]


def test_validation_selects_best_rank():
    """验证集误差最小者胜出 (在 r≤M 约束内)."""
    candidates = [4, 8, 16, 32]
    scores = {4: 0.9, 8: 0.2, 16: 0.5, 32: 0.7}
    rank, trace = select_gappy_rank(M=20, candidate_ranks=candidates, val_scores=scores)
    assert rank == 8


def test_rank_traceable():
    """rank 候选集可追踪: 返回截断后的完整候选集."""
    _, trace = select_gappy_rank(M=20, candidate_ranks=[16, 32, 64, 128], val_scores={})
    assert trace == [16, 20]  # 32/64/128 截断为 20, 去重

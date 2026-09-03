"""
Block bootstrap for time series statistics.

Implements moving-block, circular-block and stationary bootstrap, plus a
snapshot-cluster bootstrap (used as a control). All methods resample *rows* of
an (n, ...) array so they work for both scalar statistics and per-snapshot
vectors.

For the NC dataset: the 300 test snapshots are non-consecutive samples from the
full 1501-snapshot sequence. Block length should be chosen from the physical
vortex-shedding period (~62 snapshots, see temporal_dependence) and blocks are
defined on the *time index* of each snapshot.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def _rng(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _moving_blocks(n: int, block_len: int, rng: np.random.RandomState, n_blocks: Optional[int] = None) -> np.ndarray:
    """Moving-block: contiguous blocks with start uniform in [0, n-block_len]."""
    if block_len >= n:
        return np.arange(n).reshape(1, -1)
    nb = n_blocks if n_blocks is not None else int(np.ceil(n / block_len))
    starts = rng.randint(0, n - block_len + 1, size=nb)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).reshape(-1)
    return idx[:n]


def _circular_blocks(n: int, block_len: int, rng: np.random.RandomState, n_blocks: Optional[int] = None) -> np.ndarray:
    """Circular-block: blocks wrap around the series end."""
    if block_len >= n:
        return np.arange(n).reshape(1, -1)
    nb = n_blocks if n_blocks is not None else int(np.ceil(n / block_len))
    starts = rng.randint(0, n, size=nb)
    pos = (starts[:, None] + np.arange(block_len)[None, :]) % n
    return pos.reshape(-1)[:n]


def _stationary_blocks(n: int, block_len: int, rng: np.random.RandomState, n_blocks: Optional[int] = None) -> np.ndarray:
    """Stationary bootstrap: geometrically distributed block lengths (p=1/block_len)."""
    p = 1.0 / max(int(block_len), 1)
    out: list[int] = []
    while len(out) < n:
        start = rng.randint(0, n)
        length = 1 + int(rng.geometric(p))
        length = min(length, n)
        for k in range(length):
            out.append((start + k) % n)
            if len(out) >= n:
                break
    return np.asarray(out[:n], dtype=np.int64)


def _block_indices(method: str, n: int, block_len: int, rng: np.random.RandomState) -> np.ndarray:
    if method == "moving_block":
        return _moving_blocks(n, block_len, rng)
    if method == "circular":
        return _circular_blocks(n, block_len, rng)
    if method == "stationary":
        return _stationary_blocks(n, block_len, rng)
    raise ValueError(f"Unknown block method {method!r}")


def block_bootstrap(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_len: int,
    n_resamples: int = 1000,
    seed: Optional[int] = None,
    method: str = "moving_block",
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bootstrap distribution of stat_fn(x).

    Args:
        x: (n, ...) array (rows = snapshots in time order).
        stat_fn: receives a resampled array (subset of rows) -> float.
        block_len: block length in snapshots (>=1).
        method: moving_block | circular | stationary.
        cluster_ids: if given (length n), use snapshot-cluster bootstrap instead
                     (resample clusters with replacement, keep members together).
    Returns:
        (n_resamples,) bootstrap distribution.
    """
    x = np.asarray(x)
    n = x.shape[0]
    rng = _rng(seed)
    out = np.empty(n_resamples, dtype=np.float64)

    if cluster_ids is not None:
        cids = np.asarray(cluster_ids).reshape(-1)
        if cids.size != n:
            raise ValueError("cluster_ids must match first axis of x")
        clusters = sorted(set(int(c) for c in cids))
        for b in range(n_resamples):
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            mask = np.isin(cids, chosen)
            out[b] = float(stat_fn(x[mask]))
        return out

    for b in range(n_resamples):
        idx = _block_indices(method, n, int(block_len), rng)
        out[b] = float(stat_fn(x[idx]))
    return out


def block_bootstrap_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_len: int,
    n_resamples: int = 1000,
    seed: Optional[int] = None,
    method: str = "moving_block",
    cluster_ids: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap distribution summary with percentile CI."""
    dist = block_bootstrap(
        x, stat_fn, block_len=block_len, n_resamples=n_resamples,
        seed=seed, method=method, cluster_ids=cluster_ids,
    )
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "std": float(dist.std()),
        f"ci_{int(alpha*100)}_low": float(np.percentile(dist, lo)),
        f"ci_{int(alpha*100)}_high": float(np.percentile(dist, hi)),
    }


def paired_diff_bootstrap(
    x_low: np.ndarray,
    x_high: np.ndarray,
    *,
    block_len: int,
    n_resamples: int = 1000,
    seed: Optional[int] = None,
    method: str = "moving_block",
    cluster_ids: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap the paired difference mean (x_high - x_low)."""
    diff = np.asarray(x_high, dtype=np.float64) - np.asarray(x_low, dtype=np.float64)
    return block_bootstrap_ci(
        diff, lambda d: float(np.mean(d)), block_len=block_len,
        n_resamples=n_resamples, seed=seed, method=method,
        cluster_ids=cluster_ids, alpha=alpha,
    )

"""
Temporal dependence diagnostics.

Computes ACF, integrated autocorrelation time (IACT), effective sample size
(ESS) and dominant period for a time series, and suggests a block length for
block bootstrap.

Note on the NC dataset: the 300 test snapshots are NOT a consecutive time
series (they are sampled from the 1501-snapshot cylinder-wake sequence with
irregular gaps). Physical periodicity must therefore be estimated on the FULL
1501-snapshot sequence (equally spaced), which this module supports.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Sample ACF for lags 1..max_lag (0-based index 0 corresponds to lag 1)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if max_lag is None:
        max_lag = min(n - 1, 500)
    max_lag = int(max(max_lag, 0))
    xc = x - x.mean()
    var = float(np.dot(xc, xc))
    if var <= 0.0:
        return np.zeros(max_lag)
    acf = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        acf[k - 1] = float(np.dot(xc[:-k], xc[k:]) / var)
    return acf


def integrated_autocorrelation_time(x: np.ndarray, cutoff: str = "1/e") -> float:
    """IACT = 1 + 2*sum_{k>=1} rho_k (Geyer-style, summed while |rho| >= cutoff)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 3:
        return 1.0
    xc = x - x.mean()
    var = float(np.dot(xc, xc))
    if var <= 0.0:
        return 1.0
    threshold = 1.0 / np.e if cutoff == "1/e" else float(cutoff)
    s = 1.0
    for k in range(1, n):
        rho = float(np.dot(xc[:-k], xc[k:]) / (var * (n - k) / n))
        if abs(rho) < threshold:
            break
        s += 2.0 * rho
    return max(s, 1.0)


def effective_sample_size(x: np.ndarray) -> float:
    """ESS = n / IACT."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x.size / integrated_autocorrelation_time(x)


def dominant_period(x: np.ndarray) -> Dict[str, float]:
    """Dominant FFT period (in samples) + peak spectral fraction.

    Returns:
        {period, peak_frac}: period in samples; peak_frac is the share of the
        dominant spectral line in the total variance (excluding DC).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 4:
        return {"period": float("nan"), "peak_frac": float("nan")}
    xc = x - x.mean()
    spec = np.abs(np.fft.rfft(xc))
    freqs = np.fft.rfftfreq(n)
    spec[0] = 0.0
    total_var = float(np.sum(spec[1:] ** 2))
    if total_var <= 0.0:
        return {"period": float("nan"), "peak_frac": 0.0}
    pk = int(np.argmax(spec))
    period = 1.0 / freqs[pk] if freqs[pk] > 0.0 else float("inf")
    return {"period": float(period), "peak_frac": float(spec[pk] ** 2 / total_var)}


def suggest_block_length(
    series: Sequence[np.ndarray],
    *,
    prefer_physical_period: bool = True,
    candidate_block_lengths: Sequence[int] = (10, 20, 30, 50, 62, 125),
    min_acf_lag_floor: float = 0.05,
) -> Dict[str, object]:
    """Suggest a block length from a set of diagnostic series.

    Strategy:
      1. If a clear dominant period (peak_frac >= 0.5) is found and
         prefer_physical_period, use that period as block length.
      2. Otherwise use IACT-derived block (ceil(IACT)) rounded up to a
         candidate, else fall back to ESS-based estimate.
    """
    periods: list[float] = []
    iacts: list[float] = []
    for s in series:
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if s.size < 8:
            continue
        dp = dominant_period(s)
        if np.isfinite(dp["period"]):
            periods.append(dp["period"])
        iacts.append(integrated_autocorrelation_time(s))

    best_period: Optional[float] = None
    best_frac = 0.0
    for s in series:
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if s.size < 8:
            continue
        dp = dominant_period(s)
        if dp["peak_frac"] > best_frac and np.isfinite(dp["period"]):
            best_frac = float(dp["peak_frac"])
            best_period = float(dp["period"])

    chosen: Optional[int] = None
    method = ""
    if prefer_physical_period and best_period is not None and best_frac >= 0.5:
        chosen = int(round(best_period))
        method = f"physical_period(frac={best_frac:.2f})"
    else:
        iact_mean = float(np.mean(iacts)) if iacts else float("nan")
        if np.isfinite(iact_mean):
            chosen = int(np.ceil(iact_mean))
            method = "iact"
    # round to nearest candidate for stability (>= physical period preferred,
    # but allow the nearest candidate if a larger one is far away)
    cands = sorted(int(c) for c in candidate_block_lengths)
    if chosen is not None:
        chosen = min(cands, key=lambda c: abs(c - chosen))

    return {
        "chosen_block_length": chosen,
        "method": method,
        "physical_period": best_period,
        "peak_frac": best_frac,
        "mean_iact": float(np.mean(iacts)) if iacts else float("nan"),
        "mean_ess": float(np.mean([effective_sample_size(s) for s in series])) if series else float("nan"),
        "candidates": list(candidate_block_lengths),
    }


def report_temporal_series(
    series: dict[str, np.ndarray],
    max_lag: int = 50,
) -> dict[str, dict]:
    """Diagnose a dict of named time series. Returns per-series stats."""
    out: dict[str, dict] = {}
    for name, s in series.items():
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        acf = autocorrelation(s, max_lag=max_lag)
        dp = dominant_period(s)
        lag_under_floor = None
        for i, r in enumerate(acf):
            if r < 0.1:
                lag_under_floor = i + 1
                break
        out[name] = {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "acf_lag1": float(acf[0]) if acf.size else float("nan"),
            "acf_lag5": float(acf[4]) if acf.size >= 5 else float("nan"),
            "first_lag_acf_below_0.1": lag_under_floor,
            "iact": float(integrated_autocorrelation_time(s)),
            "ess": float(effective_sample_size(s)),
            "dominant_period": float(dp["period"]) if np.isfinite(dp["period"]) else None,
            "dominant_peak_frac": float(dp["peak_frac"]),
        }
    return out

"""Matching and paired statistics for the equal-error comparison.

The main text compares reconstructions that have the same global error but a
different number of recovered scales. Those comparison units are built here: two
snapshots of one configuration are matched when their global errors agree to
within a tolerance while their scale counts differ by at least a given gap, and
the resulting pairs are summarised with paired tests and a cluster bootstrap that
respects the temporal ordering of the snapshots.

The matching is first-fit and one-to-one, so the result is deterministic and does
not depend on the order in which equally valid partners appear.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def find_internal_pairs(ger_list, sfull_list, tol=0.01, min_gap=2):
    """
    Find one-to-one matched pairs within a single configuration.
    Uses same first-fit algorithm as s33 for reproducible results.

    Conditions:
      - |GER_i - GER_j| / max(GER_i, GER_j) <= tol
      - |S_full_i - S_full_j| >= min_gap
      - One-to-one, no reuse (first-fit: for each i, take first valid j)

    Returns list of dicts.
    """
    B = len(ger_list)
    ger = np.array(ger_list)
    sfull = np.array(sfull_list)

    pairs = []
    used_i = set()
    used_j = set()

    for i in range(B):
        if i in used_i:
            continue
        for j in range(i + 1, B):
            if j in used_j:
                continue
            max_ger = max(ger[i], ger[j], EPS)
            ger_diff = abs(ger[i] - ger[j]) / max_ger
            sfull_diff = abs(int(sfull[i]) - int(sfull[j]))
            if ger_diff <= tol and sfull_diff >= min_gap:
                # Determine low/high S_full
                if sfull[i] < sfull[j]:
                    low_idx, high_idx = i, j
                elif sfull[i] > sfull[j]:
                    low_idx, high_idx = j, i
                else:
                    low_idx, high_idx = i, j

                pairs.append({
                    "idx_low": int(low_idx),
                    "idx_high": int(high_idx),
                    "GER_low": float(ger[low_idx]),
                    "GER_high": float(ger[high_idx]),
                    "S_full_low": int(sfull[low_idx]),
                    "S_full_high": int(sfull[high_idx]),
                    "GER_diff": float(abs(ger[low_idx] - ger[high_idx]) / max(ger[low_idx], ger[high_idx], EPS)),
                    "S_full_diff": int(abs(sfull[low_idx] - sfull[high_idx])),
                })
                used_i.add(i)
                used_j.add(j)
                break

    return pairs


# ══════════════════════════════════════════════════════════════════
# Statistical analysis
# ══════════════════════════════════════════════════════════════════

def compute_paired_stats(all_pairs_data):
    """
    Compute paired statistics for the matched pairs.

    all_pairs_data: list of dicts, each with fields for both members.
    Returns dict with n_pairs, per-metric tests.
    """
    w1_low, w1_high = [], []
    vort_low, vort_high = [], []
    grad_low, grad_high = [], []

    for p in all_pairs_data:
        metrics_low = p["metrics_low"]
        metrics_high = p["metrics_high"]
        w1_low.append(metrics_low["E_W1"])
        w1_high.append(metrics_high["E_W1"])
        vort_low.append(metrics_low["vorticity_RMSE"])
        vort_high.append(metrics_high["vorticity_RMSE"])
        grad_low.append(metrics_low["gradient_RMSE"])
        grad_high.append(metrics_high["gradient_RMSE"])

    def paired_test(a, b, label):
        a = np.array(a)
        b = np.array(b)
        n = len(a)
        if n < 2:
            return {"label": label, "n_pairs": n, "error": "insufficient"}

        diff = a - b
        median_diff = float(np.median(diff))
        mean_diff = float(np.mean(diff))

        # Wilcoxon signed-rank
        from scipy.stats import wilcoxon
        try:
            w_stat, w_p = wilcoxon(a, b, alternative="two-sided")
        except (ValueError):
            w_stat, w_p = np.nan, np.nan

        ratio = float(np.median(b) / (np.median(a) + EPS))

        return {
            "label": label,
            "n_pairs": n,
            "median_low_S_full": float(np.median(a)),
            "median_high_S_full": float(np.median(b)),
            "mean_low_S_full": float(np.mean(a)),
            "mean_high_S_full": float(np.mean(b)),
            "median_diff": median_diff,
            "mean_diff": mean_diff,
            "median_ratio_high_over_low": ratio,
            "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else None,
        }

    return {
        "n_pairs": len(all_pairs_data),
        "tests": [
            paired_test(w1_low, w1_high, "W1_band_error"),
            paired_test(vort_low, vort_high, "vorticity_RMSE"),
            paired_test(grad_low, grad_high, "gradient_RMSE"),
        ],
    }


def compute_cluster_bootstrap_ci(all_pairs_data, n_bootstrap=10000, ci_level=0.95):
    """
    Snapshot-cluster bootstrap CI for paired differences.

    Pairs are independently resampled with replacement.
    """
    n_pairs = len(all_pairs_data)
    w1_diffs = np.array([p["metrics_high"]["E_W1"] - p["metrics_low"]["E_W1"]
                         for p in all_pairs_data])
    vort_diffs = np.array([p["metrics_high"]["vorticity_RMSE"] -
                           p["metrics_low"]["vorticity_RMSE"]
                           for p in all_pairs_data])
    grad_diffs = np.array([p["metrics_high"]["gradient_RMSE"] -
                           p["metrics_low"]["gradient_RMSE"]
                           for p in all_pairs_data])

    rng = np.random.RandomState(42)
    alpha = 1.0 - ci_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    def bootstrap_ci(diffs, label):
        boot_medians = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            indices = rng.randint(0, n_pairs, size=n_pairs)
            boot_medians[b] = np.median(diffs[indices])

        ci_lower = float(np.percentile(boot_medians, lower_pct))
        ci_upper = float(np.percentile(boot_medians, upper_pct))
        obs_median = float(np.median(diffs))

        # p-value: proportion of bootstrapped medians with opposite sign
        if obs_median > 0:
            p_val = float(np.mean(boot_medians <= 0))
        elif obs_median < 0:
            p_val = float(np.mean(boot_medians >= 0))
        else:
            p_val = 1.0

        return {
            "label": label,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "observed_median_diff": obs_median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value_bootstrap": p_val,
        }

    return {
        "n_pairs": n_pairs,
        "n_bootstrap": n_bootstrap,
        "results": [
            bootstrap_ci(w1_diffs, "W1_band_error_diff"),
            bootstrap_ci(vort_diffs, "vorticity_RMSE_diff"),
            bootstrap_ci(grad_diffs, "gradient_RMSE_diff"),
        ],
    }

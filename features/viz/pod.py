"""
POD visualization — spectrum, energy-error scatter, mode-scale plots.

Replaces: tools/plot_fig06_pod_energy_vs_error.py, tools/plot_mode_scale_energy.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from luna.core.constants import BANDS_CF
from features.viz.style import BAND_COLORS, set_paper_style, get_figsize


def plot_pod_spectrum(
    singular_values: np.ndarray,
    energy_ratio: np.ndarray,
    cumulative_energy: np.ndarray,
    *,
    rank: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot POD singular value spectrum and cumulative energy.

    Args:
        singular_values: Full singular value array.
        energy_ratio: Per-mode energy fraction.
        cumulative_energy: Cumulative energy ratio.
        rank: Mark the truncation rank with a vertical line.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if figsize is None:
        figsize = get_figsize("double")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    modes = np.arange(1, len(singular_values) + 1)

    # Left: singular values
    ax1.semilogy(modes, singular_values, "b-", linewidth=1)
    ax1.set_xlabel("Mode index")
    ax1.set_ylabel("Singular value σ_k")
    ax1.set_title("POD Spectrum")
    if rank is not None:
        ax1.axvline(rank, color="r", linestyle="--", alpha=0.5, label=f"rank={rank}")
        ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: cumulative energy
    ax2.plot(modes, cumulative_energy * 100, "g-", linewidth=1.5)
    ax2.set_xlabel("Mode index")
    ax2.set_ylabel("Cumulative Energy (%)")
    ax2.set_title("Cumulative Energy")
    thresholds = [90, 95, 99, 99.9]
    for t in thresholds:
        if t / 100 <= cumulative_energy[-1]:
            idx = np.searchsorted(cumulative_energy, t / 100)
            ax2.axhline(t, color="gray", linestyle=":", alpha=0.4)
            ax2.axvline(idx + 1, color="gray", linestyle=":", alpha=0.4)
            ax2.text(idx + 1, t, f"  r={idx+1}", fontsize=7, va="bottom")
    ax2.grid(alpha=0.3)

    return fig


def plot_pod_energy_vs_error(
    band_energies: dict[str, np.ndarray],
    band_errors: dict[str, np.ndarray],
    *,
    title: str = "POD Energy vs Prediction Error",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot per-mode POD energy fraction vs band-wise prediction error.

    Args:
        band_energies: {band: (n_modes,) energy fractions}.
        band_errors: {band: (n_modes,) prediction errors}.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with Spearman correlation annotations.
    """
    if figsize is None:
        figsize = get_figsize("square")

    fig, ax = plt.subplots(figsize=figsize)

    for band in BANDS_CF:
        if band not in band_energies or band not in band_errors:
            continue
        e = band_energies[band]
        err = band_errors[band]

        ax.scatter(
            e, err,
            c=BAND_COLORS.get(band, "gray"),
            label=band,
            alpha=0.6,
            s=20,
            edgecolors="white",
            linewidth=0.3,
        )

        # Spearman correlation
        if len(e) > 2:
            r, p = scipy_stats.spearmanr(e, err)
            ax.text(
                0.95, 0.95 - BANDS_CF.index(band) * 0.06,
                f"{band}: ρ={r:.3f}" + ("*" if p < 0.05 else ""),
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
            )

    ax.set_xlabel("POD Mode Energy Fraction")
    ax.set_ylabel("Band Prediction Error")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    return fig


def plot_mode_scale_scatter(
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    energy_ratio: np.ndarray,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot POD mode physical scale scatter (ℓ_x vs ℓ_y).

    Args:
        ell_x: Per-mode x-direction scale.
        ell_y: Per-mode y-direction scale.
        energy_ratio: Per-mode energy fraction (for marker sizing).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if figsize is None:
        figsize = get_figsize("square")

    fig, ax = plt.subplots(figsize=figsize)

    sizes = 20 + 200 * energy_ratio / energy_ratio.max()
    scatter = ax.scatter(
        ell_x, ell_y,
        s=sizes,
        c=np.arange(len(ell_x)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="white",
        linewidth=0.3,
    )

    ax.set_xlabel("ℓ_x (pixels)")
    ax.set_ylabel("ℓ_y (pixels)")
    ax.set_title("POD Mode Physical Scales")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Mode Index")

    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    return fig

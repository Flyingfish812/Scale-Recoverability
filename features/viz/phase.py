"""
Recoverability phase diagram visualization.

Plots S_full as a function of mask number (M) and noise level (σ).

Replaces: tools/plot_recoverability_phase.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from luna.core.constants import STANDARD_MASK_NUMS, STANDARD_NOISE_SIGMAS
from features.viz.style import set_paper_style


# Discrete colormap for S_full (0–5)
_SFULL_CMAP = ListedColormap([
    "#f7f7f7",  # 0: none
    "#fee090",  # 1: A4 only
    "#fdae61",  # 2: A4+W4
    "#f46d43",  # 3: A4+W4+W3
    "#d73027",  # 4: A4+W4+W3+W2
    "#4575b4",  # 5: all
])


def plot_recoverability_phase(
    s_full_matrix: np.ndarray,
    *,
    mask_nums: list[int] | None = None,
    noise_sigmas: list[float] | None = None,
    title: str = "Scale Recoverability Phase Diagram",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    model_name: str = "",
) -> plt.Axes:
    """Plot S_full as a phase diagram: M (y-axis) × σ (x-axis).

    Args:
        s_full_matrix: (n_M, n_σ) array of S_full values (0–5).
        mask_nums: Sensor counts for y-axis ticks.
        noise_sigmas: Noise levels for x-axis ticks.
        title: Plot title.
        ax: Optional axes.
        figsize: Figure size.
        model_name: Model name for subtitle.

    Returns:
        Matplotlib Axes.
    """
    if mask_nums is None:
        mask_nums = STANDARD_MASK_NUMS
    if noise_sigmas is None:
        noise_sigmas = STANDARD_NOISE_SIGMAS

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    norm = BoundaryNorm(np.arange(-0.5, 6, 1), _SFULL_CMAP.N)
    im = ax.pcolormesh(
        range(len(noise_sigmas) + 1),
        range(len(mask_nums) + 1),
        s_full_matrix,
        cmap=_SFULL_CMAP,
        norm=norm,
        edgecolors="white",
        linewidth=1.5,
    )

    ax.set_xticks(np.arange(len(noise_sigmas)) + 0.5)
    ax.set_xticklabels([f"{s:.3f}" for s in noise_sigmas])
    ax.set_yticks(np.arange(len(mask_nums)) + 0.5)
    ax.set_yticklabels([str(m) for m in mask_nums])

    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Sensors M")
    ax.set_title(f"{title}\n{model_name}" if model_name else title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(6))
    cbar.set_label("S_full")
    cbar.set_ticklabels(["0", "1", "2", "3", "4", "5"])

    return ax


def plot_S_full_matrix(
    s_full_matrix: np.ndarray,
    *,
    mask_nums: list[int] | None = None,
    noise_sigmas: list[float] | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    vmin: float = 0.0,
    vmax: float = 5.0,
    cmap: str = "RdYlBu",
) -> plt.Figure:
    """Plot S_full as a continuous heatmap (smooth transitions).

    Args:
        s_full_matrix: (n_M, n_σ) array of mean S_full values.
        mask_nums, noise_sigmas: Tick labels.
        figsize: Figure size.
        vmin, vmax: Color scale range.
        cmap: Colormap name.

    Returns:
        Matplotlib Figure.
    """
    if mask_nums is None:
        mask_nums = STANDARD_MASK_NUMS
    if noise_sigmas is None:
        noise_sigmas = STANDARD_NOISE_SIGMAS

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        s_full_matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(noise_sigmas)))
    ax.set_xticklabels([f"{s:.3f}" for s in noise_sigmas])
    ax.set_yticks(range(len(mask_nums)))
    ax.set_yticklabels([str(m) for m in mask_nums])
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Sensors M")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean S_full")

    return fig

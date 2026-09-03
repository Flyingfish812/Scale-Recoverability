"""
Multi-model comparison plots.

Replaces: tools/plot_figures_from_aggregated.py, tools/plot_ger_vs_M_baseline.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from luna.core.constants import BANDS_CF, BANDS_FC
from features.viz.style import (
    METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS, METHOD_ORDER,
    BAND_COLORS_FC, set_paper_style, get_figsize,
)


def plot_model_comparison(
    errors: dict[str, dict[str, float]],
    *,
    title: str = "Model Comparison",
    figsize: tuple[float, float] | None = None,
    log_scale: bool = True,
) -> plt.Figure:
    """Plot per-band error comparison across all models.

    Args:
        errors: {model_name: {band: error}}.
        title: Figure title.
        figsize: Figure size.
        log_scale: Use log y-axis.

    Returns:
        Matplotlib Figure.
    """
    if figsize is None:
        figsize = get_figsize("double")

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(BANDS_FC))
    width = 0.15

    for i, method in enumerate(METHOD_ORDER):
        if method not in errors:
            continue
        values = [errors[method].get(b, np.nan) for b in BANDS_FC]
        offset = (i - len(METHOD_ORDER) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, f"C{i}"),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_FC)
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    return fig


def plot_ger_vs_M(
    ger_values: dict[str, list[float]],
    mask_nums: list[int],
    *,
    title: str = "Global Error vs Sensor Count",
    figsize: tuple[float, float] | None = None,
    add_1_over_sqrt_M: bool = True,
) -> plt.Figure:
    """Plot GER as a function of sensor count M, with 1/√M reference.

    Args:
        ger_values: {model_name: [GER at each M]}.
        mask_nums: Sensor counts for x-axis.
        title: Figure title.
        figsize: Figure size.
        add_1_over_sqrt_M: Add 1/√M reference curve.

    Returns:
        Matplotlib Figure.
    """
    if figsize is None:
        figsize = get_figsize("single")

    fig, ax = plt.subplots(figsize=figsize)

    M = np.array(mask_nums, dtype=np.float64)

    for method in METHOD_ORDER:
        if method not in ger_values:
            continue
        vals = np.array(ger_values[method])
        ax.plot(
            M, vals,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method, f"C{METHOD_ORDER.index(method)}"),
            label=METHOD_LABELS.get(method, method),
            linewidth=1.5,
            markersize=6,
        )

    if add_1_over_sqrt_M:
        ref = 1.0 / np.sqrt(M)
        ref *= ger_values.get("vcnn", [1.0])[0] / ref[0]
        ax.plot(M, ref, "k--", linewidth=0.8, alpha=0.5, label=r"$\propto 1/\sqrt{M}$")

    ax.set_xlabel("Number of Sensors M")
    ax.set_ylabel("Global Error Rate (GER)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    return fig

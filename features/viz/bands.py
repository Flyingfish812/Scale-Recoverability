"""
Band error visualization — bar charts, residual heatmaps.

Replaces: tools/plot_per_band_residual.py, parts of plot_figures_from_aggregated.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from luna.core.constants import BANDS_CF, BANDS_FC
from features.viz.style import BAND_COLORS, set_paper_style


def plot_band_error_bars(
    errors: dict[str, dict[str, float]],
    *,
    methods: list[str] | None = None,
    title: str = "Per-Band Relative L2 Error",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 4),
    log_scale: bool = False,
) -> plt.Axes:
    """Plot per-band error bar chart comparing multiple methods.

    Args:
        errors: {method: {band: error_value}}.
        methods: Order of methods (left to right groups).
        title: Plot title.
        ax: Optional existing axes.
        figsize: Figure size.
        log_scale: Use log scale for y-axis.

    Returns:
        Matplotlib Axes.
    """
    if methods is None:
        methods = list(errors.keys())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n_methods = len(methods)
    n_bands = len(BANDS_FC)
    width = 0.8 / n_methods
    x = np.arange(n_bands)

    for i, method in enumerate(methods):
        if method not in errors:
            continue
        values = [errors[method].get(b, np.nan) for b in BANDS_FC]
        bars = ax.bar(
            x + i * width - (n_methods - 1) * width / 2,
            values,
            width,
            label=method,
            color=BAND_COLORS.get(method, f"C{i}"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_FC)
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_per_band_residual_grid(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    components: dict[str, np.ndarray] | None = None,
    *,
    sample_idx: int = 0,
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Plot a grid showing per-band residuals for each model.

    Layout: rows = bands (A4..W1), columns = models + target.

    Args:
        target: (N, H, W) ground truth fields.
        predictions: {model_name: (N, H, W)} predictions.
        components: Pre-computed wavelet components for target. Computed if None.
        sample_idx: Which sample to plot.
        figsize: Figure size.
        cmap: Colormap for residuals.

    Returns:
        Matplotlib Figure.
    """
    from luna.wavelet.transform import decompose_field_2d

    n_models = len(predictions)
    n_bands = len(BANDS_CF)

    fig, axes = plt.subplots(
        n_bands, n_models + 1,
        figsize=figsize,
        constrained_layout=True,
    )
    if n_bands == 1:
        axes = axes[None, :]
    if n_models == 1:
        axes = axes[:, None]

    # Compute wavelet components for the target
    if components is None:
        components = decompose_field_2d(target[sample_idx])

    for i, band in enumerate(BANDS_CF):
        # Target band
        im0 = axes[i, 0].imshow(components[band], cmap="viridis", aspect="auto")
        axes[i, 0].set_ylabel(band, fontsize=9)
        if i == 0:
            axes[i, 0].set_title("Target", fontsize=8)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Model residuals
        for j, (name, preds) in enumerate(predictions.items()):
            pred_bands = decompose_field_2d(preds[sample_idx])
            residual = pred_bands[band] - components[band]

            vmax = max(abs(residual.min()), abs(residual.max()))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = axes[i, j + 1].imshow(residual, cmap=cmap, norm=norm, aspect="auto")
            axes[i, j + 1].set_xticks([])
            axes[i, j + 1].set_yticks([])
            if i == 0:
                axes[i, j + 1].set_title(name, fontsize=8)

    plt.colorbar(im, ax=axes, orientation="vertical", shrink=0.6, label="Residual")
    return fig

"""
Consistent visual style for all Luna figures.

Single source of truth for colors, fonts, and layout parameters.
Previously scattered across every plot_*.py file.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Method colors & markers ────────────────────────────────────────

METHOD_ORDER = ["pod_oracle", "gappy_pod", "ridge", "mlp", "vcnn"]

METHOD_LABELS = {
    "pod_oracle": "POD Oracle",
    "gappy_pod": "Gappy POD",
    "ridge": "Ridge",
    "mlp": "MLP",
    "vcnn": "VCNN",
    "voronoi_input": "Voronoi Input",
}

METHOD_COLORS = {
    "pod_oracle": "#2ecc71",   # green
    "gappy_pod": "#f39c12",    # orange
    "ridge": "#3498db",        # blue
    "mlp": "#9b59b6",          # purple
    "vcnn": "#e74c3c",         # red
    "voronoi_input": "#95a5a6",  # gray
}

METHOD_MARKERS = {
    "pod_oracle": "o",
    "gappy_pod": "s",
    "ridge": "D",
    "mlp": "^",
    "vcnn": "v",
}

# ── Band colors (coarse → fine, warm → cool) ──────────────────────

BAND_COLORS = {
    "A4": "#d62728",  # red (coarsest)
    "W4": "#ff7f0e",  # orange
    "W3": "#2ca02c",  # green
    "W2": "#1f77b4",  # blue
    "W1": "#9467bd",  # purple (finest)
}

BAND_COLORS_FC = {  # fine-to-coarse ordering
    "W1": "#9467bd",
    "W2": "#1f77b4",
    "W3": "#2ca02c",
    "W4": "#ff7f0e",
    "A4": "#d62728",
}

# ── Figure dimensions ──────────────────────────────────────────────

# Standard figure sizes (inches) for different layouts
FIGSIZE_SINGLE = (5.5, 4.0)      # single panel
FIGSIZE_DOUBLE = (7.0, 3.5)      # two panels side by side
FIGSIZE_WIDE = (10.0, 4.0)       # wide figure
FIGSIZE_SQUARE = (5.0, 5.0)      # square (heatmaps, phase diagrams)
FIGSIZE_FULLPAGE = (7.0, 8.5)    # full-page figure

# ── Typography ─────────────────────────────────────────────────────

PAPER_FONT_FAMILY = "serif"
PAPER_FONT_SIZE = 9
PAPER_TITLE_SIZE = 10
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7


def set_paper_style(
    font_size: int = PAPER_FONT_SIZE,
    font_family: str = PAPER_FONT_FAMILY,
    dpi: int = 150,
) -> None:
    """Apply consistent paper-ready matplotlib style."""
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": PAPER_TITLE_SIZE,
        "axes.labelsize": PAPER_LABEL_SIZE,
        "xtick.labelsize": PAPER_TICK_SIZE,
        "ytick.labelsize": PAPER_TICK_SIZE,
        "legend.fontsize": PAPER_TICK_SIZE,
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_figsize(
    layout: str = "single",
    scale: float = 1.0,
) -> tuple[float, float]:
    """Get a standard figure size.

    Args:
        layout: 'single', 'double', 'wide', 'square', 'fullpage'.
        scale: Multiplicative scale factor.

    Returns:
        (width, height) in inches.
    """
    sizes = {
        "single": FIGSIZE_SINGLE,
        "double": FIGSIZE_DOUBLE,
        "wide": FIGSIZE_WIDE,
        "square": FIGSIZE_SQUARE,
        "fullpage": FIGSIZE_FULLPAGE,
    }
    w, h = sizes.get(layout, FIGSIZE_SINGLE)
    return (w * scale, h * scale)

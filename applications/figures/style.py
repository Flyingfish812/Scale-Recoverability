#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publication_style.py — unified publication style for the manuscript figures
(v3-3 visual rework, 2026-08-31)

Design goals:
  1. one shared style for all Matplotlib data figures, not chosen per script;
  2. axis/legend font >= 8 pt, panel label 9-10 pt after final embedding;
  3. one font family (Times-like, matching TeX Gyre Termes in the manuscript);
  4. shared axis lines, line widths and markers;
  5. shared colours for MLP / VCNN / Ridge / Gappy / Oracle and bands A4..W1;
  6. every figure must stay readable in greyscale (Okabe-Ito palette,
     distinct luminance);
  7. vector PDF output only (lossless), no PNG;
  8. figures show data only; conclusions stay in the caption, titles short.

Usage:
    import style as ps
    ps.apply()                # call once before drawing
    fig = ps.figure(...)      # create a figure of the shared size
    ps.save(fig, "figXX_...") # write the vector PDF to the output directory
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global font: Times-like (Liberation Serif matches TeX Gyre Termes metrics)
# mathtext uses STIX (Times-style math)
# ---------------------------------------------------------------------------
FONT_FAMILY = "Liberation Serif"
MATHTEXT_FONT = "stix"

# Target: axis/legend text ~8 pt, panel label ~9.5 pt once embedded in the paper
# Figures are included 1:1 (width=\textwidth / \linewidth), so figsize is final
BASE_FONT = 8.0        # baseline for ticks and axis labels
LABEL_FONT = 9.5       # (a)(b) panel label
TITLE_FONT = 9.0       # panel titles (keep short, conclusions in the caption)


def apply() -> None:
    """Apply the shared manuscript figure style (call once before drawing)."""
    plt.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "Liberation Serif", "Nimbus Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": MATHTEXT_FONT,
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "axes.titlesize": TITLE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
        # axes: light grey, top/right spines removed (open journal frame)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # line widths / markers
        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
        # grid
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # legend
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        # output
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # embed TrueType so glyphs stay vector
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Colour maps — Okabe-Ito palette (colour-blind safe, greyscale separable)
# Models: blue(MLP) / purple(VCNN) / vermillion(Ridge) / green(Gappy) / black(Oracle)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "MLP":    "#0072B2",   # blue            (grey ~0.36)
    "VCNN":   "#CC79A7",   # reddish purple  (grey ~0.55)
    "Ridge":  "#D55E00",   # vermillion      (grey ~0.42)
    "Gappy":  "#009E73",   # bluish green    (grey ~0.57)
    "Oracle": "#000000",   # black           (grey ~0.00)
}
MODEL_MARKERS = {
    "MLP": "o",
    "VCNN": "s",
    "Ridge": "^",
    "Gappy": "D",
    "Oracle": "*",
}

# Bands: A4(coarsest) -> W1(finest), 5 colours with distinct luminance
BAND_COLORS = {
    "A4": "#1F3864",   # deep navy   (grey 0.16)
    "W4": "#2E75B6",   # blue        (0.35)
    "W3": "#5B9BD5",   # light blue  (0.52)
    "W2": "#A9C4E5",   # pale blue   (0.71)
    "W1": "#E7EEF7",   # near white  (0.91)
}
BAND_MARKERS = {"A4": "o", "W4": "s", "W3": "^", "W2": "D", "W1": "v"}

# Noise level σ: single-hue ramp (blue -> dark blue)
SIGMA_COLORS = {0.0: "#56B4E9", 0.001: "#3D8FC4", 0.01: "#2C6FA3", 0.1: "#1B4F82"}
SIGMA_MARKERS = {0.0: "o", 0.001: "s", 0.01: "^", 0.1: "D"}

# Threshold / accents
TAU_COLOR = "#C00000"      # threshold line (red)
ACCENT = "#C00000"         # accent (red circle / first-fail marker)
GRAY = "#777777"

BANDS = ["A4", "W4", "W3", "W2", "W1"]
TAU = 0.05


# ---------------------------------------------------------------------------
# Figure size conventions
#   full width (figure*):  6.6 in   (final width=\textwidth)
#   half width (figure):   3.25 in  (final width=\linewidth)
# height follows the content, 2.5-5.2 in
# ---------------------------------------------------------------------------
def figure(width_in: float, height_in: float, nrows: int = 1, ncols: int = 1,
           **kwargs):
    """Create a figure of the shared size and return (fig, ax).

    With nrows = ncols = 1 the returned ax is a single Axes.
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=(width_in, height_in), **kwargs)
    if nrows == 1 and ncols == 1:
        return fig, ax
    return fig, ax


def panel_label(ax, letter: str, x: float = -0.20, y: float = 1.02) -> None:
    """Shared (a)/(b)/(c) panel label style: 9.5 pt, bold, top left."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=LABEL_FONT, fontweight="bold", va="bottom", ha="left")


def style_axis(ax, xlabel: str | None = None, ylabel: str | None = None,
               legend: bool = False, legend_kw: dict | None = None,
               grid_axis: str = "both") -> None:
    """Apply the shared axis style."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis=grid_axis, which="major")
    if legend:
        kw = {"frameon": False, "fontsize": BASE_FONT, "loc": "best"}
        if legend_kw:
            kw.update(legend_kw)
        ax.legend(**kw)


def save(fig, out_dir, stem: str, dpi: int = 300) -> None:
    """Write the figure as the vector PDF that the manuscript includes.

    The creation timestamp is omitted so that the file is deterministic: a
    regenerated figure can then be compared with the published one directly,
    and no diff appears when the drawing has not changed.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=dpi,
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  [pdf] {path}")


def close(fig) -> None:
    plt.close(fig)

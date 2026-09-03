"""
Visualization layer — reusable plotting components & consistent styling.

Provides:
    - style: Color schemes, font settings, figure dimensions
    - bands: Band error bar charts, residual heatmaps
    - phase: Recoverability phase diagrams
    - comparison: Multi-model comparison plots
    - pod: POD energy spectrum & mode-scale scatter plots
"""

from features.viz.style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_ORDER,
    BAND_COLORS,
    set_paper_style,
    get_figsize,
)
from features.viz.bands import (
    plot_band_error_bars,
    plot_per_band_residual_grid,
)
from features.viz.phase import (
    plot_recoverability_phase,
    plot_S_full_matrix,
)
from features.viz.comparison import (
    plot_model_comparison,
    plot_ger_vs_M,
)
from features.viz.pod import (
    plot_pod_spectrum,
    plot_pod_energy_vs_error,
    plot_mode_scale_scatter,
)

__all__ = [
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_MARKERS",
    "METHOD_ORDER",
    "BAND_COLORS",
    "set_paper_style",
    "get_figsize",
    "plot_band_error_bars",
    "plot_per_band_residual_grid",
    "plot_recoverability_phase",
    "plot_S_full_matrix",
    "plot_model_comparison",
    "plot_ger_vs_M",
    "plot_pod_spectrum",
    "plot_pod_energy_vs_error",
    "plot_mode_scale_scatter",
]

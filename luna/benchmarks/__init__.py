"""
Analytical benchmarks for metric validation.

  analytical_wake — NC-inspired analytical multiscale wake field with
                    known ground-truth scale content (P0-1).
"""
from luna.benchmarks.analytical_wake import (
    WakeParams,
    WAVENUM,
    base_streamfunction,
    wake_envelope,
    scale_component,
    streamfunction,
    velocity,
    scale_u_components,
    snapshot,
    generate_ensemble,
    controlled_reconstructions,
    case_metrics,
)

__all__ = [
    "WakeParams",
    "WAVENUM",
    "base_streamfunction",
    "wake_envelope",
    "scale_component",
    "streamfunction",
    "velocity",
    "scale_u_components",
    "snapshot",
    "generate_ensemble",
    "controlled_reconstructions",
    "case_metrics",
]

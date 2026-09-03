"""
Analysis modules for scale-resolved field reconstruction evaluation.

Provides:
    - oracle: POD oracle computation & comparison
    - baselines: Gappy POD & POD-LS baseline methods
    - sensitivity: Hyperparameter sensitivity analysis
    - controlled: Controlled scale validation experiments
    - audit: Data consistency & wording audits
"""

from features.analysis.oracle import (
    compute_oracle_comparison,
    compute_oracle_band_errors,
)
from features.analysis.baselines import (
    run_gappy_pod,
    run_pod_ls,
    compute_baseline_comparison,
)
from features.analysis.sensitivity import (
    compute_sensitivity_sweep,
)
from features.analysis.controlled import (
    run_controlled_scale_validation,
)
from features.analysis.audit import (
    run_digital_audit,
    run_wording_audit,
)

__all__ = [
    "compute_oracle_comparison",
    "compute_oracle_band_errors",
    "run_gappy_pod",
    "run_pod_ls",
    "compute_baseline_comparison",
    "compute_sensitivity_sweep",
    "run_controlled_scale_validation",
    "run_digital_audit",
    "run_wording_audit",
]

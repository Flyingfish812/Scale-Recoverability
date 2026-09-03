"""
Training modules for POD-coefficient prediction models (supplementary).

Provides:
    - train_pod_model: Ridge/MLP training loop (P0, luna-based)
    - run_pod_model_sweep: sweep skeleton (P0)
    - run_mlp_case / run_ridge_closed_form_case / run_gappy_case: supplementary
      per-(family, M, seed) runners producing test_raw.npz
    - PODObservationDataset / split_indices / save_test_raw: shared protocol

Note: VCNN training wrappers (features.training.trainer / .sweep) depended on
the legacy top-level `training/` package which is archived in _legacy; they are
NOT exported here. Supplementary VCNN uses the existing P0 family_01 artifacts and
marks family_02/03 training as a follow-up.
"""

from features.training.pod_trainer import (
    train_pod_model,
    run_pod_model_sweep,
)
from features.training.pod_sweep import (
    PODObservationDataset,
    split_indices,
    save_test_raw,
    run_mlp_case,
    run_ridge_closed_form_case,
    run_gappy_case,
    compute_channel_mean_std,
)

__all__ = [
    "train_pod_model",
    "run_pod_model_sweep",
    "PODObservationDataset",
    "split_indices",
    "save_test_raw",
    "run_mlp_case",
    "run_ridge_closed_form_case",
    "run_gappy_case",
    "compute_channel_mean_std",
]

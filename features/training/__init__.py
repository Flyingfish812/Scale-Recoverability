"""Training of the estimators compared in the paper.

POD-coefficient estimators (linear / MLP / gappy) operate on a rank-r POD representation of the field; the convolutional estimator maps the sparse observation grid directly to the field.

Modules
    pod_sweep.py    per-configuration runners for the POD-coefficient estimators;
                    each writes test_raw.npz (target and reconstruction) that the statistics stage consumes
    pod_trainer.py  generic training loop shared by the POD-coefficient models
    vcnn_sweep.py   sensor-mask sweep for the convolutional estimator
    vcnn_trainer.py training loop of the convolutional estimator vcnn_config.py  configuration dataclasses for the sweep above
"""

from features.training.pod_sweep import (
    PODObservationDataset,
    compute_channel_mean_std,
    run_gappy_case,
    run_mlp_case,
    run_ridge_closed_form_case,
    save_test_raw,
    split_indices,
)
from features.training.pod_trainer import (
    run_pod_model_sweep,
    train_pod_model,
)
from features.training.vcnn_config import (
    CheckpointConfig,
    DataSourceConfig,
    MaskConfig,
    SweepConfig,
    TrainConfig,
    VcnnConfig,
)
from features.training.vcnn_sweep import run_vcnn_sweep
from features.training.vcnn_trainer import train_vcnn

__all__ = [
    # POD-coefficient estimators
    "PODObservationDataset",
    "compute_channel_mean_std",
    "run_gappy_case",
    "run_mlp_case",
    "run_ridge_closed_form_case",
    "run_pod_model_sweep",
    "save_test_raw",
    "split_indices",
    "train_pod_model",
    # convolutional estimator
    "CheckpointConfig",
    "DataSourceConfig",
    "MaskConfig",
    "SweepConfig",
    "TrainConfig",
    "VcnnConfig",
    "run_vcnn_sweep",
    "train_vcnn",
]

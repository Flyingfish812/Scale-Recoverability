"""Configuration objects for the convolutional estimator and its sweeps.

Dataclasses only: data source, sensor mask, network, training and checkpoint settings. The values that define the paper's runs are set by the pipeline step ``applications/pipelines/03_train_estimators.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class DataSourceConfig:
    array_path: Path
    mmap: bool = True


@dataclass
class MaskConfig:
    mode: str = "random"
    include_mask_channel: bool = True
    seed: int | None = 0
    mask_rate: float | None = None
    mask_num: int | None = None
    mask_path: Path | None = None


@dataclass
class VcnnConfig:
    hidden_channels: int = 48
    num_layers: int = 8
    kernel_size: int = 7
    input_representation: str = "voronoi"
    include_mask_channel: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 40
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.0
    min_lr: float = 0.0
    warmup_epochs: int = 0
    use_cosine_schedule: bool = True
    early_stop: bool = True
    early_patience: int = 10
    early_min_delta: float = 0.0
    early_warmup: int = 5
    device: str | None = "auto"
    seed: int | None = 0
    normalize_mean_std: bool = True
    loss_type: str = "mse"
    obs_weight: float = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    progress_every: int = 1


@dataclass
class CheckpointConfig:
    out_dir: Path
    save_best_only: bool = True
    save_last: bool = True
    prefix: str = "vcnn"
    save_epochs: Sequence[int] = field(default_factory=tuple)


@dataclass
class SweepConfig:
    mask_rates: Sequence[float] = field(default_factory=tuple)
    mask_nums: Sequence[int] = field(default_factory=tuple)
    noise_sigmas: Sequence[float] = field(default_factory=tuple)
    train_noise_sigma: float = 0.0
    test_noise_sigmas: Sequence[float] = field(default_factory=lambda: (0.0, 1e-3, 1e-2, 1e-1))
    mask_seeds: Sequence[int] = field(default_factory=lambda: (0,))
    mask_paths: Sequence[Path] = field(default_factory=tuple)

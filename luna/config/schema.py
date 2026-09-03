"""
Unified configuration schema for all Luna experiments.

Consolidates config dataclasses from:
    training/config.py  — VCNN training + sweep
    l1_pod/config.py    — L1 POD parameters
    run_pod_model_sweep.py — POD model training
    thesis_figure_pipeline configs — figure generation

All paths in configs are relative to the project root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ── Data source ────────────────────────────────────────────────────

@dataclass
class DataSourceConfig:
    """Dataset file and loading parameters."""
    array_path: Path | str = ""
    mmap: bool = True


# ── Mask ───────────────────────────────────────────────────────────

@dataclass
class MaskConfig:
    """Sensor mask configuration."""
    mode: str = "random"             # "random", "grid", "csv"
    include_mask_channel: bool = True
    seed: int | None = 0
    mask_rate: float | None = None
    mask_num: int | None = None
    mask_path: Path | str | None = None
    mask_dir: Path | str | None = None  # directory containing mask CSVs
    mask_names: Sequence[str] = field(default_factory=tuple)
    mask_csvs: Sequence[str] = field(default_factory=tuple)


# ── VCNN model ─────────────────────────────────────────────────────

@dataclass
class VcnnModelConfig:
    """VCNN architecture hyperparameters."""
    hidden_channels: int = 48
    num_layers: int = 8
    kernel_size: int = 7
    input_representation: str = "voronoi"
    include_mask_channel: bool = True


# ── Training ───────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training loop configuration."""
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
    device: str = "auto"
    seed: int | None = 0
    normalize_mean_std: bool = True
    loss_type: str = "mse"
    obs_weight: float = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    progress_every: int = 1


@dataclass
class CheckpointConfig:
    """Model checkpoint save configuration."""
    out_dir: Path | str = "artifacts/vcnn_sweep"
    save_best_only: bool = True
    save_last: bool = True
    prefix: str = "vcnn"
    save_epochs: Sequence[int] = field(default_factory=tuple)


# ── Sweep ──────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """Parameter sweep configuration."""
    mask_rates: Sequence[float] = field(default_factory=tuple)
    mask_nums: Sequence[int] = field(default_factory=tuple)
    noise_sigmas: Sequence[float] = field(default_factory=tuple)
    train_noise_sigma: float = 0.0
    test_noise_sigmas: Sequence[float] = field(default_factory=lambda: (0.0, 1e-3, 1e-2, 1e-1))
    mask_seeds: Sequence[int] = field(default_factory=lambda: (0,))
    mask_paths: Sequence[Path] = field(default_factory=tuple)


# ── POD model ──────────────────────────────────────────────────────

@dataclass
class PodModelConfig:
    """Configuration for Ridge/MLP POD-coefficient prediction models."""
    data_array: str = ""
    pod_bundle_path: str = ""
    out_dir: str = "artifacts/pod_model_sweep"
    mask_dir: str = "masks2"
    mask_names: Sequence[str] = field(default_factory=tuple)
    mask_nums: Sequence[int] = field(default_factory=tuple)
    noise_sigmas: Sequence[float] = field(default_factory=lambda: (0.0, 0.001, 0.01, 0.1))
    n_modes: int = 128
    # Ridge
    ridge_alpha: float = 1.0
    # MLP
    mlp_hidden_sizes: Sequence[int] = (256, 256)
    mlp_dropout: float = 0.0
    # Training
    batch_size: int = 32
    num_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    seed: int = 42


# ── Wavelet ────────────────────────────────────────────────────────

@dataclass
class WaveletConfig:
    """Wavelet transform parameters."""
    name: str = "db2"
    level: int = 4
    mode: str = "periodization"


# ── Thesis figure pipeline ─────────────────────────────────────────

@dataclass
class ThesisFigureConfig:
    """Configuration for the thesis figure generation pipeline."""
    root: str = "."
    output_dir: str = "artifacts/thesis_figures"
    cache_dir: str = "artifacts/thesis_figures/cache"
    paths: dict[str, str] = field(default_factory=dict)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    threshold: float = 0.05
    representative: dict[str, Any] = field(default_factory=dict)
    pod_debug_mode_index_1based: int = 48
    wavelet_pod_band_name: str = "W3"
    wavelet_pod_proj_rank: int = 15
    main_figures: list[int] = field(default_factory=list)
    appendix_figures: list[str] = field(default_factory=list)


# ── Top-level experiment config ────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs."""
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    vcnn: VcnnModelConfig = field(default_factory=VcnnModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    extra: dict[str, Any] = field(default_factory=dict)

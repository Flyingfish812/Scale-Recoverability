"""
Configuration loader — reads JSON/TOML config files and produces typed config objects.

Supports both legacy JSON configs and new TOML configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from luna.config.schema import (
    ExperimentConfig,
    DataSourceConfig,
    MaskConfig,
    VcnnModelConfig,
    TrainConfig,
    CheckpointConfig,
    SweepConfig,
    WaveletConfig,
)
from luna.core.types import DatasetInfo
from luna.data.io import resolve_project_path


def _try_import_toml() -> bool:
    try:
        import tomllib  # Python 3.11+
        return True
    except ImportError:
        try:
            import tomli  # third-party backport
            return True
    except ImportError:
        return False


_HAS_TOML = _try_import_toml()


# ── JSON loader (legacy compat) ────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── TOML loader ────────────────────────────────────────────────────

def _load_toml(path: Path) -> dict[str, Any]:
    if not _HAS_TOML:
        raise ImportError(
            "TOML config requires Python 3.11+ (tomllib) or `pip install tomli`. "
            "Use JSON configs as a fallback."
        )
    with open(path, "rb") as f:
        try:
            import tomllib
            return tomllib.load(f)
        except ImportError:
            import tomli
            return tomli.load(f)


# ── Generic config loader ──────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file (JSON or TOML), auto-detecting format by extension."""
    p = resolve_project_path(path)
    if p.suffix in (".toml", ".tml"):
        return _load_toml(p)
    return _load_json(p)


# ── Typed config builders ──────────────────────────────────────────

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a VCNN sweep experiment config (JSON or TOML) into a typed object."""
    raw = load_config(path)

    # Data
    data = DataSourceConfig(
        array_path=raw.get("array_path", ""),
        mmap=raw.get("mmap", True),
    )

    # Mask
    mask = MaskConfig(
        mode=raw.get("mask_mode", "random"),
        mask_rate=raw.get("mask_rate"),
        mask_num=raw.get("mask_num"),
        mask_names=tuple(raw.get("mask_names", [])),
        mask_csvs=tuple(raw.get("mask_csvs", [])),
        mask_dir=raw.get("mask_dir", "masks2"),
        seed=raw.get("seed", 0),
    )

    # VCNN
    vcnn = VcnnModelConfig(
        hidden_channels=raw.get("hidden_channels", 48),
        num_layers=raw.get("num_layers", 8),
        kernel_size=raw.get("kernel_size", 7),
        input_representation=raw.get("input_representation", "voronoi"),
        include_mask_channel=raw.get("include_mask_channel", True),
    )

    # Training
    train = TrainConfig(
        batch_size=raw.get("batch_size", 16),
        num_epochs=raw.get("num_epochs", 40),
        val_ratio=raw.get("val_ratio", 0.1),
        test_ratio=raw.get("test_ratio", 0.2),
        lr=raw.get("lr", 1e-3),
        weight_decay=raw.get("weight_decay", 0.0),
        min_lr=raw.get("min_lr", 0.0),
        warmup_epochs=raw.get("warmup_epochs", 0),
        use_cosine_schedule=raw.get("use_cosine_schedule", True),
        early_stop=raw.get("early_stop", True),
        early_patience=raw.get("early_patience", 10),
        early_min_delta=raw.get("early_min_delta", 0.0),
        early_warmup=raw.get("early_warmup", 5),
        device=raw.get("device", "auto"),
        seed=raw.get("seed", 0),
        normalize_mean_std=raw.get("normalize_mean_std", True),
        loss_type=raw.get("loss_type", "mse"),
        obs_weight=raw.get("obs_weight", 1.0),
        progress_every=raw.get("progress_every", 1),
    )

    # Checkpoint
    checkpoint = CheckpointConfig(
        out_dir=raw.get("out_dir", "artifacts/vcnn_sweep"),
        save_best_only=raw.get("save_best_only", True),
        save_last=raw.get("save_last", True),
        prefix=raw.get("prefix", "vcnn"),
    )

    # Sweep
    sweep = SweepConfig(
        mask_rates=tuple(raw.get("mask_rates", [])),
        mask_nums=tuple(raw.get("mask_nums", [])),
        noise_sigmas=tuple(raw.get("noise_sigmas", [])),
        train_noise_sigma=raw.get("train_noise_sigma", 0.0),
        test_noise_sigmas=tuple(raw.get("test_noise_sigmas", [0.0, 0.001, 0.01, 0.1])),
        mask_seeds=tuple(raw.get("random_seeds", [0])),
    )

    # Wavelet
    wavelet = WaveletConfig(
        name=raw.get("wavelet", "db2"),
        level=raw.get("wavelet_level", 4),
        mode=raw.get("wavelet_mode", "periodization"),
    )

    return ExperimentConfig(
        data=data,
        mask=mask,
        vcnn=vcnn,
        train=train,
        checkpoint=checkpoint,
        sweep=sweep,
        wavelet=wavelet,
        extra={k: v for k, v in raw.items() if k not in {
            "array_path", "mmap", "mask_mode", "mask_rate", "mask_num",
            "mask_names", "mask_csvs", "mask_dir", "seed",
            "hidden_channels", "num_layers", "kernel_size",
            "input_representation", "include_mask_channel",
            "batch_size", "num_epochs", "val_ratio", "test_ratio",
            "lr", "weight_decay", "min_lr", "warmup_epochs",
            "use_cosine_schedule", "early_stop", "early_patience",
            "early_min_delta", "early_warmup", "device",
            "normalize_mean_std", "loss_type", "obs_weight", "progress_every",
            "out_dir", "save_best_only", "save_last", "prefix",
            "mask_rates", "mask_nums", "noise_sigmas",
            "train_noise_sigma", "test_noise_sigmas", "random_seeds",
            "wavelet", "wavelet_level", "wavelet_mode",
        }},
    )


def load_dataset_config(dataset_name: str) -> DatasetInfo:
    """Load a dataset config from configs/datasets/{name}.toml or .json."""
    from luna.data.registry import get_dataset

    # First try the built-in registry
    try:
        return get_dataset(dataset_name)
    except KeyError:
        pass

    # Try loading from a config file
    for ext in (".toml", ".json"):
        config_path = resolve_project_path(f"configs/datasets/{dataset_name}{ext}")
        if config_path.exists():
            raw = load_config(config_path)
            return DatasetInfo(
                name=raw.get("name", dataset_name),
                label=raw.get("label", dataset_name),
                data_array=resolve_project_path(raw["data_array"]),
                vcnn_roots={
                    k: resolve_project_path(v)
                    for k, v in raw.get("vcnn_roots", {}).items()
                },
                pod_sweep_root=resolve_project_path(raw["pod_sweep_root"]) if "pod_sweep_root" in raw else None,
                pod_bundle_path=resolve_project_path(raw["pod_bundle_path"]) if "pod_bundle_path" in raw else None,
                band_pod_bundle_path=resolve_project_path(raw["band_pod_bundle_path"]) if "band_pod_bundle_path" in raw else None,
                mask_pattern=raw.get("mask_pattern", ""),
                mask_dir=resolve_project_path(raw["mask_dir"]) if "mask_dir" in raw else None,
                extra=raw.get("extra", {}),
            )

    raise FileNotFoundError(f"No config found for dataset '{dataset_name}'")

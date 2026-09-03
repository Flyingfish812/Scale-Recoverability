"""
Dataset registry — single source of truth for dataset paths & metadata.

Consolidates the duplicated dataset config from:
    tools/dataset_config.py
    tools/oracle_and_baseline_comparison.py
    tools/run_baselines_efficient.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luna.core.types import DatasetInfo
from luna.data.io import resolve_project_path


class DatasetRegistry:
    """Central registry of all known datasets and their artifact paths."""

    _datasets: dict[str, DatasetInfo] = {}

    @classmethod
    def register(cls, info: DatasetInfo) -> None:
        cls._datasets[info.name] = info

    @classmethod
    def get(cls, name: str) -> DatasetInfo:
        if name not in cls._datasets:
            raise KeyError(
                f"Unknown dataset '{name}'. Available: {list(cls._datasets.keys())}"
            )
        return cls._datasets[name]

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._datasets.keys())


def register_dataset(info: DatasetInfo) -> None:
    DatasetRegistry.register(info)


def get_dataset(name: str) -> DatasetInfo:
    return DatasetRegistry.get(name)


def list_datasets() -> list[str]:
    return DatasetRegistry.list_all()


# ── Built-in dataset definitions ───────────────────────────────────

def _p(path: str) -> Path:
    return resolve_project_path(path)


# NC: cylinder2d_q1 (numerical cylinder wake)
register_dataset(DatasetInfo(
    name="nc",
    label="cylinder2d_q1 (NC)",
    data_array=_p("data/cylinder2d_q1.npy"),
    vcnn_roots={
        "vcnn_seed000": _p("artifacts/vcnn_results/vcnn_sweep_nc_2000"),
        "vcnn_seed101": _p("artifacts/vcnn_results/vcnn_sweep_nc_2000_seed101"),
        "vcnn_seed202": _p("artifacts/vcnn_results/vcnn_sweep_nc_2000_seed202"),
    },
    pod_sweep_root=_p("artifacts/pod_model_sweep_nc"),
    pod_bundle_path=_p("artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"),
    band_pod_bundle_path=_p("artifacts/wavelet_pod_nc_2000/band_pod_bundle.npz"),
    mask_pattern="cylinder2d_80x160_random_inc_n{num:03d}.csv",
    mask_dir=_p("masks2"),
))

# RDB: rdb_h5 (turbulent channel flow)
register_dataset(DatasetInfo(
    name="rdb_h5",
    label="rdb_h5 (B)",
    data_array=_p("data/rdb_h5.npy"),
    vcnn_roots={
        "vcnn_seed000": _p("artifacts/vcnn_results/vcnn_sweep_rdb_2000_seed000"),
    },
    pod_sweep_root=_p("artifacts/pod_model_sweep_h5"),
    pod_bundle_path=_p("artifacts/pod_bases/rdb_h5/pod_base_bundle.npz"),
    band_pod_bundle_path=_p("artifacts/wavelet_pod_rdb_2000/band_pod_bundle.npz"),
    mask_pattern="rdb_h5_128x128_radial_inc_n{num:03d}.csv",
    mask_dir=_p("masks2"),
))

# SST: sst_weekly (sea surface temperature)
register_dataset(DatasetInfo(
    name="sst_weekly",
    label="sst_weekly (SST)",
    data_array=_p("data/sst_weekly.npy"),
    vcnn_roots={},
    pod_sweep_root=_p("artifacts/pod_model_sweep_sst_1024"),
    pod_bundle_path=_p("artifacts/pod_bases/sst_weekly/pod_base_bundle.npz"),
    band_pod_bundle_path=_p("artifacts/wavelet_pod_sst_2000/band_pod_bundle.npz"),
    mask_pattern="sst_weekly_180x360_random_avoid_nan_inc_n{num:03d}.csv",
    mask_dir=_p("masks2"),
))

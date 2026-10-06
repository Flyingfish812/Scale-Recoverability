"""Application-level configuration.

Single place where the paper's experimental parameters live: sensor counts, noise levels, training seeds, POD rank, wavelet settings and tolerance. Values come from ``applications/configs/*.yaml`` when present and fall back to the defaults below, which are the ones used for the paper.

The pipeline steps and the statistics producers read parameters from here so that changing, for instance, the tolerance requires editing one YAML file rather than individual scripts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "applications"
CONFIG_DIR = APP / "configs"

_DEFAULTS = {
    "dataset_nc": {
        "dataset": {"name": "nc", "grid_h": 80, "grid_w": 160, "test_snapshots": 300},
        "sensors": {"counts": [10, 15, 20, 30, 50]},
        "noise": {"test_sigmas": [0.0, 0.001, 0.01, 0.1]},
        "pod": {"main_rank": 128, "audit_ranks": [16, 32, 64, 128]},
        "wavelet": {"family": "db2", "level": 4, "boundary_mode": "periodization"},
        "models": {"mlp": {"seeds": [0, 101, 202]}, "vcnn": {"seeds": [0, 101, 202]}},
        "mask_families": ["family_01", "family_02", "family_03", "family_04", "family_05"],
    },
    "metrics": {
        "bands": ["A4", "W4", "W3", "W2", "W1"],
        "tau": 0.05,
        "eta": 0.99,
        "robust_threshold": 0.01,
        "denominator_audit": {"eps_abs": 1e-8, "eps_rel": 1e-6},
    },
    "statistics": {
        "block_bootstrap": {
            "method": "moving_block",
            "n_resamples": 10000,
            "seed": 20260806,
            "candidate_block_lengths": [10, 20, 30, 50, 62, 125],
            "prefer_physical_period": True,
        },
        "snapshot_cluster": {"n_resamples": 10000, "seed": 20260806},
    },
}


class Config:
    """Typed access to the application configuration."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = data or _load()

    # ── experiment definition ────────────────────────────────────────
    @property
    def bands(self) -> list[str]:
        return list(self._data["metrics"]["bands"])

    @property
    def tau(self) -> float:
        return float(self._data["metrics"]["tau"])

    @property
    def eta(self) -> float:
        return float(self._data["metrics"]["eta"])

    @property
    def M_values(self) -> list[int]:
        return list(self._data["dataset_nc"]["sensors"]["counts"])

    @property
    def sigma_values(self) -> list[float]:
        return list(self._data["dataset_nc"]["noise"]["test_sigmas"])

    @property
    def seeds(self) -> list[int]:
        return list(self._data["dataset_nc"]["models"]["mlp"]["seeds"])

    @property
    def mlp_seeds(self) -> list[int]:
        return list(self._data["dataset_nc"]["models"]["mlp"]["seeds"])

    @property
    def vcnn_seeds(self) -> list[int]:
        return list(self._data["dataset_nc"]["models"]["vcnn"]["seeds"])

    @property
    def mask_families(self) -> list[str]:
        return list(self._data["dataset_nc"]["mask_families"])

    @property
    def pod_rank(self) -> int:
        return int(self._data["dataset_nc"]["pod"]["main_rank"])

    @property
    def wavelet_family(self) -> str:
        return self._data["dataset_nc"]["wavelet"]["family"]

    @property
    def wavelet_level(self) -> int:
        return int(self._data["dataset_nc"]["wavelet"]["level"])

    @property
    def wavelet_mode(self) -> str:
        return self._data["dataset_nc"]["wavelet"]["boundary_mode"]

    # ── statistics ───────────────────────────────────────────────────
    @property
    def eps_abs(self) -> float:
        return float(self._data["metrics"]["denominator_audit"]["eps_abs"])

    @property
    def eps_rel(self) -> float:
        return float(self._data["metrics"]["denominator_audit"]["eps_rel"])

    @property
    def block_bootstrap(self) -> dict:
        return dict(self._data["statistics"]["block_bootstrap"])

    @property
    def snapshot_cluster(self) -> dict:
        return dict(self._data["statistics"]["snapshot_cluster"])

    # ── output locations ─────────────────────────────────────────────
    @property
    def artifacts_root(self) -> Path:
        return ROOT / "artifacts"

    @property
    def build_dir(self) -> Path:
        return ROOT / "artifacts" / "build"

    @property
    def figures_out(self) -> Path:
        return ROOT / "artifacts" / "figures"

    @property
    def tables_out(self) -> Path:
        return ROOT / "artifacts" / "tables"


def _load() -> dict:
    """Defaults, overridden by any YAML file in applications/configs/."""
    data = copy_defaults()
    for path in sorted(CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else []:
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merge(data, override)
    return data


def copy_defaults() -> dict:
    import copy

    return copy.deepcopy(_DEFAULTS)


def merge(base: dict, override: dict) -> None:
    """Recursively merge a configuration override into the defaults."""
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            merge(base[section], values)
        else:
            base[section] = values



_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG

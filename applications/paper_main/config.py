"""paper_main.config — 应用层统一配置加载 (P0-5)。

从 applications/paper_main/configs/*.yaml 加载, 提供默认值与便捷访问。
原则: 应用层入口不硬编码实验参数; 改 tau / boundary mode / M / σ / seeds
等只需改 configs/*.yaml。
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "applications" / "paper_main"
CONFIG_DIR = APP / "configs"

_DEFAULTS = {
    "metrics": {
        "bands": ["A4", "W4", "W3", "W2", "W1"],
        "tau": 0.05,
        "eta": 0.99,
        "robust_threshold": 0.01,
    },
    "dataset_nc": {
        "dataset": {"name": "nc", "grid_h": 80, "grid_w": 160, "test_snapshots": 300},
        "sensors": {"counts": [10, 15, 20, 30, 50]},
        "noise": {"test_sigmas": [0.0, 0.001, 0.01, 0.1]},
        "pod": {"main_rank": 128, "oracle_ranks": [16, 32, 64, 128]},
        "wavelet": {"family": "db2", "level": 4, "boundary_mode": "periodization"},
        "models": {
            "mlp": {"seeds": [0, 101, 202]},
            "vcnn": {"seeds": [0, 101, 202]},
        },
    },
    "statistics": {
        "bootstrap": {"type": "snapshot_cluster", "n_resamples": 10000, "seed": 20260804},
    },
}


def _load(name: str) -> dict:
    p = CONFIG_DIR / f"{name}.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


class Config:
    """合并默认值 + yaml 配置。"""

    def __init__(self) -> None:
        self.metrics = {**_DEFAULTS["metrics"], **(_load("metrics") or {})}
        self.dataset_nc = {**_DEFAULTS["dataset_nc"], **(_load("dataset_nc") or {})}
        self.statistics = {**_DEFAULTS["statistics"], **(_load("statistics") or {})}
        self.figures_tables = _load("figures_tables")

    # ── 便捷属性 ──────────────────────────────────────────────
    @property
    def bands(self) -> list[str]:
        return self.metrics["bands"]

    @property
    def tau(self) -> float:
        return float(self.metrics["tau"])

    @property
    def eta(self) -> float:
        return float(self.metrics["eta"])

    @property
    def M_values(self) -> list[int]:
        return list(self.dataset_nc["sensors"]["counts"])

    @property
    def sigma_values(self) -> list[float]:
        return list(self.dataset_nc["noise"]["test_sigmas"])

    @property
    def seeds(self) -> list[int]:
        return list(self.dataset_nc["models"]["mlp"]["seeds"])

    @property
    def pod_rank(self) -> int:
        return int(self.dataset_nc["pod"]["main_rank"])

    @property
    def wavelet_family(self) -> str:
        return self.dataset_nc["wavelet"]["family"]

    @property
    def wavelet_level(self) -> int:
        return int(self.dataset_nc["wavelet"]["level"])

    @property
    def wavelet_mode(self) -> str:
        return self.dataset_nc["wavelet"]["boundary_mode"]

    @property
    def bootstrap_n(self) -> int:
        return int(self.statistics["bootstrap"]["n_resamples"])

    @property
    def build_dir(self) -> Path:
        return ROOT / "applications" / "paper_main" / "build"

    @property
    def figures_out(self) -> Path:
        return self.build_dir / "figures"

    @property
    def tables_out(self) -> Path:
        return self.build_dir / "generated" / "tables"


_config_singleton: Config | None = None


def get_config() -> Config:
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton

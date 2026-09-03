"""Dataset adapters for paper-expand (P2) — 第二数据集适配层。

封装数据集特有细节（数据路径、网格、通道、split、物理域 mean/std、POD 基、
mask 前缀），使 P0/P1 流水线可以在第二个数据集（默认 RDB）上重复运行，
而无需修改 luna/features 核心实现。

NC 是 P0/P1 的参考实现（默认值），paper-expand 通过 adapter 路由所有
dataset-specific 的部分。

用法:
    from features.datasets.adapter import get_adapter
    ad = get_adapter("rdb_h5")
    split = ad.split_indices(0)              # {'train','val','test'}
    m, s = ad.phys_mean_std()                # 物理域噪声 mean/std
    mask = ad.load_mask(family_id, M)        # (H, W) bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# NC 物理域 mean/std（P0 协议，compute_ridge_closed_form 的确定性噪声用）
NC_PHYS_MEAN = np.asarray([1.0004944, -0.00017817653], dtype=np.float64)
NC_PHYS_STD = np.asarray([0.21863055, 0.19121747], dtype=np.float64)
# RDB 物理域 mean/std（从 data/rdb_h5.npy 全量计算，2026-08-07 实测）
RDB_PHYS_MEAN = np.asarray([1.0327148], dtype=np.float64)
RDB_PHYS_STD = np.asarray([0.11445527], dtype=np.float64)

DEFAULT_TEST_RATIO = 0.2
DEFAULT_VAL_RATIO = 0.1


class DatasetAdapter:
    """公共接口 + NC 默认值（P0/P1 参考实现，行为保持不变）。"""

    name: str = "nc"
    label: str = "cylinder2d_q1 (NC)"
    data_path: Path = ROOT / "data/cylinder2d_q1.npy"
    pod_bundle_path: Path = ROOT / "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz"
    band_pod_bundle_path: Path | None = ROOT / "artifacts/wavelet_pod_nc_2000/band_pod_bundle.npz"
    grid: tuple[int, int] = (80, 160)
    channels: int = 2
    test_ratio: float = DEFAULT_TEST_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    phys_mean: np.ndarray = NC_PHYS_MEAN
    phys_std: np.ndarray = NC_PHYS_STD
    data_in_physical_domain: bool = False   # NC: 数据文件为标准化域（噪声需反标准化）
    mask_prefix: str = "cylinder2d_80x160_random_inc"
    mask_dir: Path = ROOT / "masks2"
    # 历史 VCNN 根（family_01 = P0 masks2 权威源）
    vcnn_roots: dict[str, Path] = {
        "seed000": ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000",
        "seed101": ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000_seed101",
        "seed202": ROOT / "artifacts/vcnn_results/vcnn_sweep_nc_2000_seed202",
    }

    # ── 数据 ─────────────────────────────────────────────────
    def load_snapshots(self, mmap: bool = True) -> np.ndarray:
        return np.load(str(self.data_path), mmap_mode="r" if mmap else None)

    def n_snapshots(self) -> int:
        arr = self.load_snapshots(mmap=True)
        return int(arr.shape[0])

    def grid_shape(self) -> tuple[int, int]:
        return self.grid

    def n_channels(self) -> int:
        return self.channels

    # ── split ────────────────────────────────────────────────
    def split_indices(
        self,
        seed: int,
        test_ratio: float | None = None,
        val_ratio: float | None = None,
    ) -> dict[str, np.ndarray]:
        """torch random_split 语义（与 NC MLP 协议一致，luna 实现）。"""
        from features.training.pod_sweep import split_indices as _si

        n = self.n_snapshots()
        return _si(
            n,
            seed,
            test_ratio=self.test_ratio if test_ratio is None else test_ratio,
            val_ratio=self.val_ratio if val_ratio is None else val_ratio,
        )

    # ── 物理域噪声参数 ───────────────────────────────────────
    def phys_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        return self.phys_mean, self.phys_std

    # ── POD ──────────────────────────────────────────────────
    def pod_bundle(self) -> Path:
        return self.pod_bundle_path

    def band_pod_bundle(self) -> Path | None:
        return self.band_pod_bundle_path

    # ── mask ─────────────────────────────────────────────────
    def mask_csv_name(self, M: int) -> str:
        return f"{self.mask_prefix}_n{M:03d}.csv"

    def mask_csv_path(self, family_id: str, M: int, root: Path | None = None) -> Path:
        """返回 family 目录下的 mask CSV（masks_expand/<dataset>/<family>/masks/...）。"""
        base = root if root is not None else ROOT / "masks_expand" / self.name
        return base / family_id / "masks" / self.mask_csv_name(M)

    def load_mask(self, family_id: str, M: int, root: Path | None = None) -> np.ndarray:
        """加载 (M, 2) int 传感器坐标（row, col）。"""
        path = self.mask_csv_path(family_id, M, root)
        if not path.exists():
            raise FileNotFoundError(
                f"Mask {path} 不存在 — 先运行 tools/prepare_incremental_masks.py 生成 "
                f"{self.name} 的 family（masks_expand/{self.name}/）"
            )
        coords = np.loadtxt(str(path), delimiter=",", skiprows=1).astype(np.int64)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != M:
            raise ValueError(f"Unexpected mask shape for {path}: {coords.shape}")
        return coords

    def load_mask_bool(self, family_id: str, M: int, root: Path | None = None) -> np.ndarray:
        """返回 (H, W) bool mask。"""
        H, W = self.grid
        coords = self.load_mask(family_id, M, root)
        m = np.zeros((H, W), dtype=bool)
        m[coords[:, 0], coords[:, 1]] = True
        return m

    def mask_root_for(self, family_id: str) -> Path:
        """paper-expand 使用的 family mask 根目录。"""
        return ROOT / "masks_expand" / self.name / family_id

    # ── 有效区域（传感器采样候选区）────────────────────────
    def get_valid_mask(self) -> np.ndarray:
        """返回 (H, W) bool：可放置传感器的有效区域。默认全有效（NC）。"""
        H, W = self.grid
        return np.ones((H, W), dtype=bool)

    # ── VCNN 历史（family_01 = masks2 权威源）────────────────
    def vcnn_root(self, seed: int) -> Path | None:
        return self.vcnn_roots.get(f"seed{int(seed):03d}")


class RdbAdapter(DatasetAdapter):
    """RDB（rdb_h5，湍流槽道流）— 第二数据集默认 adapter。

    - 单通道 128×128，5050 快照，无 NaN；
    - POD 128 阶已存在，能量收敛 100%（2026-08-07 实测）；
    - split = torch random_split（test 1010），历史 VCNN seed000 的
      test_indices 与 seed0 split 完全一致（已核验），可直接登记复用；
    - mask = radial incremental（masks2 权威源 = family_01）。
    """

    name: str = "rdb_h5"
    label: str = "rdb_h5 (turbulent channel flow)"
    data_path: Path = ROOT / "data/rdb_h5.npy"
    # 主基 = 512 阶（满足 mean criterion，2026-08-07 实测）；128 阶作为 fallback/对照
    pod_bundle_path: Path = ROOT / "artifacts/pod_bases/rdb_h5/pod_base_bundle_r512.npz"
    pod_bundle_path_r128: Path = ROOT / "artifacts/pod_bases/rdb_h5/pod_base_bundle.npz"
    band_pod_bundle_path: Path | None = ROOT / "artifacts/wavelet_pod_rdb_2000/band_pod_bundle.npz"
    grid: tuple[int, int] = (128, 128)
    channels: int = 1
    phys_mean: np.ndarray = RDB_PHYS_MEAN
    phys_std: np.ndarray = RDB_PHYS_STD
    data_in_physical_domain: bool = True    # RDB: 数据文件即物理域（噪声直接加）
    mask_prefix: str = "rdb_h5_128x128_radial_inc"
    vcnn_roots: dict[str, Path] = {
        "seed000": ROOT / "artifacts/vcnn_results/vcnn_sweep_rdb_2000_seed000",
    }
    # 有效区域缓存：RDB 43% 网格点恒定（std=0，边界填充），传感器只放有效区
    valid_mask_path: Path = ROOT / "artifacts/derived/expand/rdb_valid_mask.npy"

    def get_valid_mask(self) -> np.ndarray:
        """RDB 有效区域 = 非恒定网格点（std > 1e-4），缓存到 artifacts。"""
        if self.valid_mask_path.exists():
            return np.load(str(self.valid_mask_path))
        fields = np.load(str(self.data_path), mmap_mode="r").astype(np.float64, copy=False)
        std = np.asarray(fields).std(axis=0)[:, :, 0]
        m = std > 1e-4
        self.valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.valid_mask_path), m)
        return m

    def pod_bundle(self) -> Path:
        """优先 r512（满足 criterion），缺失时回退 r128。"""
        if self.pod_bundle_path.exists():
            return self.pod_bundle_path
        return self.pod_bundle_path_r128


_ADAPTERS: dict[str, DatasetAdapter] = {
    "nc": DatasetAdapter(),
    "rdb_h5": RdbAdapter(),
}


def get_adapter(dataset: str) -> DatasetAdapter:
    if dataset not in _ADAPTERS:
        raise KeyError(
            f"Unknown adapter '{dataset}'. Available: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[dataset]


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)

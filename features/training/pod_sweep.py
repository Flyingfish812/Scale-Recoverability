"""
Supplementary — POD model sweep core (MLP training / closed-form Ridge / Gappy POD).

Replicates the P0 training & NPZ protocol exactly (same split via
torch.Generator().manual_seed(seed), same noise protocol, same test_raw.npz
schema) but parameterizes the sensor-mask family so the 5 supplementary families
share the SAME test snapshots per seed (cross-family comparability).

Protocol notes (locked to P0):
  - split: random_split with torch.Generator().manual_seed(training_seed);
    test_ratio=0.2, val_ratio=0.1  -> 300 test / 120 val / 1081 train
  - noise: physical-domain Gaussian via RandomState(42) (closed-form Ridge);
    per-sample noise_seed = training_seed + idx (dataset protocol)
  - MLP: AdamW(lr=1e-3, wd=1e-4), early stop patience=30, max 5000 epochs,
    hidden (256,256) relu, batch 64
  - NPZ: output_nchw (N,C,H,W), target_nchw, test_indices, noise_sigma

Outputs under artifacts/derived/supplementary/predictions/{family}/{model}_{tag}/...
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from luna.data.io import load_npy
from luna.models.pod_linear import build_pod_linear_model
from luna.models.pod_mlp import build_pod_mlp_model

EPS = 1e-12

# P0 权威 test 划分（MLP seed0 的 300 测试快照）；Ridge/Gappy 复用
MLP_SEED0_TEST_NPZ = (
    Path(__file__).resolve().parents[2]
    / "artifacts/pod_model_sweep_nc/mlp_n0010/seed000/tests/s0000/test_raw.npz"
)


# ══════════════════════════════════════════════════════════════════
# Data & split
# ══════════════════════════════════════════════════════════════════

def compute_channel_mean_std(fields_thwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(fields_thwc, dtype=np.float64)
    mean = np.mean(data, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.std(data, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class PODObservationDataset(Dataset):
    """Maps sparse observations at mask points to POD coefficients.

    Identical protocol to the P0 training code (mask sampling, per-sample
    noise seed = base_seed + idx, per-channel normalization).
    """

    def __init__(
        self,
        fields_thwc: np.ndarray,
        mask_hw: np.ndarray,
        pod_coeff: np.ndarray,
        *,
        noise_sigma: float = 0.0,
        normalize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fields = np.asarray(fields_thwc, dtype=np.float32)
        self.T, self.H, self.W, self.C = self.fields.shape
        self.mask = np.asarray(mask_hw, dtype=bool)
        if self.mask.shape != (self.H, self.W):
            raise ValueError(f"Mask shape {self.mask.shape} != ({self.H}, {self.W})")
        self.obs_indices = np.argwhere(self.mask)
        self.n_obs = int(self.obs_indices.shape[0])
        self.coeff = np.asarray(pod_coeff, dtype=np.float32)
        self.noise_sigma = float(noise_sigma)
        self.seed = None if seed is None else int(seed)
        self.normalize = bool(normalize)
        if self.normalize:
            self.chan_mean, self.chan_std = compute_channel_mean_std(self.fields)
        else:
            self.chan_mean = np.zeros(self.C, dtype=np.float32)
            self.chan_std = np.ones(self.C, dtype=np.float32)

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int):
        field_hwc = self.fields[int(idx)]
        obs_vals = field_hwc[self.obs_indices[:, 0], self.obs_indices[:, 1], :]
        if self.noise_sigma > 0.0:
            seed = None if self.seed is None else self.seed + int(idx)
            obs_vals = obs_vals + np.random.RandomState(seed).normal(
                0.0, self.noise_sigma, size=obs_vals.shape
            ).astype(np.float32)
        obs_flat = obs_vals.reshape(-1).astype(np.float32)
        ch_idx = np.tile(np.arange(self.C), self.n_obs)
        obs_norm = (obs_flat - self.chan_mean[ch_idx]) / self.chan_std[ch_idx]
        target = self.coeff[int(idx)]
        return torch.from_numpy(obs_norm), torch.from_numpy(target)


def split_indices(n_total: int, seed: int, test_ratio: float = 0.2, val_ratio: float = 0.1) -> dict[str, np.ndarray]:
    """Reproduce P0's random_split with torch.Generator().manual_seed(seed)."""
    n_test = max(1, min(int(round(n_total * test_ratio)), n_total - 2))
    remain = n_total - n_test
    n_val = max(1, min(int(round(remain * val_ratio)), remain - 1))
    n_train = n_total - n_val - n_test
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=gen).tolist()
    return {
        "train": np.asarray(perm[:n_train], dtype=np.int64),
        "val": np.asarray(perm[n_train:n_train + n_val], dtype=np.int64),
        "test": np.asarray(perm[n_train + n_val:], dtype=np.int64),
    }


def split_like_p0_seed0(n_total: int, mlp_seed0_test_npz: Path) -> dict[str, np.ndarray]:
    """P0 closed-form Ridge / Gappy split (compute_ridge_closed_form.py):

        test      = MLP seed0 test indices (300, sorted)
        train_val = remaining 1201 (sorted)
        val       = np.random.choice(train_val, 120, replace=False) with seed 42
        train     = train_val minus val
    """
    ref = np.load(str(mlp_seed0_test_npz))
    test = np.asarray(sorted(set(ref["test_indices"].tolist())), dtype=np.int64)
    train_val = np.asarray(sorted(set(range(int(n_total))) - set(int(i) for i in test)),
                           dtype=np.int64)
    rng = np.random.RandomState(42)
    n_val = int(round(len(train_val) * 0.1))
    val = set(int(i) for i in rng.choice(train_val, size=n_val, replace=False))
    train = np.asarray([int(i) for i in train_val if int(i) not in val], dtype=np.int64)
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "val": np.asarray(sorted(val), dtype=np.int64),
        "test": test,
    }


def _resolve_split(
    test_indices: Optional[np.ndarray],
    n_total: int,
) -> dict[str, np.ndarray]:
    """Ridge/Gappy 的 split 解析（paper-expand 用）：

    - test_indices 为 None 时回退 NC 默认（split_like_p0_seed0）；
    - 否则构造与 split_like_p0_seed0 同构的 split：
      test = 给定 indices（sorted），val = RandomState(42) 选 10%，余为 train。
    """
    if test_indices is None:
        return split_like_p0_seed0(n_total, MLP_SEED0_TEST_NPZ)
    test = np.asarray(sorted(set(int(i) for i in test_indices)), dtype=np.int64)
    train_val = np.asarray(sorted(set(range(int(n_total))) - set(int(i) for i in test)),
                           dtype=np.int64)
    rng = np.random.RandomState(42)
    n_val = int(round(len(train_val) * 0.1))
    val = set(int(i) for i in rng.choice(train_val, size=n_val, replace=False))
    train = np.asarray([int(i) for i in train_val if int(i) not in val], dtype=np.int64)
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "val": np.asarray(sorted(val), dtype=np.int64),
        "test": test,
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def _format_duration(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def train_model_route(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_epochs: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    early_patience: int = 30,
    progress_every: int = 50,
    verbose: bool = True,
) -> dict[str, Any]:
    """P0 training protocol: AdamW, no scheduler, early stop on val loss."""
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state: Optional[dict] = None
    patience = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_sum = 0.0
        for obs, tgt in train_loader:
            obs, tgt = obs.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(obs), tgt)
            loss.backward()
            optimizer.step()
            tr_sum += float(loss.item()) * obs.size(0)
        avg_tr = tr_sum / max(1, len(train_loader.dataset))
        train_losses.append(avg_tr)

        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for obs, tgt in val_loader:
                obs, tgt = obs.to(device), tgt.to(device)
                va_sum += float(loss_fn(model(obs), tgt).item()) * obs.size(0)
        avg_va = va_sum / max(1, len(val_loader.dataset))
        val_losses.append(avg_va)

        if avg_va < best_val:
            best_val = avg_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        elif epoch >= 5:
            patience += 1
            if patience >= early_patience:
                break

        if verbose and (epoch % progress_every == 0 or epoch == 1):
            el = time.perf_counter() - t0
            print(f"  epoch {epoch:04d}/{num_epochs}  train={avg_tr:.4e}  val={avg_va:.4e}  "
                  f"elapsed={_format_duration(el)}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_loss": float(best_val),
        "best_epoch": int(np.argmin(val_losses)) + 1 if val_losses else 0,
        "epochs_ran": int(len(train_losses)),
        "stopped_early": bool(len(train_losses) < num_epochs),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ══════════════════════════════════════════════════════════════════
# Evaluation & NPZ saving (same schema as P0)
# ══════════════════════════════════════════════════════════════════

def reconstruct_field(pod_coeff: np.ndarray, pod_basis: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
    R = pod_coeff.shape[0]
    basis_flat = np.asarray(pod_basis, dtype=np.float64).reshape(R, -1)
    recon = mean_field.ravel().astype(np.float64) + np.asarray(pod_coeff, dtype=np.float64) @ basis_flat
    return recon.reshape(mean_field.shape).astype(np.float32)


def save_test_raw(
    pred_fields_nchw: np.ndarray,
    target_fields_nchw: np.ndarray,
    test_indices: np.ndarray,
    noise_sigma: float,
    out_dir: Path,
    *,
    mask_meta: dict,
    train_info: dict,
    model_type: str,
    family: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "test_raw.npz"
    np.savez_compressed(
        str(npz_path),
        output_nchw=pred_fields_nchw.astype(np.float32),
        target_nchw=target_fields_nchw.astype(np.float32),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        noise_sigma=np.asarray(float(noise_sigma), dtype=np.float32),
    )
    meta = {
        "schema_version": "luna.supplementary.pod_model.test_raw.v1",
        "model_type": model_type,
        "mask_family": family,
        "noise_sigma": float(noise_sigma),
        "test_count": int(pred_fields_nchw.shape[0]),
        "output_shape": list(pred_fields_nchw.shape),
        "test_indices": np.asarray(test_indices).tolist(),
        "mask_meta": mask_meta,
        "train_best_epoch": int(train_info.get("best_epoch", 0)),
        "train_best_val_loss": float(train_info.get("best_val_loss", float("nan"))),
    }
    (out_dir / "test_raw_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


# ══════════════════════════════════════════════════════════════════
# Case runners
# ══════════════════════════════════════════════════════════════════

def _load_pod_bundle(pod_bundle_path: Path, n_modes: int):
    pod = np.load(str(pod_bundle_path))
    basis = np.asarray(pod["pod_basis"], dtype=np.float32)[:n_modes]
    coeff = np.asarray(pod["coefficients"], dtype=np.float32)[:, :n_modes]
    mean_field = np.asarray(pod["mean_field"], dtype=np.float32)
    return basis, coeff, mean_field


def run_mlp_case(
    *,
    family: str,
    M: int,
    training_seed: int,
    data_path: Path,
    pod_bundle_path: Path,
    mask_hw: np.ndarray,
    out_root: Path,
    test_sigmas: Sequence[float] = (0.0, 0.001, 0.01, 0.1),
    n_modes: int = 128,
    num_epochs: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    early_patience: int = 30,
    device: str = "cpu",
    skip_if_exists: bool = True,
    expected_test: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train MLP for (family, M, seed) and save test_raw.npz per noise level."""
    fields = np.load(str(data_path), mmap_mode="r").astype(np.float32, copy=False)
    T, H, W, C = fields.shape
    basis, coeff, mean_field = _load_pod_bundle(pod_bundle_path, n_modes)
    R = int(basis.shape[0])

    mask_tag = f"n{M:04d}"
    seed_tag = f"seed{int(training_seed):03d}"
    case_dir = out_root / f"mlp_{mask_tag}" / seed_tag
    out = {"family": family, "model": "mlp", "M": M, "training_seed": int(training_seed),
           "case_dir": str(case_dir), "npz_paths": {}}

    split = split_indices(T, training_seed)
    test_idx = np.sort(split["test"])  # sorted order = P0 protocol
    if expected_test is not None and len(test_idx) != expected_test:
        print(f"  [warn] test split = {len(test_idx)} (expected {expected_test})")

    dataset = PODObservationDataset(fields, mask_hw, coeff, noise_sigma=0.0, normalize=True, seed=int(training_seed))
    from torch.utils.data import Subset
    train_ds = Subset(dataset, split["train"].tolist())
    val_ds = Subset(dataset, split["val"].tolist())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_pod_mlp_model(n_obs=M, n_channels=C, n_modes=R,
                                hidden_sizes=(256, 256), dropout=0.0, activation="relu")
    model = model.to(device)
    t0 = time.time()
    if verbose:
        print(f"  [MLP] {family} M={M} seed={training_seed}  train={len(split['train'])} "
              f"val={len(split['val'])} test={len(test_idx)}  device={device}")
    info = train_model_route(model, train_loader, val_loader, num_epochs=num_epochs,
                             lr=lr, weight_decay=weight_decay, device=device,
                             early_patience=early_patience, verbose=verbose)
    train_sec = time.time() - t0

    # save checkpoint + summary
    case_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "train_info": {k: v for k, v in info.items()
                if k not in ("train_losses", "val_losses")}}, case_dir / "mlp_best.pt")
    summary = {k: v for k, v in info.items() if k not in ("train_losses", "val_losses")}
    summary.update({"model_type": "mlp", "mask_family": family, "M": M, "training_seed": int(training_seed),
                    "train_seconds": round(train_sec, 1), "device": str(device)})
    (case_dir / "mlp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # evaluate at each test noise
    mask_meta = {"mode": "csv", "mask_family": family, "mask_num": M}
    with torch.no_grad():
        model.eval()
        for sigma in test_sigmas:
            code = f"s{int(round(float(sigma) * 10000)):04d}"
            test_dir = case_dir / "tests" / code
            npz_path = test_dir / "test_raw.npz"
            if skip_if_exists and npz_path.exists():
                out["npz_paths"][float(sigma)] = str(npz_path)
                continue
            # apply test noise (dataset noise_sigma override + same per-sample seed)
            dataset.noise_sigma = float(sigma)
            outs, tgts = [], []
            for i in test_idx:
                obs, _ = dataset[int(i)]
                pred = model(obs.unsqueeze(0).to(device)).detach().cpu().numpy().ravel()
                pf = reconstruct_field(pred, basis, mean_field)
                tf = dataset.fields[int(i)]
                outs.append(np.transpose(pf, (2, 0, 1))[None])
                tgts.append(np.transpose(tf, (2, 0, 1))[None])
            dataset.noise_sigma = 0.0
            out_npz = np.concatenate(outs, axis=0)
            tgt_npz = np.concatenate(tgts, axis=0)
            save_test_raw(out_npz, tgt_npz, test_idx, float(sigma), test_dir,
                          mask_meta=mask_meta, train_info=info, model_type="mlp", family=family)
            if verbose:
                rel = float(np.linalg.norm(out_npz.ravel() - tgt_npz.ravel()) /
                            (np.linalg.norm(tgt_npz.ravel()) + EPS))
                print(f"    σ={sigma}: test_raw.npz saved (N={len(test_idx)}, relL2={rel:.6f})")
            out["npz_paths"][float(sigma)] = str(npz_path)
    out["train_seconds"] = round(train_sec, 1)
    out["best_val_loss"] = info["best_val_loss"]
    out["best_epoch"] = info["best_epoch"]
    out["epochs_ran"] = info["epochs_ran"]
    return out


def run_ridge_closed_form_case(
    *,
    family: str,
    M: int,
    data_path: Path,
    pod_bundle_path: Path,
    mask_hw: np.ndarray,
    out_root: Path,
    test_sigmas: Sequence[float] = (0.0, 0.001, 0.01, 0.1),
    n_modes: int = 128,
    lambda_grid: Optional[np.ndarray] = None,
    phys_mean: Optional[np.ndarray] = None,
    phys_std: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    noise_domain: str = "normalized",
    obs_normalize_from_mask: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Closed-form Ridge (deterministic) — replicates compute_ridge_closed_form
    but parameterized by mask family; uses training_seed=0 split & noise seed 42.

    paper-expand:
      - phys_mean/phys_std 覆盖物理域噪声参数（默认 NC）;
      - test_indices 覆盖测试集（默认 NC MLP seed0 的 300）;
      - noise_domain: "normalized"（默认，NC：数据标准化域，反标准化加噪再标准化）
        | "physical"（数据本身在物理域，噪声直接加，如 RDB）;
      - obs_normalize_from_mask: True 时观测 mean/std 严格按 mask 点位置计算
        （RDB 网格展平前 n_obs 含恒定点会令 obs_std≈1e-8 导致数值爆炸；
        NC 默认 False 保持与 compute_ridge_closed_form/s05 一致）。"""
    if lambda_grid is None:
        lambda_grid = np.logspace(-8, 2, 21)
    fields = np.load(str(data_path), mmap_mode="r")  # float32 (P0 protocol)
    T, H, W, C = fields.shape
    pod = np.load(str(pod_bundle_path))
    basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)[:n_modes]
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)[:, :n_modes]

    split = _resolve_split(test_indices, T)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    train_f, val_f, test_f = fields[train_idx], fields[val_idx], fields[test_idx]
    train_c, val_c = full_coeffs[train_idx], full_coeffs[val_idx]

    obs_idx = np.argwhere(mask_hw)
    n_obs = len(obs_idx)

    def obs_matrix(fields_sel):
        o = np.empty((fields_sel.shape[0], n_obs * C), dtype=np.float64)
        for i in range(fields_sel.shape[0]):
            o[i] = fields_sel[i, obs_idx[:, 0], obs_idx[:, 1], :].ravel()
        return o

    # 观测标准化参数（NC 历史协议：网格展平前 n_obs；RDB：按 mask 点位置）
    tr_obs_raw = obs_matrix(train_f)
    if obs_normalize_from_mask:
        obs_mean_m = tr_obs_raw.mean(axis=0)
        obs_std_m = tr_obs_raw.std(axis=0) + 1e-8
    else:
        obs_mean_m = np.mean(train_f, axis=0).ravel()[:n_obs * C]
        obs_std_m = np.std(train_f, axis=0).ravel()[:n_obs * C] + 1e-8

    coeff_mean = np.mean(train_c, axis=0)
    coeff_std = np.std(train_c, axis=0) + 1e-8
    basis_flat = basis_4d.reshape(n_modes, -1).T
    mean_flat = mean_field.ravel()

    tr_obs = (tr_obs_raw - obs_mean_m) / obs_std_m
    va_obs = (obs_matrix(val_f) - obs_mean_m) / obs_std_m
    tr_coeff = (train_c - coeff_mean) / coeff_std
    va_coeff = (val_c - coeff_mean) / coeff_std
    tr_X = np.concatenate([tr_obs, np.ones((tr_obs.shape[0], 1))], axis=1)
    va_X = np.concatenate([va_obs, np.ones((va_obs.shape[0], 1))], axis=1)
    XTX = tr_X.T @ tr_X
    XTA = tr_X.T @ tr_coeff
    d = XTX.shape[0]

    best_loss, best_Wmat = float("inf"), None
    for lam in lambda_grid:
        I = np.eye(d); I[-1, -1] = 0.0
        Wmat = np.linalg.solve(XTX + lam * I, XTA)
        loss = float(np.mean((va_X @ Wmat - va_coeff) ** 2))
        if loss < best_loss:
            best_loss, best_Wmat = loss, Wmat

    mask_tag = f"n{M:04d}"
    case_dir = out_root / f"ridge_{mask_tag}" / "seed000"
    out = {"family": family, "model": "ridge", "M": M, "training_seed": 0,
           "case_dir": str(case_dir), "npz_paths": {}, "best_val_loss": best_loss}

    # deterministic noise (RandomState(42)) on physical domain — same as P0
    if phys_mean is None:
        phys_mean = np.asarray([1.0004944, -0.00017817653], dtype=np.float64)
    if phys_std is None:
        phys_std = np.asarray([0.21863055, 0.19121747], dtype=np.float64)
    mean_v = np.asarray(phys_mean, dtype=np.float64).reshape(-1)
    std_v = np.asarray(phys_std, dtype=np.float64).reshape(-1)
    tgt_nchw = np.asarray(test_f).transpose(0, 3, 1, 2).astype(np.float32)
    for sigma in test_sigmas:
        code = f"s{int(round(float(sigma) * 10000)):04d}"
        test_dir = case_dir / "tests" / code
        npz_path = test_dir / "test_raw.npz"
        if npz_path.exists():
            out["npz_paths"][float(sigma)] = str(npz_path)
            continue
        if sigma == 0.0:
            te_f = test_f
        elif noise_domain == "physical":
            # 数据本身在物理域（如 RDB）：噪声直接加在原始值上
            noise = np.random.RandomState(42).randn(*test_f.shape).astype(np.float64) * sigma
            te_f = test_f + noise
        else:
            # NC 协议：标准化域 → 物理域加噪 → 标准化回来
            phys = test_f * std_v[None, None, None, :] + mean_v[None, None, None, :]
            noise = np.random.RandomState(42).randn(*phys.shape).astype(np.float64) * sigma
            te_f = (phys + noise - mean_v[None, None, None, :]) / std_v[None, None, None, :]
        te_obs = (obs_matrix(te_f) - obs_mean_m) / obs_std_m
        te_X = np.concatenate([te_obs, np.ones((te_obs.shape[0], 1))], axis=1)
        pred = (te_X @ best_Wmat) * coeff_std + coeff_mean
        pred_flat = mean_flat[None, :] + (pred @ basis_flat.T)
        pred_nchw = pred_flat.reshape(len(test_idx), H, W, C).transpose(0, 3, 1, 2).astype(np.float32)
        save_test_raw(pred_nchw, tgt_nchw, test_idx, float(sigma), test_dir,
                      mask_meta={"mode": "csv", "mask_family": family, "mask_num": M},
                      train_info={"best_epoch": 0, "best_val_loss": best_loss},
                      model_type="ridge", family=family)
        if verbose:
            rel = float(np.linalg.norm(pred_nchw - tgt_nchw) / (np.linalg.norm(tgt_nchw) + EPS))
            print(f"    σ={sigma}: ridge saved (N={len(test_idx)}, relL2={rel:.6f})")
        out["npz_paths"][float(sigma)] = str(npz_path)
    return out


def run_gappy_case(
    *,
    family: str,
    M: int,
    data_path: Path,
    pod_bundle_path: Path,
    mask_hw: np.ndarray,
    out_root: Path,
    test_sigmas: Sequence[float] = (0.0, 0.001, 0.01, 0.1),
    n_modes: int = 128,
    candidate_ranks: Sequence[int] = (4, 8, 12, 16, 20, 24, 32),
    phys_mean: Optional[np.ndarray] = None,
    phys_std: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    noise_domain: str = "normalized",
    verbose: bool = True,
) -> dict[str, Any]:
    """Gappy POD (deterministic) — replicates compute_s23_gappy: rank<=M chosen
    on validation set; uses seed0 split.

    paper-expand: phys_mean/phys_std 覆盖物理域噪声参数（默认 NC），
    test_indices 覆盖测试集（默认 NC MLP seed0 的 300），
    noise_domain: "normalized"（默认，NC）| "physical"（数据本身物理域，直接加噪）。"""
    fields = np.load(str(data_path), mmap_mode="r").astype(np.float64, copy=False)
    T, H, W, C = fields.shape
    pod = np.load(str(pod_bundle_path))
    basis_4d = np.asarray(pod["pod_basis"], dtype=np.float64)[:n_modes]
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    full_coeffs = np.asarray(pod["coefficients"], dtype=np.float64)[:, :n_modes]

    split = _resolve_split(test_indices, T)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

    obs_idx = np.argwhere(mask_hw)
    n_obs = len(obs_idx)

    def obs_matrix(fields_sel):
        o = np.empty((fields_sel.shape[0], n_obs * C), dtype=np.float64)
        for i in range(fields_sel.shape[0]):
            o[i] = fields_sel[i, obs_idx[:, 0], obs_idx[:, 1], :].ravel()
        return o

    def gappy_predict(obs_sel, rank):
        """â = (C_M Φ_r)† (y - C_M ū); C_M selects obs positions (n_obs*C rows)."""
        phi = basis_4d[:rank].reshape(rank, -1).T  # (D, r)
        rowcol = obs_idx[:, 0] * W + obs_idx[:, 1]  # (n_obs,)
        flat_idx = (rowcol[:, None] * C + np.arange(C)[None, :]).ravel()  # (n_obs*C,)
        cm_phi = phi[flat_idx, :]  # (n_obs*C, r)
        mean_obs = mean_field.ravel()[flat_idx]
        pseudo = np.linalg.pinv(cm_phi)
        coeffs = np.empty((obs_sel.shape[0], rank), dtype=np.float64)
        for i in range(obs_sel.shape[0]):
            y = obs_sel[i].ravel() - mean_obs
            coeffs[i] = pseudo @ y
        return coeffs

    # select rank on validation
    train_c = full_coeffs[train_idx]
    val_f = fields[val_idx]
    ranks = [r for r in candidate_ranks if r <= n_obs and r <= n_modes]
    best_rank, best_err = ranks[0], float("inf")
    val_obs = obs_matrix(val_f)
    for r in ranks:
        coeffs = gappy_predict(val_obs, r)
        recon = mean_field.ravel()[None, :] + coeffs @ basis_4d[:r].reshape(r, -1)
        err = float(np.mean(np.linalg.norm(recon - val_f.reshape(len(val_idx), -1), axis=1) /
                            (np.linalg.norm(val_f.reshape(len(val_idx), -1), axis=1) + EPS)))
        if err < best_err:
            best_err, best_rank = err, r
    if verbose:
        print(f"  [Gappy] {family} M={M} rank={best_rank} (val err={best_err:.5f})")

    mask_tag = f"n{M:04d}"
    case_dir = out_root / f"gappy_{mask_tag}" / "seed000"
    out = {"family": family, "model": "gappy", "M": M, "training_seed": 0,
           "case_dir": str(case_dir), "rank": best_rank, "npz_paths": {}}
    test_f = fields[test_idx]
    tgt_nchw = np.asarray(test_f).transpose(0, 3, 1, 2).astype(np.float32)
    for sigma in test_sigmas:
        code = f"s{int(round(float(sigma) * 10000)):04d}"
        test_dir = case_dir / "tests" / code
        npz_path = test_dir / "test_raw.npz"
        if npz_path.exists():
            out["npz_paths"][float(sigma)] = str(npz_path)
            continue
        if sigma == 0.0:
            te_f = test_f
        elif noise_domain == "physical":
            noise = np.random.RandomState(42).randn(*test_f.shape).astype(np.float64) * sigma
            te_f = test_f + noise
        else:
            if phys_mean is None:
                phys_mean = np.asarray([1.0004944, -0.00017817653], dtype=np.float64)
            if phys_std is None:
                phys_std = np.asarray([0.21863055, 0.19121747], dtype=np.float64)
            mean_v = np.asarray(phys_mean, dtype=np.float64).reshape(-1)
            std_v = np.asarray(phys_std, dtype=np.float64).reshape(-1)
            phys = test_f * std_v[None, None, None, :] + mean_v[None, None, None, :]
            noise = np.random.RandomState(42).randn(*phys.shape).astype(np.float64) * sigma
            te_f = (phys + noise - mean_v[None, None, None, :]) / std_v[None, None, None, :]
        coeffs = gappy_predict(obs_matrix(te_f), best_rank)
        recon = mean_field.ravel()[None, :] + coeffs @ basis_4d[:best_rank].reshape(best_rank, -1)
        pred_nchw = recon.reshape(len(test_idx), H, W, C).transpose(0, 3, 1, 2).astype(np.float32)
        save_test_raw(pred_nchw, tgt_nchw, test_idx, float(sigma), test_dir,
                      mask_meta={"mode": "csv", "mask_family": family, "mask_num": M, "rank": best_rank},
                      train_info={"best_epoch": 0, "best_val_loss": best_err},
                      model_type="gappy", family=family)
        out["npz_paths"][float(sigma)] = str(npz_path)
    return out

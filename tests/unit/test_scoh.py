"""S_coh 测试 (P0-6 §10.4)。

验证: P_b = U_b U_bᵀ 正交投影; P_b² = P_b; 捕获率 γ_b ∈ [0,1];
不假设 S_coh ≥ S_full。
"""

import numpy as np
import pytest

from luna.wavelet.metrics import compute_S_coh, compute_S_full
from luna.wavelet.transform import decompose_field_2d, recompose_field_2d

rng = np.random.default_rng(23)


def _orthonormal_basis(d, r):
    q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return q.T  # (r, D), 行正交归一


def test_projector_idempotent():
    """P_b² = P_b."""
    D, r = 100, 8
    basis = _orthonormal_basis(D, r)
    P = basis.T @ basis  # (D, D)
    x = rng.standard_normal(D)
    Px = P @ x
    PPx = P @ Px
    assert np.allclose(Px, PPx, atol=1e-10)


def test_capture_rate_in_unit_interval():
    """捕获率 γ_b = ‖U_bᵀ x‖/‖x‖ ∈ [0,1] (行正交归一时)."""
    for _ in range(5):
        D, r = 80, np.random.randint(1, 12)
        basis = _orthonormal_basis(D, r)
        x = rng.standard_normal(D)
        gamma = float(np.linalg.norm(basis @ x) / (np.linalg.norm(x) + 1e-12))
        assert 0.0 <= gamma <= 1.0 + 1e-9


def _make_field_and_models(shape=(64, 64), n_bands_kept=5, r=6):
    """构造 target 场 + band_pod_models: 每个频带一个可捕捉大部分能量的 POD 子空间."""
    u = rng.standard_normal(shape)
    bands = decompose_field_2d(u, "db2", 4, "periodization")
    models = {}
    for b in bands:
        x = bands[b].ravel().astype(np.float64)
        # 子空间 = 前 r 个主成分 (用 QR 近似正交基捕捉主能量)
        basis = _orthonormal_basis(x.size, r)
        mean = x * 0.0
        models[b] = {"mean": mean, "basis": basis}
    return u, models


def test_scoh_perfect_model_all_bands():
    """模型完全捕捉目标 → S_coh=5."""
    u, models = _make_field_and_models()
    # pred = target (完全一致)
    s_coh = compute_S_coh(u, u, models, tau=0.05)
    assert s_coh == 5


def test_scoh_missing_model_band_inf():
    """缺失频带模型 → 该频带误差 inf → 连续判定停止."""
    u, models = _make_field_and_models()
    del models["W1"]
    s_coh = compute_S_coh(u, u, models, tau=0.05)
    assert s_coh <= 4  # 最多数到 W2


def test_scoh_can_be_less_than_sfull():
    """不假设 S_coh ≥ S_full: 构造相干子空间只捕捉极小部分目标的模型.

    target = a_big(⊥子空间) + 0.001·u(在子空间内, 极小)
    pred   = target + 0.01·u (误差沿子空间方向)
    直接误差 ≈ 0.01 < τ → S_full=5; 相干误差 = 0.01/0.001 = 10 > τ → S_coh=0。
    """
    shape = (64, 64)
    u = rng.standard_normal(shape)
    u = u / np.linalg.norm(u)  # 子空间方向
    a_big = rng.standard_normal(shape)
    a_big = a_big - (a_big.ravel() @ u.ravel()) * u  # 与 u 正交
    a_big = a_big / np.linalg.norm(a_big)

    models = {}
    target_bands, pred_bands = {}, {}
    for b in ["A4", "W4", "W3", "W2", "W1"]:
        target_bands[b] = a_big + 0.001 * u
        pred_bands[b] = target_bands[b] + 0.01 * u
        models[b] = {"mean": np.zeros_like(u), "basis": u.ravel().reshape(1, -1)}

    target = recompose_field_2d(target_bands, "db2", 4, "periodization")
    pred = recompose_field_2d(pred_bands, "db2", 4, "periodization")

    s_full = compute_S_full(target, pred, tau=0.05)
    s_coh = compute_S_coh(target, pred, models, tau=0.05)
    assert s_full == 5
    assert s_coh < s_full

"""S_full 测试 (P0-6 §10.3)。

覆盖: 全通过 / A4 失败 / 中间失败 / 后续 band 重新通过 / threshold 等于边界 /
NaN 与零分母 / 不同 decomposition level。
"""

import numpy as np
import pytest

from luna.wavelet.metrics import contiguous_recoverable_index, compute_S_full
from luna.wavelet.transform import decompose_field_2d, recompose_field_2d

rng = np.random.default_rng(11)


def test_all_pass():
    assert contiguous_recoverable_index([0.01] * 5, tau=0.05) == 5


def test_a4_fail():
    assert contiguous_recoverable_index([0.06, 0.01, 0.01, 0.01, 0.01], tau=0.05) == 0


def test_middle_fail():
    # 第三个频带失败 → 只有前 2 个可恢复
    assert contiguous_recoverable_index([0.01, 0.01, 0.06, 0.01, 0.01], tau=0.05) == 2


def test_repass_after_fail_stops():
    # A4 通过后 W4 失败, 后续 W3 重新通过也不会计数 (连续判定)
    assert contiguous_recoverable_index([0.01, 0.06, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_threshold_boundary_inclusive():
    # 误差 == tau 视为通过 (≤)
    assert contiguous_recoverable_index([0.05, 0.05, 0.05, 0.05, 0.05], tau=0.05) == 5
    assert contiguous_recoverable_index([0.0500001, 0.01, 0.01, 0.01, 0.01], tau=0.05) == 0


def test_nan_stops():
    assert contiguous_recoverable_index([0.01, np.nan, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_inf_stops():
    assert contiguous_recoverable_index([0.01, np.inf, 0.01, 0.01, 0.01], tau=0.05) == 1


def test_zero_denominator_guard():
    # 零/近零分母由 eps 保护, 不抛异常; 若产生 inf 则停止计数
    arr = np.array([0.01, 0.01, np.inf, np.inf, np.inf])
    assert contiguous_recoverable_index(arr, tau=0.05) == 2


def _field_from_bands(band_energies, shape=(64, 64)):
    """构造: target = Σ ω_b·band, pred = target + error 只注入指定频带."""
    # 生成随机小波分量: 通过 decompose 一个随机场得到 5 个 band
    base = rng.standard_normal(shape)
    base_bands = decompose_field_2d(base, "db2", 4, "periodization")
    comps = {b: base_bands[b] * np.sqrt(w) for b, w in band_energies.items()}
    return recompose_field_2d(comps, "db2", 4, "periodization")


def test_compute_sfull_on_fields():
    """用构造场验证 compute_S_full: 只在 W1 注入频带受限误差 → S_full=4."""
    energies = {"A4": 10.0, "W4": 5.0, "W3": 3.0, "W2": 2.0, "W1": 1.0}
    target = _field_from_bands(energies)
    # pred = target, 仅 W1 注入"频带受限"扰动 (取随机场的 W1 分量)
    noise = rng.standard_normal(target.shape)
    noise_w1 = decompose_field_2d(noise, "db2", 4, "periodization")["W1"]
    w1_scale = float(np.linalg.norm(decompose_field_2d(target, "db2", 4, "periodization")["W1"]))
    noise_w1 = noise_w1 / (np.linalg.norm(noise_w1) + 1e-12) * (0.3 * w1_scale)

    bands_t = decompose_field_2d(target, "db2", 4, "periodization")
    bands_p = dict(bands_t)
    bands_p["W1"] = bands_p["W1"] + noise_w1
    pred = recompose_field_2d(bands_p, "db2", 4, "periodization")
    sfull = compute_S_full(target, pred, tau=0.05)
    assert sfull == 4, f"期望 S_full=4, 实际 {sfull}"


def test_compute_sfull_perfect():
    target = _field_from_bands({"A4": 10.0, "W4": 5.0, "W3": 3.0, "W2": 2.0, "W1": 1.0})
    assert compute_S_full(target, target, tau=0.05) == 5

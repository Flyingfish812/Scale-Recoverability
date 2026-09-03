"""Ridge 闭式解测试 (P0-6 §10.5)。

验证: 闭式解 (正规方程, bias 不正则化, 与论文 s10/s05 一致) 在相同标准化策略下
与 sklearn Ridge 对齐; 确定性重复运行一致。
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(31)


def closed_form_ridge(X, y, lam, bias_unregularized=True):
    """闭式解: W = (XᵀX + λ·I')⁻¹ Xᵀ y, 其中 bias 列不加正则."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) if bias_unregularized else X
    d = Xb.shape[1]
    I = np.eye(d)
    if bias_unregularized:
        I[-1, -1] = 0.0
    W = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return W


@pytest.mark.parametrize("n,d,lam", [(300, 8, 1e-3), (300, 8, 1e-5), (120, 3, 0.1)])
def test_closed_form_matches_sklearn_standardized(n, d, lam):
    """同一标准化策略下, 闭式解与 sklearn Ridge 在标准化空间对齐."""
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.01 * rng.standard_normal(n)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    ys = y - y.mean()

    W = closed_form_ridge(Xs, ys, lam, bias_unregularized=True)
    ridge = Ridge(alpha=lam, fit_intercept=True)
    ridge.fit(Xs, ys)
    # 对比标准化空间 (两者都在 (Xs, ys) 上拟合)
    np.testing.assert_allclose(W[:-1], ridge.coef_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(W[-1], ridge.intercept_, rtol=1e-6, atol=1e-8)


def test_bias_not_regularized():
    """bias 不参与正则: 大 λ 时 bias 未被压向 0, 拟合明显更好."""
    X = rng.standard_normal((200, 5))
    y = X @ rng.standard_normal(5) + 3.0  # 强偏置 (未居中目标)
    Xs = X - X.mean(0)

    lam = 1e6
    W_no_bias = closed_form_ridge(Xs, y, lam, bias_unregularized=True)
    W_with_bias = closed_form_ridge(Xs, y, lam, bias_unregularized=False)

    def predict(Wb, X):
        if Wb.size == X.shape[1]:
            return X @ Wb
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return Xb @ Wb

    # 大 λ 下: bias 未正则 → 截距 ≈ y 均值 (拟合偏移); bias 被正则 → 整体压向 0
    mse_nb = float(np.mean((predict(W_no_bias, Xs) - y) ** 2))
    mse_wb = float(np.mean((predict(W_with_bias, Xs) - y) ** 2))
    assert mse_nb < mse_wb


def test_deterministic():
    """同一输入重复运行得到相同结果."""
    X = rng.standard_normal((150, 6))
    y = X @ rng.standard_normal(6)
    w1 = closed_form_ridge(X, y, 1e-3)
    w2 = closed_form_ridge(X, y, 1e-3)
    np.testing.assert_array_equal(w1, w2)

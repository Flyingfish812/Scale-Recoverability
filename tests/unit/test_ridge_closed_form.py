"""Ridge closed-form solution tests.

Verifies: the closed form (normal equations, unregularized bias) matches sklearn Ridge under the same standardization; repeated runs are deterministic.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(31)


def closed_form_ridge(X, y, lam, bias_unregularized=True):
    """Closed form: W = (XᵀX + λ·I')⁻¹ Xᵀ y, where the bias column is not regularized."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) if bias_unregularized else X
    d = Xb.shape[1]
    I = np.eye(d)
    if bias_unregularized:
        I[-1, -1] = 0.0
    W = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return W


@pytest.mark.parametrize("n,d,lam", [(300, 8, 1e-3), (300, 8, 1e-5), (120, 3, 0.1)])
def test_closed_form_matches_sklearn_standardized(n, d, lam):
    """Under the same standardization, the closed form matches sklearn Ridge."""
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.01 * rng.standard_normal(n)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    ys = y - y.mean()

    W = closed_form_ridge(Xs, ys, lam, bias_unregularized=True)
    ridge = Ridge(alpha=lam, fit_intercept=True)
    ridge.fit(Xs, ys)
    # Compare in standardized space (both are fitted on (Xs, ys))
    np.testing.assert_allclose(W[:-1], ridge.coef_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(W[-1], ridge.intercept_, rtol=1e-6, atol=1e-8)


def test_bias_not_regularized():
    """Bias is not regularized: at large λ it is not shrunk to 0, giving a better fit."""
    X = rng.standard_normal((200, 5))
    y = X @ rng.standard_normal(5) + 3.0  # strong bias (uncentered target)
    Xs = X - X.mean(0)

    lam = 1e6
    W_no_bias = closed_form_ridge(Xs, y, lam, bias_unregularized=True)
    W_with_bias = closed_form_ridge(Xs, y, lam, bias_unregularized=False)

    def predict(Wb, X):
        if Wb.size == X.shape[1]:
            return X @ Wb
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return Xb @ Wb

    # Large λ: unregularized bias → intercept ≈ mean(y); regularized bias → shrunk to 0
    mse_nb = float(np.mean((predict(W_no_bias, Xs) - y) ** 2))
    mse_wb = float(np.mean((predict(W_with_bias, Xs) - y) ** 2))
    assert mse_nb < mse_wb


def test_deterministic():
    """Repeated runs on the same input give identical results."""
    X = rng.standard_normal((150, 6))
    y = X @ rng.standard_normal(6)
    w1 = closed_form_ridge(X, y, 1e-3)
    w2 = closed_form_ridge(X, y, 1e-3)
    np.testing.assert_array_equal(w1, w2)

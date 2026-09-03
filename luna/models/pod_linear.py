"""
POD Linear Regression (Ridge) — linear baseline for POD coefficient prediction.

Migrated from models/pod_linear.py
"""

from __future__ import annotations

import torch
from torch import nn


class PODLinearRegression(nn.Module):
    """Linear (Ridge) regression for predicting POD coefficients
    from sparse spatial observations.

    Maps a flattened observation vector (n_obs * C) to POD coefficients (R).
    L2 regularization is applied via weight decay in the optimizer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bool(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"bias={self.linear.bias is not None}"
        )


def build_pod_linear_model(
    *,
    n_obs: int,
    n_channels: int,
    n_modes: int,
    bias: bool = True,
) -> PODLinearRegression:
    """Convenience factory: infer in_features from observation count and channels."""
    return PODLinearRegression(
        in_features=int(n_obs) * int(n_channels),
        out_features=int(n_modes),
        bias=bool(bias),
    )

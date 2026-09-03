"""
POD MLP — nonlinear multi-layer perceptron for POD coefficient prediction.

Migrated from models/pod_mlp.py
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PODMLP(nn.Module):
    """Multi-layer perceptron for predicting POD coefficients
    from sparse spatial observations.

    Maps a flattened observation vector (n_obs * C) through a stack of
    Linear → ReLU → (Dropout)? layers, then outputs POD coefficients (R).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        out_features: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act: type[nn.Module]
        act_name = str(activation).strip().lower()
        if act_name == "relu":
            act = nn.ReLU
        elif act_name == "tanh":
            act = nn.Tanh
        elif act_name == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Use relu, tanh, or gelu.")

        hidden = list(int(v) for v in hidden_sizes)
        layers: list[nn.Module] = []
        prev = int(in_features)

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act(inplace=True))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h

        layers.append(nn.Linear(prev, int(out_features)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_pod_mlp_model(
    *,
    n_obs: int,
    n_channels: int,
    n_modes: int,
    hidden_sizes: Sequence[int] = (256, 256),
    dropout: float = 0.0,
    activation: str = "relu",
) -> PODMLP:
    """Convenience factory: infer in_features from observation count and channels."""
    return PODMLP(
        in_features=int(n_obs) * int(n_channels),
        hidden_sizes=hidden_sizes,
        out_features=int(n_modes),
        dropout=float(dropout),
        activation=activation,
    )

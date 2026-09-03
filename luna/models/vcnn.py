"""
VCNN: fully-convolutional field reconstruction network.

Migrated from models/vcnn.py — unchanged model architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# ── Loss functions ─────────────────────────────────────────────────

def _weighted_error(
    error: torch.Tensor,
    obs_mask: Optional[torch.Tensor],
    obs_weight: float = 1.0,
) -> torch.Tensor:
    if obs_mask is None:
        return error
    weight = obs_mask * float(obs_weight) - (obs_mask - 1.0)
    return error * weight


class FieldL1Loss(nn.Module):
    """Weighted L1 (MAE) loss on the spatial field."""

    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        error = torch.abs(target - pred)
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class FieldL2Loss(nn.Module):
    """Weighted L2 (MSE) loss on the spatial field."""

    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        error = (target - pred) ** 2
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class RelativeL2Loss(nn.Module):
    """Relative L2 loss: ‖pred − target‖₂ / ‖target‖₂."""

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = target - pred
        target_eval = target
        if obs_mask is not None:
            diff = diff * obs_mask
            target_eval = target_eval * obs_mask
        num = torch.sum(diff ** 2, dim=(1, 2, 3))
        den = torch.sum(target_eval ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        return torch.mean(torch.sqrt(num / den))


def get_field_loss(loss_type: str = "mae", obs_weight: float = 1.0) -> nn.Module:
    name = str(loss_type or "mae").strip().lower()
    if name == "mae":
        return FieldL1Loss(obs_weight=obs_weight)
    if name == "mse":
        return FieldL2Loss(obs_weight=obs_weight)
    if name == "l2norm":
        return RelativeL2Loss()
    raise ValueError(f"Unsupported loss_type='{loss_type}'. Use: mae, mse, l2norm")


# ── VCNN model ─────────────────────────────────────────────────────

class VCNN(nn.Module):
    """Minimal fully-convolutional field reconstructor.

    Architecture: Conv → ReLU → (Conv → ReLU) × (L−2) → Conv
    All convolutions use same padding to preserve spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 8,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if int(num_layers) < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        padding = int(kernel_size) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        for _ in range(int(num_layers) - 2):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_vcnn_from_config(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 48,
    num_layers: int = 8,
    kernel_size: int = 7,
) -> VCNN:
    """Factory: build VCNN from configuration parameters."""
    return VCNN(
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
        num_layers=int(num_layers),
        kernel_size=int(kernel_size),
    )

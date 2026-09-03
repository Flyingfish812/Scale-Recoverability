"""
POD model (Ridge/MLP) training for POD coefficient prediction.

Extracted from run_pod_model_sweep.py — provides clean training API
using luna.models and luna.pod.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from luna.data.masks import load_mask_csv


def train_pod_model(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    batch_size: int = 32,
    num_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a POD coefficient prediction model (Ridge or MLP).

    Args:
        model: A PODLinearRegression or PODMLP instance.
        train_dataset: Training dataset yielding (obs_vec, pod_coeff).
        val_dataset: Optional validation dataset.
        batch_size: Mini-batch size.
        num_epochs: Maximum number of epochs.
        lr: Learning rate.
        weight_decay: L2 regularization strength.
        device: 'auto', 'cuda', or 'cpu'.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict with keys: best_val_loss, best_epoch, epochs_ran, train_losses, val_losses.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        scheduler.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item() * x.size(0)
            avg_val_loss = val_loss / len(val_dataset)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
        else:
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_epoch = epoch

        if verbose and epoch % max(1, num_epochs // 10) == 0:
            val_str = f" val={val_losses[-1]:.6f}" if val_losses else ""
            print(f"  epoch {epoch:4d}/{num_epochs}  train={avg_train_loss:.6f}{val_str}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_ran": num_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def run_pod_model_sweep(
    *,
    data_array_path: str,
    pod_bundle_path: str,
    out_dir: str,
    mask_paths: list[str],
    mask_nums: list[int],
    noise_sigmas: list[float],
    n_modes: int = 128,
    ridge_alpha: float = 1.0,
    mlp_hidden_sizes: tuple[int, ...] = (256, 256),
    batch_size: int = 32,
    num_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Run a full POD model sweep (Ridge + MLP) across masks and noise levels.

    This is the refactored entry point replacing run_pod_model_sweep.py.

    Returns:
        Summary dict with all test results.
    """
    from pathlib import Path

    from luna.data.io import load_npy, load_npz
    from luna.models.pod_linear import build_pod_linear_model
    from luna.models.pod_mlp import build_pod_mlp_model

    # Load data
    fields = load_npy(data_array_path)  # (T, H, W, C)
    pod = load_npz(pod_bundle_path)

    T, H, W, C = fields.shape
    basis = np.asarray(pod["basis"], dtype=np.float64)[:n_modes, :]
    mean = np.asarray(pod["mean_flat"], dtype=np.float64).ravel()
    coeffs = np.asarray(pod["coefficients"], dtype=np.float32)[:, :n_modes]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"config": {}, "models": []}
    # ... (full sweep implementation would go here)
    # For now, this is a skeleton showing the API design.

    return results

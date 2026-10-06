"""Training loop for the convolutional estimator.

Trains the observation-to-field network on a fixed sensor mask and writes, for every test noise level, the reconstructions consumed by the statistics stage: ``test_raw.npz`` holding ``target_nchw`` (ground truth) and ``output_nchw`` (reconstruction) in normalised units, with the normalisation constants stored in the checkpoint next to the run.

Entry point: ``applications/pipelines/03_train_estimators.py``.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from luna.models.vcnn import RelativeL2Loss, VCNN, get_field_loss
from features.sensors.observations import (
    build_nearest_seed_index,
    build_observation_feature,
    compute_channelwise_mean_std,
    denormalize_field_nchw,
)
from features.training.vcnn_config import CheckpointConfig, TrainConfig, VcnnConfig


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def resolve_torch_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    name = str(device).strip().lower()
    if name in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}' but CUDA is not available")
    return str(device)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    progress: float,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    total = max(1, int(total_epochs))
    warmup = max(0, int(warmup_epochs))
    current = float(progress)
    if warmup > 0 and current < warmup:
        lr = float(base_lr) * current / float(warmup)
    else:
        cosine_total = max(1.0, float(total - warmup))
        cosine_progress = min(max((current - warmup) / cosine_total, 0.0), 1.0)
        lr = float(min_lr) + (float(base_lr) - float(min_lr)) * 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    norm_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name.endswith("bias") or isinstance(module, norm_modules):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class ObservationFeatureDataset(Dataset):
    def __init__(
        self,
        fields_thwc: np.ndarray,
        *,
        mask_hw: np.ndarray,
        noise_sigma: float,
        representation: str,
        include_mask_channel: bool,
        normalize_mean_std: bool,
        seed: int | None,
    ) -> None:
        super().__init__()
        fields = np.asarray(fields_thwc, dtype=np.float32)
        if fields.ndim != 4:
            raise ValueError(f"fields_thwc must be [T,H,W,C], got {fields.shape}")

        self.fields_thwc = fields
        self.mask_hw = np.asarray(mask_hw, dtype=bool)
        if self.mask_hw.shape != tuple(fields.shape[1:3]):
            raise ValueError(f"mask shape {self.mask_hw.shape} != spatial shape {fields.shape[1:3]}")

        self.noise_sigma = float(noise_sigma)
        self.representation = str(representation)
        self.include_mask_channel = bool(include_mask_channel)
        self.seed = None if seed is None else int(seed)
        self.nearest_index_hw = build_nearest_seed_index(self.mask_hw)
        if bool(normalize_mean_std):
            self.norm_mean_c, self.norm_std_c = compute_channelwise_mean_std(self.fields_thwc)
        else:
            self.norm_mean_c = None
            self.norm_std_c = None

    def __len__(self) -> int:
        return int(self.fields_thwc.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_index = int(index)
        noise_seed = None if self.seed is None else self.seed + item_index
        feature, target, obs_mask = build_observation_feature(
            self.fields_thwc[item_index],
            mask_hw=self.mask_hw,
            nearest_index_hw=self.nearest_index_hw,
            noise_sigma=self.noise_sigma,
            representation=self.representation,
            include_mask_channel=self.include_mask_channel,
            norm_mean_c=self.norm_mean_c,
            norm_std_c=self.norm_std_c,
            noise_seed=noise_seed,
        )
        return (
            torch.from_numpy(feature.astype(np.float32, copy=True)),
            torch.from_numpy(target.astype(np.float32, copy=True)),
            torch.from_numpy(obs_mask.astype(np.float32, copy=True)),
        )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    train_info: dict[str, Any],
    mask_hw: np.ndarray,
    dataset: ObservationFeatureDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    model_config: VcnnConfig,
    train_config: TrainConfig,
    sweep_meta: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "train_info": train_info,
        "mask_hw": np.asarray(mask_hw, dtype=np.uint8),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "val_indices": np.asarray(val_indices, dtype=np.int64),
        "representation": dataset.representation,
        "include_mask_channel": dataset.include_mask_channel,
        "normalize_mean_std": dataset.norm_mean_c is not None,
        "norm_mean_c": dataset.norm_mean_c,
        "norm_std_c": dataset.norm_std_c,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "sweep_meta": sweep_meta,
    }
    torch.save(payload, checkpoint_path)


def train_vcnn(
    fields_thwc: np.ndarray,
    *,
    mask_hw: np.ndarray,
    noise_sigma: float,
    model_config: VcnnConfig,
    train_config: TrainConfig,
    checkpoint_config: CheckpointConfig | None = None,
    sweep_meta: dict[str, Any] | None = None,
    dataset_builder: Callable[..., Dataset] | None = None,
    split_indices: dict[str, np.ndarray] | None = None,
    artifact_saver: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    set_global_seed(train_config.seed)

    builder = dataset_builder or ObservationFeatureDataset
    dataset = builder(
        fields_thwc,
        mask_hw=np.asarray(mask_hw, dtype=bool),
        noise_sigma=float(noise_sigma),
        representation=model_config.input_representation,
        include_mask_channel=bool(model_config.include_mask_channel),
        normalize_mean_std=bool(train_config.normalize_mean_std),
        seed=train_config.seed,
    )

    total = int(len(dataset))
    if total < 3:
        raise ValueError(f"Dataset must contain at least 3 samples, got {total}")

    raw_n_test = int(round(float(total) * float(train_config.test_ratio)))
    if float(train_config.test_ratio) > 0.0:
        n_test = min(max(1, raw_n_test), total - 2)
    else:
        n_test = 1

    remain_after_test = total - n_test
    raw_n_val = int(round(float(remain_after_test) * float(train_config.val_ratio)))
    if float(train_config.val_ratio) > 0.0:
        n_val = min(max(1, raw_n_val), remain_after_test - 1)
    else:
        n_val = 1

    n_train = total - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}, total={total}. "
            "Please reduce val_ratio/test_ratio."
        )

    if split_indices is None:
        generator = None if train_config.seed is None else torch.Generator().manual_seed(int(train_config.seed))
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
        train_indices = np.asarray(getattr(train_ds, "indices", np.arange(n_train)), dtype=np.int64)
        val_indices = np.asarray(getattr(val_ds, "indices", np.arange(n_val)), dtype=np.int64)
        test_indices = np.asarray(getattr(test_ds, "indices", np.arange(n_test)), dtype=np.int64)
    else:
        # Explicit split override (blocked/contiguous holdout sensitivity):
        # dataset positions are snapshot indices, as for the POD-coefficient models.
        train_indices = np.asarray(sorted(int(i) for i in split_indices["train"]), dtype=np.int64)
        val_indices = np.asarray(sorted(int(i) for i in split_indices["val"]), dtype=np.int64)
        test_indices = np.asarray(sorted(int(i) for i in split_indices["test"]), dtype=np.int64)
        train_ds = torch.utils.data.Subset(dataset, train_indices.tolist())
        val_ds = torch.utils.data.Subset(dataset, val_indices.tolist())
        test_ds = torch.utils.data.Subset(dataset, test_indices.tolist())

    loader_generator = None if train_config.seed is None else torch.Generator().manual_seed(int(train_config.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_config.batch_size),
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_config.batch_size),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_config.batch_size),
        shuffle=False,
        drop_last=False,
    )

    feature_sample, target_sample, _ = dataset[0]
    device_name = resolve_torch_device(train_config.device)
    model = VCNN(
        in_channels=int(feature_sample.shape[0]),
        out_channels=int(target_sample.shape[0]),
        hidden_channels=int(model_config.hidden_channels),
        num_layers=int(model_config.num_layers),
        kernel_size=int(model_config.kernel_size),
    ).to(device_name)

    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, weight_decay=float(train_config.weight_decay)),
        lr=float(train_config.lr),
        betas=(0.9, 0.95),
    )
    train_loss_fn = get_field_loss(train_config.loss_type, obs_weight=train_config.obs_weight)
    monitor_loss_fn = RelativeL2Loss()

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    last_lr = float(train_config.lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    stopped_early = False
    checkpoint_epochs = (
        {int(epoch) for epoch in checkpoint_config.save_epochs if int(epoch) > 0}
        if checkpoint_config is not None
        else set()
    )

    print(
        f"[train_vcnn] N={total} train={n_train} val={n_val} test={n_test} device={device_name} "
        f"mask_obs={int(np.asarray(mask_hw, dtype=bool).sum())} noise_sigma={float(noise_sigma):.4e}"
    )

    train_start_time = time.perf_counter()

    for epoch in range(1, int(train_config.num_epochs) + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        num_train_batches = max(1, len(train_loader))

        for batch_idx, batch in enumerate(train_loader):
            if train_config.max_train_batches is not None and batch_idx >= int(train_config.max_train_batches):
                break
            if train_config.use_cosine_schedule:
                progress = float(epoch - 1) + float(batch_idx) / float(num_train_batches)
                last_lr = adjust_learning_rate(
                    optimizer,
                    progress=progress,
                    base_lr=float(train_config.lr),
                    min_lr=float(train_config.min_lr),
                    total_epochs=int(train_config.num_epochs),
                    warmup_epochs=int(train_config.warmup_epochs),
                )

            feature, target, obs_mask = batch
            feature = feature.to(device_name)
            target = target.to(device_name)
            obs_mask = obs_mask.to(device_name)

            optimizer.zero_grad()
            pred = model(feature)
            loss = train_loss_fn(target, pred, obs_mask)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            n_train_batches += 1

        if not train_config.use_cosine_schedule:
            last_lr = float(optimizer.param_groups[0]["lr"])

        avg_train_loss = total_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_monitor = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if train_config.max_val_batches is not None and batch_idx >= int(train_config.max_val_batches):
                    break
                feature, target, obs_mask = batch
                feature = feature.to(device_name)
                target = target.to(device_name)
                pred = model(feature)

                if dataset.norm_mean_c is not None and dataset.norm_std_c is not None:
                    pred_eval = torch.as_tensor(
                        denormalize_field_nchw(pred.detach().cpu().numpy(), dataset.norm_mean_c, dataset.norm_std_c),
                        dtype=pred.dtype,
                        device=device_name,
                    )
                    target_eval = torch.as_tensor(
                        denormalize_field_nchw(target.detach().cpu().numpy(), dataset.norm_mean_c, dataset.norm_std_c),
                        dtype=target.dtype,
                        device=device_name,
                    )
                else:
                    pred_eval = pred
                    target_eval = target
                monitor = monitor_loss_fn(target_eval, pred_eval, None)
                total_val_monitor += float(monitor.item())
                n_val_batches += 1

        avg_val = total_val_monitor / max(1, n_val_batches)
        val_losses.append(avg_val)

        if epoch == 1 or epoch == int(train_config.num_epochs) or epoch % max(1, int(train_config.progress_every)) == 0:
            elapsed_seconds = time.perf_counter() - train_start_time
            progress = float(epoch) / max(1.0, float(train_config.num_epochs))
            eta_seconds = (elapsed_seconds / max(progress, 1e-12)) - elapsed_seconds
            print(
                f"[train_vcnn] epoch {epoch:03d}/{int(train_config.num_epochs):03d} "
                f"train={avg_train_loss:.4e} val={avg_val:.4e} lr={last_lr:.3e} "
                f"elapsed={format_duration(elapsed_seconds)} eta={format_duration(eta_seconds)}"
            )

        improved = (best_val - avg_val) > float(train_config.early_min_delta)
        if improved:
            best_val = float(avg_val)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            if checkpoint_config is not None and not checkpoint_config.save_best_only:
                save_checkpoint(
                    checkpoint_config.out_dir / f"{checkpoint_config.prefix}_epoch{epoch:03d}.pt",
                    model=model,
                    train_info={"epoch": int(epoch), "val_loss": float(avg_val)},
                    mask_hw=np.asarray(mask_hw, dtype=bool),
                    dataset=dataset,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    model_config=model_config,
                    train_config=train_config,
                    sweep_meta=sweep_meta or {},
                )
        elif train_config.early_stop and epoch >= int(train_config.early_warmup):
            patience += 1
            if patience >= int(train_config.early_patience):
                stopped_early = True
                print(f"[train_vcnn] early stop at epoch {epoch}")
                break

        if checkpoint_config is not None and epoch in checkpoint_epochs:
            save_checkpoint(
                checkpoint_config.out_dir / f"{checkpoint_config.prefix}_epoch{epoch:04d}.pt",
                model=model,
                train_info={
                    "epoch": int(epoch),
                    "epochs_ran": int(epoch),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                    "lr": float(last_lr),
                    "in_channels": int(feature_sample.shape[0]),
                    "out_channels": int(target_sample.shape[0]),
                    "hidden_channels": int(model_config.hidden_channels),
                    "num_layers": int(model_config.num_layers),
                    "kernel_size": int(model_config.kernel_size),
                    "normalize_mean_std": bool(dataset.norm_mean_c is not None),
                },
                mask_hw=np.asarray(mask_hw, dtype=bool),
                dataset=dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                model_config=model_config,
                train_config=train_config,
                sweep_meta=sweep_meta or {},
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(train_losses)),
        "stopped_early": bool(stopped_early),
        "batch_size": int(train_config.batch_size),
        "lr": float(train_config.lr),
        "last_lr": float(last_lr),
        "device": str(device_name),
        "loss_type": str(train_config.loss_type),
        "obs_weight": float(train_config.obs_weight),
        "noise_sigma": float(noise_sigma),
        "input_representation": str(model_config.input_representation),
        "include_mask_channel": bool(model_config.include_mask_channel),
        "normalize_mean_std": bool(dataset.norm_mean_c is not None),
        "in_channels": int(feature_sample.shape[0]),
        "out_channels": int(target_sample.shape[0]),
        "hidden_channels": int(model_config.hidden_channels),
        "num_layers": int(model_config.num_layers),
        "kernel_size": int(model_config.kernel_size),
        "param_count": int(sum(param.numel() for param in model.parameters())),
    }
    artifacts = {
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "mask_hw": np.asarray(mask_hw, dtype=bool),
    }

    if checkpoint_config is not None:
        if checkpoint_config.save_last:
            save_checkpoint(
                checkpoint_config.out_dir / f"{checkpoint_config.prefix}_last.pt",
                model=model,
                train_info=train_info,
                mask_hw=np.asarray(mask_hw, dtype=bool),
                dataset=dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                model_config=model_config,
                train_config=train_config,
                sweep_meta=sweep_meta or {},
            )
        if checkpoint_config.save_best_only:
            save_checkpoint(
                checkpoint_config.out_dir / f"{checkpoint_config.prefix}_best.pt",
                model=model,
                train_info=train_info,
                mask_hw=np.asarray(mask_hw, dtype=bool),
                dataset=dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                model_config=model_config,
                train_config=train_config,
                sweep_meta=sweep_meta or {},
            )
        save_json(
            checkpoint_config.out_dir / f"{checkpoint_config.prefix}_summary.json",
            {
                "train_info": train_info,
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "mask_obs_count": int(np.asarray(mask_hw, dtype=bool).sum()),
                "mask_density": float(np.asarray(mask_hw, dtype=bool).mean()),
                "sweep_meta": sweep_meta or {},
            },
        )

    if artifact_saver is not None:
        artifact_saver(
            {
                "model": model,
                "train_info": train_info,
                "artifacts": artifacts,
                "sweep_meta": sweep_meta or {},
            }
        )

    return model, train_info, artifacts

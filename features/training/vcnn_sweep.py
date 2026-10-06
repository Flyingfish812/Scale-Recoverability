"""Sensor-mask sweep for the convolutional estimator.

Trains one run per (data source, sensor mask, random seed) and evaluates it at a fixed set of test noise levels. Each run writes its checkpoint and the reconstructions under the sweep root, which is where the statistics stage reads them from.

Entry point: ``applications/pipelines/03_train_estimators.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from features.sensors.observations import (
    generate_grid_mask_hw,
    generate_random_mask_hw,
    load_mask_csv,
)
from features.training.vcnn_config import (
    CheckpointConfig,
    DataSourceConfig,
    MaskConfig,
    SweepConfig,
    TrainConfig,
    VcnnConfig,
)
from features.training.vcnn_trainer import train_vcnn


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_field_array(config: DataSourceConfig) -> np.ndarray:
    mmap_mode = "r" if bool(config.mmap) else None
    array = np.load(Path(config.array_path), mmap_mode=mmap_mode)
    if array.ndim != 4:
        raise ValueError(f"Expected data array [T,H,W,C], got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def encode_value(value: float) -> int:
    return int(round(float(value) * 10000.0))


def encode_mask_tag(mask_meta: dict[str, Any]) -> str:
    mask_num = mask_meta.get("mask_num")
    if mask_num is None:
        return f"p{encode_value(mask_meta['mask_rate']):04d}"
    return f"n{int(mask_num):04d}"


def build_mask_for_case(
    *,
    height: int,
    width: int,
    base_mask_config: MaskConfig,
    mask_rate: float | None,
    mask_num: int | None,
    mask_seed: int | None,
    mask_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mask_path is not None:
        mask_hw = load_mask_csv(mask_path, height=height, width=width)
        return mask_hw, {
            "mode": "mask_csv",
            "mask_path": str(mask_path),
            "mask_rate": float(np.asarray(mask_hw, dtype=bool).mean()),
            "mask_num": int(np.asarray(mask_hw, dtype=bool).sum()),
            "mask_seed": None if mask_seed is None else int(mask_seed),
        }

    mode = str(base_mask_config.mode).strip().lower()
    if mode == "random":
        mask_hw = generate_random_mask_hw(
            height,
            width,
            mask_rate=mask_rate,
            mask_num=mask_num,
            seed=mask_seed,
        )
    elif mode == "grid":
        mask_hw = generate_grid_mask_hw(
            height,
            width,
            mask_rate=mask_rate,
            mask_num=mask_num,
            seed=mask_seed,
        )
    else:
        raise ValueError(f"Unsupported mask mode for VCNN sweep: {base_mask_config.mode}")

    return mask_hw, {
        "mode": mode,
        "mask_path": None,
        "mask_rate": float(np.asarray(mask_hw, dtype=bool).mean()),
        "mask_num": int(np.asarray(mask_hw, dtype=bool).sum()),
        "mask_seed": None if mask_seed is None else int(mask_seed),
    }


def save_test_raw_artifacts(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    out_dir: Path,
    noise_sigma: float,
    test_indices: np.ndarray,
    device_name: str,
    train_info: dict[str, Any],
    mask_meta: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    input_list: list[np.ndarray] = []
    output_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    obs_mask_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            feature, target, obs_mask = batch
            pred = model(feature.to(device_name)).detach().cpu().numpy().astype(np.float32, copy=False)
            input_list.append(feature.detach().cpu().numpy().astype(np.float32, copy=False))
            output_list.append(pred)
            target_list.append(target.detach().cpu().numpy().astype(np.float32, copy=False))
            obs_mask_list.append(obs_mask.detach().cpu().numpy().astype(np.float32, copy=False))

    input_all = np.concatenate(input_list, axis=0) if input_list else np.zeros((0,), dtype=np.float32)
    output_all = np.concatenate(output_list, axis=0) if output_list else np.zeros((0,), dtype=np.float32)
    target_all = np.concatenate(target_list, axis=0) if target_list else np.zeros((0,), dtype=np.float32)
    obs_mask_all = np.concatenate(obs_mask_list, axis=0) if obs_mask_list else np.zeros((0,), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "test_raw.npz"
    np.savez_compressed(
        npz_path,
        input_nchw=input_all,
        output_nchw=output_all,
        target_nchw=target_all,
        obs_mask_nchw=obs_mask_all,
        test_indices=np.asarray(test_indices, dtype=np.int64),
        noise_sigma=np.asarray(float(noise_sigma), dtype=np.float32),
    )

    meta = {
        "schema_version": "luna.vcnn.test_raw.v1",
        "npz_path": str(npz_path),
        "noise_sigma": float(noise_sigma),
        "test_count": int(output_all.shape[0]) if output_all.ndim >= 1 else 0,
        "input_shape": list(input_all.shape),
        "output_shape": list(output_all.shape),
        "target_shape": list(target_all.shape),
        "obs_mask_shape": list(obs_mask_all.shape),
        "test_indices": np.asarray(test_indices, dtype=np.int64).tolist(),
        "mask_meta": mask_meta,
        "model_device": str(device_name),
        "train_best_epoch": int(train_info.get("best_epoch", 0)),
        "train_best_val_loss": float(train_info.get("best_val_loss", 0.0)),
    }
    (out_dir / "test_raw_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_vcnn_sweep(
    *,
    data_config: DataSourceConfig,
    mask_config: MaskConfig,
    model_config: VcnnConfig,
    train_config: TrainConfig,
    sweep_config: SweepConfig,
    checkpoint_root: Path,
    checkpoint_options: CheckpointConfig | None = None,
    split_indices: dict[str, np.ndarray] | None = None,
    dataset_builder: Callable[..., Any] | None = None,
    artifact_saver: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    fields_thwc = load_field_array(data_config)
    _, height, width, _ = fields_thwc.shape

    mask_rates = tuple(float(value) for value in sweep_config.mask_rates)
    mask_nums = tuple(int(value) for value in sweep_config.mask_nums)
    fallback_noise_sigmas = tuple(float(value) for value in sweep_config.noise_sigmas)
    test_noise_sigmas = tuple(float(value) for value in sweep_config.test_noise_sigmas)
    if not test_noise_sigmas and fallback_noise_sigmas:
        test_noise_sigmas = fallback_noise_sigmas
    train_noise_sigma = float(sweep_config.train_noise_sigma)
    mask_seeds = tuple(int(value) for value in sweep_config.mask_seeds)
    mask_paths = tuple(Path(value) for value in sweep_config.mask_paths)

    if bool(mask_rates) and bool(mask_nums):
        raise ValueError("Use either sweep.mask_rates or sweep.mask_nums, not both")
    if not bool(mask_rates) and not bool(mask_nums) and not bool(mask_paths):
        raise ValueError("Sweep must define mask_rates, mask_nums, or mask_paths")
    if not bool(test_noise_sigmas):
        raise ValueError("Sweep must define at least one test_noise_sigma")

    model_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if mask_paths:
        if mask_seeds:
            mask_specs = [(None, None, seed, path) for path in mask_paths for seed in mask_seeds]
        else:
            mask_specs = [(None, None, None, path) for path in mask_paths]
    elif mask_rates:
        mask_specs = [(rate, None, seed, None) for rate in mask_rates for seed in mask_seeds]
    else:
        mask_specs = [(None, count, seed, None) for count in mask_nums for seed in mask_seeds]

    total_model_cases = len(mask_specs)
    total_test_cases = len(mask_specs) * len(test_noise_sigmas)
    model_case_index = 0
    test_case_index = 0
    test_t0 = time.perf_counter()

    for mask_rate, mask_num, mask_seed, mask_path in mask_specs:
        model_case_index += 1
        mask_hw, mask_meta = build_mask_for_case(
            height=height,
            width=width,
            base_mask_config=mask_config,
            mask_rate=mask_rate,
            mask_num=mask_num,
            mask_seed=mask_seed,
            mask_path=mask_path,
        )
        model_name = f"vcnn_{encode_mask_tag(mask_meta)}"
        if mask_meta["mask_seed"] is not None:
            model_name += f"_seed{int(mask_meta['mask_seed']):03d}"
        if mask_meta["mode"] == "grid":
            model_name += "_grid"
        if mask_meta["mask_path"] is not None:
            model_name += "_custom"

        model_dir = checkpoint_root / model_name
        print(
            f"[run_vcnn_sweep][train] ({model_case_index}/{total_model_cases}) {model_name} "
            f"mask_rate={mask_meta['mask_rate']:.6f} mask_num={mask_meta['mask_num']} train_noise_sigma={train_noise_sigma:.4e}"
        )

        effective_checkpoint = CheckpointConfig(
            out_dir=model_dir,
            save_best_only=True if checkpoint_options is None else bool(checkpoint_options.save_best_only),
            save_last=True if checkpoint_options is None else bool(checkpoint_options.save_last),
            prefix="vcnn" if checkpoint_options is None else str(checkpoint_options.prefix),
            save_epochs=tuple() if checkpoint_options is None else tuple(int(v) for v in checkpoint_options.save_epochs),
        )

        trained_model, train_info, train_artifacts = train_vcnn(
            fields_thwc,
            mask_hw=mask_hw,
            noise_sigma=train_noise_sigma,
            model_config=model_config,
            train_config=train_config,
            checkpoint_config=effective_checkpoint,
            sweep_meta={
                "model_name": model_name,
                "mask_meta": mask_meta,
                "train_noise_sigma": float(train_noise_sigma),
                "data_config": {"array_path": str(data_config.array_path), "mmap": bool(data_config.mmap)},
            },
            dataset_builder=dataset_builder,
            split_indices=split_indices,
            artifact_saver=artifact_saver,
        )

        model_record = {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "mask_meta": mask_meta,
            "train_noise_sigma": float(train_noise_sigma),
            "best_val_loss": float(train_info["best_val_loss"]),
            "best_epoch": int(train_info["best_epoch"]),
            "epochs_ran": int(train_info["epochs_ran"]),
            "stopped_early": bool(train_info["stopped_early"]),
        }
        model_cases.append(model_record)

        dataset = train_artifacts["dataset"]
        test_indices = np.asarray(train_artifacts["test_indices"], dtype=np.int64)
        for noise_sigma in test_noise_sigmas:
            test_case_index += 1
            sigma_code = encode_value(noise_sigma)
            test_name = f"s{sigma_code:04d}"
            test_dir = model_dir / "tests" / test_name

            elapsed = time.perf_counter() - test_t0
            progress = float(test_case_index) / max(1.0, float(total_test_cases))
            eta = (elapsed / max(progress, 1e-12)) - elapsed

            print(
                f"[run_vcnn_sweep][test] ({test_case_index}/{total_test_cases}) {model_name}/{test_name} "
                f"noise_sigma={float(noise_sigma):.4e} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )

            eval_dataset = type(dataset)(
                fields_thwc,
                mask_hw=np.asarray(mask_hw, dtype=bool),
                noise_sigma=float(noise_sigma),
                representation=model_config.input_representation,
                include_mask_channel=bool(model_config.include_mask_channel),
                normalize_mean_std=bool(train_config.normalize_mean_std),
                seed=train_config.seed,
            )
            eval_subset = Subset(eval_dataset, test_indices.tolist())
            eval_loader = DataLoader(
                eval_subset,
                batch_size=int(train_config.batch_size),
                shuffle=False,
                drop_last=False,
            )

            test_meta = save_test_raw_artifacts(
                model=trained_model,
                loader=eval_loader,
                out_dir=test_dir,
                noise_sigma=float(noise_sigma),
                test_indices=test_indices,
                device_name=str(train_info["device"]),
                train_info=train_info,
                mask_meta=mask_meta,
            )
            test_cases.append(
                {
                    "model_name": model_name,
                    "test_name": test_name,
                    "test_dir": str(test_dir),
                    "noise_sigma": float(noise_sigma),
                    "test_count": int(test_meta["test_count"]),
                    "npz_path": str(test_dir / "test_raw.npz"),
                }
            )

    summary = {
        "schema_version": "luna.vcnn.sweep.v2",
        "data_config": {"array_path": str(data_config.array_path), "mmap": bool(data_config.mmap)},
        "mask_config": asdict(mask_config),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "checkpoint_config": None
        if checkpoint_options is None
        else {
            "save_best_only": bool(checkpoint_options.save_best_only),
            "save_last": bool(checkpoint_options.save_last),
            "prefix": str(checkpoint_options.prefix),
            "save_epochs": [int(v) for v in checkpoint_options.save_epochs],
        },
        "sweep_config": {
            "mask_rates": list(mask_rates),
            "mask_nums": list(mask_nums),
            "noise_sigmas": list(fallback_noise_sigmas),
            "train_noise_sigma": float(train_noise_sigma),
            "test_noise_sigmas": list(test_noise_sigmas),
            "mask_seeds": list(mask_seeds),
            "mask_paths": [str(path) for path in mask_paths],
        },
        "model_count": int(len(model_cases)),
        "test_case_count": int(len(test_cases)),
        "model_cases": model_cases,
        "test_cases": test_cases,
    }
    (checkpoint_root / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
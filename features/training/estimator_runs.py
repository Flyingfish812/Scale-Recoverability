"""Access to trained estimator runs and their physical-unit convention.

Every training run writes one ``test_raw.npz`` per test noise level under a path that encodes estimator, sensor count, training seed and noise level. This module owns that layout so that the statistics producers do not each re-derive it.

The two estimator families also store their fields in different units: the POD-coefficient estimators write physical fields, whereas the convolutional estimator writes fields normalised with the per-channel mean and standard deviation kept in its checkpoint. ``load_run`` returns both fields in physical units, which is the convention of every table and figure in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# : Raw snapshot sequence, used to recover the normalisation of runs whose : checkpoint was not kept.
RAW_SEQUENCE = ROOT / "data" / "cylinder2d_q1.npy"

# : Roots of the trained runs, one per estimator family.
ESTIMATOR_ROOTS = {
    "mlp": ROOT / "artifacts" / "pod_model_sweep_nc",
    "vcnn": ROOT / "artifacts" / "vcnn_results",
    "ridge": ROOT / "artifacts" / "ridge_closed_form_sweep_nc",
}

# : Root of the Gappy POD runs of the main sequence. Gappy POD is gappy only in : its sensor count, so it is trained once per configuration.
GAPPY_ROOT = ROOT / "artifacts" / "gappy_closed_form_sweep_nc"

# : Root of the sensor-family study, in which the estimators are retrained on the : five independently sampled nested sensor sequences under the same protocol.
FAMILY_ROOT = ROOT / "artifacts" / "derived" / "supplementary" / "predictions"

# : Suffix encoding a noise level in a run path (``s0010`` = 1e-3).
NOISE_CODES = {0.0: "s0000", 0.001: "s0010", 0.01: "s0100", 0.1: "s1000"}


def noise_code(sigma: float) -> str:
    """Directory name encoding the test noise level."""
    if sigma in NOISE_CODES:
        return NOISE_CODES[sigma]
    return f"s{int(round(sigma * 10_000)):04d}"


def run_path(model: str, sensor_count: int, sigma: float, seed: int) -> Path | None:
    """Path to the test output of one configuration, or ``None`` if absent.

    The linear estimator is deterministic and therefore has no training seed;
    seed 0 of the convolutional estimator is stored under a bespoke directory name inherited from the original sweep.
    """
    code = noise_code(sigma)
    if model == "mlp":
        path = (
            ESTIMATOR_ROOTS["mlp"] / f"mlp_n{sensor_count:04d}" / f"seed{seed:03d}"
            / "tests" / code / "test_raw.npz"
        )
    elif model == "ridge":
        path = (
            ESTIMATOR_ROOTS["ridge"] / f"ridge_n{sensor_count:04d}" / "seed000"
            / "tests" / code / "test_raw.npz"
        )
    elif model == "vcnn":
        if seed == 0:
            path = (
                ESTIMATOR_ROOTS["vcnn"] / "vcnn_sweep_nc_2000"
                / f"vcnn_n{sensor_count:04d}_seed000_custom" / "tests" / code / "test_raw.npz"
            )
        else:
            path = (
                ESTIMATOR_ROOTS["vcnn"] / f"vcnn_sweep_nc_2000_seed{seed:03d}"
                / f"vcnn_n{sensor_count:04d}_seed000_custom" / "tests" / code / "test_raw.npz"
            )
    elif model == "gappy":
        # Gappy POD is deterministic: the validation-selected rank replaces the training seed, so every configuration is stored under seed000.
        path = (
            GAPPY_ROOT / f"gappy_n{sensor_count:04d}" / "seed000"
            / "tests" / code / "test_raw.npz"
        )
    else:
        raise ValueError(f"unknown estimator: {model!r}")
    return path if path.exists() else None


def snapshot_indices(path: Path) -> np.ndarray:
    """Sorted, unique snapshot times covered by a run's test split."""
    data = np.load(path, allow_pickle=True)
    return np.asarray(sorted(set(data["test_indices"].tolist())), dtype=np.int64)


def load_npz(path: Path) -> dict:
    """Raw contents of a ``test_raw.npz`` file as a plain dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalisation_from_checkpoint(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation stored in a checkpoint."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return (
        np.asarray(state["norm_mean_c"], dtype=np.float64),
        np.asarray(state["norm_std_c"], dtype=np.float64),
    )


def normalisation_from_sequence(
    normalised_target: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the normalisation constants of a run from its stored test fields.

    A run whose checkpoint was not kept can still be converted back to physical units: its test snapshots are a subset of the raw sequence, and the normalisation is an exact per-channel affine map, so solving for the affine coefficients recovers the constants. The fit is verified against the raw fields, which turns a silent unit mismatch into an error.
    """
    raw = np.load(str(RAW_SEQUENCE), mmap_mode="r")
    physical = np.asarray(raw[np.asarray(indices)], dtype=np.float64).transpose(0, 3, 1, 2)
    if physical.shape != normalised_target.shape:
        raise ValueError(
            f"snapshot layout mismatch: run {normalised_target.shape}, "
            f"sequence {physical.shape}"
        )
    mean = np.empty(normalised_target.shape[1], dtype=np.float64)
    std = np.empty(normalised_target.shape[1], dtype=np.float64)
    for channel in range(normalised_target.shape[1]):
        values = normalised_target[:, channel].ravel()
        reference = physical[:, channel].ravel()
        design = np.stack([values, np.ones_like(values)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, reference, rcond=None)
        std[channel], mean[channel] = coefficients[0], coefficients[1]

    residual = float(
        np.max(
            np.abs(
                normalised_target * std[:, None, None] + mean[:, None, None] - physical
            )
        )
    )
    if residual > 1e-4:
        raise ValueError(
            f"normalisation could not be recovered from the stored fields "
            f"(residual {residual:.2e})"
        )
    return mean, std


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Target and reconstruction of one run, both in physical units.

    The POD-coefficient estimators store physical fields. The convolutional estimator stores normalised fields and its constants are read from the checkpoint when it is kept, and otherwise recovered from the stored test fields themselves.
    """
    data = np.load(path, allow_pickle=True)
    target = np.asarray(data["target_nchw"], dtype=np.float64)
    recon = np.asarray(data["output_nchw"], dtype=np.float64)
    if "input_nchw" not in data.files:
        return target, recon

    checkpoint = path.parents[2] / "vcnn_best.pt"
    if checkpoint.exists():
        mean, std = normalisation_from_checkpoint(checkpoint)
    else:
        mean, std = normalisation_from_sequence(target, data["test_indices"])
    target = target * std[:, None, None] + mean[:, None, None]
    recon = recon * std[:, None, None] + mean[:, None, None]
    return target, recon

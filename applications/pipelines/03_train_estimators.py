"""Train and evaluate every reconstruction estimator of the main paper.

Four estimators are compared: the closed-form least-squares map, Gappy POD, the
POD-coefficient network and the convolutional estimator. They differ in what
they store and in how they are trained, so the work is done by the routines in
``features.training.pod_sweep`` and ``features.training.vcnn_sweep``; this module
defines the experiment grid and writes each run to the location that the
statistics layer looks in (``features.training.estimator_runs``).

Grid
    sensor counts   10, 15, 20, 30, 50 of one fixed nested sequence
    noise levels    0, 1e-3, 1e-2, 1e-1, applied to the test measurements only
    training seeds  0, 101, 202 (the least-squares map is deterministic)

Each run writes ``tests/<noise code>/test_raw.npz`` holding the target fields, the
reconstructions, the test snapshot indices and the noise level. For a given
training seed every estimator is evaluated on the same test snapshots, which is
what makes the configuration-wise comparisons of the statistics layer valid.

Outputs (main sequence, ``--family family_01``)
    artifacts/pod_model_sweep_nc/{mlp,ridge}_n{count:04d}/seed{seed:03d}/
    artifacts/gappy_closed_form_sweep_nc/gappy_n{count:04d}/seed000/
    artifacts/vcnn_results/vcnn_sweep_nc_2000[_seed{seed:03d}]/

Outputs (any other sensor family, used by the placement study)
    artifacts/derived/supplementary/predictions/{family}/...

Usage
    python applications/pipelines/03_train_estimators.py --models ridge gappy
    python applications/pipelines/03_train_estimators.py --models mlp --jobs 4
    python applications/pipelines/03_train_estimators.py --family family_02 --models mlp
    python applications/pipelines/03_train_estimators.py --check

``--check`` only reports which of the expected runs are present, which is the
cheap way to confirm that the statistics layer will find its inputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402
from features.sensors.mask_registry import (  # noqa: E402
    MASK_COUNTS,
    NC_PREFIX,
    load_nc_mask_bool,
)
from features.training.estimator_runs import (  # noqa: E402
    ESTIMATOR_ROOTS,
    FAMILY_ROOT,
    GAPPY_ROOT,
    noise_code,
)
from features.training.vcnn_config import (  # noqa: E402
    CheckpointConfig,
    DataSourceConfig,
    MaskConfig,
    SweepConfig,
    TrainConfig,
    VcnnConfig,
)
from features.training.vcnn_sweep import run_vcnn_sweep  # noqa: E402

DATA_ARRAY = ROOT / "data" / "cylinder2d_q1.npy"
POD_BUNDLE = ROOT / "artifacts" / "pod_bases" / "cylinder2d_q1" / "pod_base_bundle.npz"
VCNN_ROOT = ROOT / "artifacts" / "vcnn_results"

#: Sensor sequence of the main experiments; the placement study uses the others.
MAIN_FAMILY = "family_01"

MODELS = ("mlp", "ridge", "gappy", "vcnn")
#: Estimators without a training seed, evaluated once per configuration.
DETERMINISTIC = ("ridge", "gappy")

#: Training settings of the convolutional estimator, as used for the paper.
VCNN_TRAINING = {
    "batch_size": 32, "num_epochs": 2000, "lr": 1e-3, "min_lr": 1e-5,
    "warmup_epochs": 5, "cosine_schedule": True, "early_stop": False,
    "weight_decay": 0.0, "val_ratio": 0.1, "test_ratio": 0.2,
}
#: Training settings of the POD-coefficient network.
MLP_TRAINING = {
    "num_epochs": 5000, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
    "early_patience": 30,
}
#: Truncated settings used by --smoke, which only exercises the code path.
SMOKE = {"vcnn_epochs": 2, "vcnn_batches": 2, "mlp_epochs": 2}
#: Grid of the convolutional estimator in the placement study: retraining it on
#: every family and noise level would be prohibitive, so the paper validates it
#: on two sequences with three sensor counts and two noise levels.
VCNN_VALIDATION = {"sensor_counts": (10, 30, 50), "sigmas": (0.0, 0.1), "seeds": (0,)}

#: Redirects every output below a scratch directory. ``--smoke`` sets it so that
#: a truncated test run can never overwrite a run that the paper depends on.
_SCRATCH: Path | None = None


def _redirect(path: Path) -> Path:
    """Output path, moved under the scratch root while --smoke is active."""
    if _SCRATCH is None:
        return path
    return _SCRATCH / path.relative_to(ROOT / "artifacts")


def mask_path(family: str, sensors: int) -> Path:
    """CSV of the nested sequence of one family with ``sensors`` locations."""
    if sensors not in MASK_COUNTS:
        raise ValueError(f"sensor count {sensors} is not one of {MASK_COUNTS}")
    if family == MAIN_FAMILY:
        return ROOT / "masks2" / f"{NC_PREFIX}_n{sensors:03d}.csv"
    return ROOT / "masks_families" / family / "masks" / f"{NC_PREFIX}_n{sensors:03d}.csv"


def output_root(model: str, family: str) -> Path:
    """Directory under which the runs of one estimator are written."""
    if model == "vcnn":
        path = VCNN_ROOT if family == MAIN_FAMILY else FAMILY_ROOT / family
    elif family == MAIN_FAMILY:
        path = GAPPY_ROOT if model == "gappy" else ESTIMATOR_ROOTS[model]
    else:
        path = FAMILY_ROOT / family
    return _redirect(path)


def vcnn_root(training_seed: int, family: str) -> Path:
    """Checkpoint root of a convolutional run; seed 0 keeps the original name."""
    if family != MAIN_FAMILY:
        path = FAMILY_ROOT / family / "vcnn_train"
    elif training_seed == 0:
        path = VCNN_ROOT / "vcnn_sweep_nc_2000"
    else:
        path = VCNN_ROOT / f"vcnn_sweep_nc_2000_seed{training_seed:03d}"
    return _redirect(path)


def run_directory(model: str, family: str, sensors: int, sigma: float, seed: int) -> Path:
    """Directory in which one run stores its test output.

    The convolutional sweep names a model directory after its sensor count and
    mask seed, and marks the use of explicit mask files with ``_custom``. The
    placement study stores its runs in the same layout as the other estimators,
    so that all of them can be summarised by a single scan.
    """
    if model == "vcnn":
        if family == MAIN_FAMILY:
            name = f"vcnn_n{sensors:04d}_seed000_custom"
            return vcnn_root(seed, family) / name / "tests" / noise_code(sigma)
        return _redirect(FAMILY_ROOT / family / f"vcnn_n{sensors:04d}"
                         / f"seed{seed:03d}" / "tests" / noise_code(sigma))
    root = output_root(model, family)
    return root / f"{model}_n{sensors:04d}" / f"seed{seed:03d}" / "tests" / noise_code(sigma)


def expected_runs(models: list[str], family: str, sensor_counts: list[int],
                  sigmas: list[float], seeds: list[int]) -> list[dict]:
    """Every run the statistics layer will look for, with its expected path.

    Outside the main sequence the convolutional estimator is trained on the
    reduced grid of the placement study, so the requested grid is intersected
    with :data:`VCNN_VALIDATION` for that estimator only.
    """
    runs = []
    for model in models:
        model_counts, model_sigmas, model_seeds = sensor_counts, sigmas, seeds
        if model == "vcnn" and family != MAIN_FAMILY:
            model_counts = [M for M in sensor_counts if M in VCNN_VALIDATION["sensor_counts"]]
            model_sigmas = [s for s in sigmas if s in VCNN_VALIDATION["sigmas"]]
            model_seeds = [s for s in seeds if s in VCNN_VALIDATION["seeds"]]
        for sensors in model_counts:
            for seed in (0,) if model in DETERMINISTIC else model_seeds:
                for sigma in model_sigmas:
                    path = run_directory(model, family, sensors, sigma, seed) / "test_raw.npz"
                    runs.append({
                        "model": model, "sensor_count": sensors, "sigma": sigma,
                        "training_seed": seed, "path": path, "exists": path.exists(),
                    })
    return runs


def train_pod_case(model: str, family: str, sensors: int, seed: int,
                   sigmas: tuple[float, ...], device: str, smoke: bool) -> dict:
    """Train one POD-coefficient estimator on one sensor count and evaluate it."""
    from features.training import pod_sweep

    settings = {
        "family": family, "M": sensors, "data_path": DATA_ARRAY,
        "pod_bundle_path": POD_BUNDLE, "mask_hw": load_nc_mask_bool(family, sensors),
        "out_root": output_root(model, family), "test_sigmas": sigmas, "verbose": False,
    }
    if model == "mlp":
        # Only the network has a training seed; the two closed-form estimators
        # are deterministic and always use the seed-0 split.
        settings.update(MLP_TRAINING, training_seed=seed, device=device)
        if smoke:
            settings["num_epochs"] = SMOKE["mlp_epochs"]
        return pod_sweep.run_mlp_case(**settings)
    if model == "ridge":
        return pod_sweep.run_ridge_closed_form_case(**settings)
    if model == "gappy":
        return pod_sweep.run_gappy_case(**settings)
    raise ValueError(f"unknown POD estimator: {model!r}")


def train_vcnn(family: str, sensors: int, seed: int, sigmas: tuple[float, ...],
               device: str, smoke: bool) -> dict:
    """Train the convolutional estimator on one sensor count and evaluate it."""
    root = vcnn_root(seed, family)
    summary = run_vcnn_sweep(
        data_config=DataSourceConfig(array_path=DATA_ARRAY, mmap=True),
        mask_config=MaskConfig(mode="csv", include_mask_channel=True, seed=0),
        model_config=VcnnConfig(
            hidden_channels=48, num_layers=8, kernel_size=7,
            input_representation="voronoi", include_mask_channel=True,
        ),
        train_config=TrainConfig(
            batch_size=VCNN_TRAINING["batch_size"],
            num_epochs=SMOKE["vcnn_epochs"] if smoke else VCNN_TRAINING["num_epochs"],
            val_ratio=VCNN_TRAINING["val_ratio"],
            test_ratio=VCNN_TRAINING["test_ratio"], lr=VCNN_TRAINING["lr"],
            min_lr=VCNN_TRAINING["min_lr"], warmup_epochs=VCNN_TRAINING["warmup_epochs"],
            use_cosine_schedule=VCNN_TRAINING["cosine_schedule"],
            early_stop=VCNN_TRAINING["early_stop"], device=device, seed=seed,
            normalize_mean_std=True, loss_type="mse",
            max_train_batches=SMOKE["vcnn_batches"] if smoke else None,
            progress_every=1,
        ),
        sweep_config=SweepConfig(
            mask_paths=(mask_path(family, sensors),), mask_seeds=(0,),
            train_noise_sigma=0.0, test_noise_sigmas=sigmas,
        ),
        checkpoint_root=root,
        checkpoint_options=CheckpointConfig(
            out_dir=root, save_best_only=True, save_last=True, prefix="vcnn",
        ),
    )
    if family != MAIN_FAMILY:
        place_family_output(family, sensors, seed, sigmas)
    return summary


def place_family_output(family: str, sensors: int, seed: int,
                        sigmas: tuple[float, ...]) -> list[Path]:
    """Move the placement-study output into the common run layout.

    The sweep writes its own directory names; the summary scans all estimators
    with one rule, so the test outputs of this estimator are copied to the same
    location the other estimators use. The reported quantities are evaluated on
    the copied files, so the check in ``--check`` is meaningful.
    """
    import shutil

    source = vcnn_root(seed, family) / f"vcnn_n{sensors:04d}_seed{seed:03d}_custom" / "tests"
    placed = []
    for sigma in sigmas:
        src = source / noise_code(sigma) / "test_raw.npz"
        if not src.exists():
            print(f"   [warn] no test output for M={sensors}, sigma={sigma}")
            continue
        dst_dir = run_directory("vcnn", family, sensors, sigma, seed)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "test_raw.npz")
        placed.append(dst_dir / "test_raw.npz")
    return placed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--family", default=MAIN_FAMILY,
                        help="sensor family; family_01 is the main sequence")
    parser.add_argument("--sensor-counts", nargs="+", type=int, default=None)
    parser.add_argument("--sigmas", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check", action="store_true", help="report present runs only")
    parser.add_argument("--smoke", action="store_true", help="truncated run to exercise the path")
    args = parser.parse_args()

    cfg = get_config()
    sensor_counts = args.sensor_counts or cfg.M_values
    sigmas = args.sigmas or cfg.sigma_values
    seeds = args.seeds or cfg.mlp_seeds
    if args.smoke:
        global _SCRATCH
        _SCRATCH = ROOT / "artifacts" / "smoke"
        sensor_counts, sigmas, seeds = sensor_counts[:1], sigmas[:1], seeds[:1]
        print(f"[smoke] outputs are redirected to {_SCRATCH.relative_to(ROOT)}")

    runs = expected_runs(args.models, args.family, sensor_counts, sigmas, seeds)
    if args.check:
        present = sum(run["exists"] for run in runs)
        print(f"[{args.family}] expected {len(runs)} runs, found {present}")
        missing = [run for run in runs if not run["exists"]]
        for run in missing[:20]:
            print(f"   missing {run['model']:6s} M={run['sensor_count']:3d} "
                  f"sigma={run['sigma']:g} seed={run['training_seed']:3d}")
        if len(missing) > 20:
            print(f"   ... and {len(missing) - 20} more")
        return 1 if missing else 0

    for path, label in ((DATA_ARRAY, "raw data"), (POD_BUNDLE, "POD basis")):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    start = time.time()
    print(f"== training estimators on {args.family} (models={args.models}, "
          f"M={sensor_counts}, sigma={sigmas}, seeds={seeds})")

    jobs = [
        (model, sensors, seed)
        for model in args.models
        for sensors in sensor_counts
        for seed in seeds
        if model not in DETERMINISTIC or seed == seeds[0]
    ]

    def run_one(job: tuple) -> dict:
        model, sensors, seed = job
        print(f"   {model:6s} M={sensors:3d} seed={seed:3d}")
        if model == "vcnn":
            return train_vcnn(args.family, sensors, seed, tuple(sigmas), args.device,
                              args.smoke)
        return train_pod_case(model, args.family, sensors, seed, tuple(sigmas),
                              args.device, args.smoke)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_one, jobs))
    else:
        results = [run_one(job) for job in jobs]

    print(f"[OK] {len(results)} runs trained ({time.time() - start:.1f}s)")
    missing = [
        run for run in expected_runs(args.models, args.family, sensor_counts, sigmas, seeds)
        if not run["exists"]
    ]
    if missing:
        print(f"   [warn] {len(missing)} runs are still missing after training")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

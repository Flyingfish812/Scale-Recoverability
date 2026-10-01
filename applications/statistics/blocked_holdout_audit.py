"""Audit the contiguous-holdout (blocked) sensitivity end to end.

Two questions are answered from one deterministic pass over the canonical runs.

*Estimator table.* Each estimator is retrained with contiguous
train/validation/test blocks separated by a guard gap; the module rebuilds the
reported mean global relative error and mean ``S_full`` of every model and
placement at the representative condition $(M,\sigma)=(20,0)$, evaluating the
inner \FBlockedNTest{} snapshots of the test block in physical units over both
velocity components. This is the table the supplementary material prints, so the
numbers have a reproducible source rather than being read off a one-off script.

*Representation audit.* Placement 2 degrades far more than placements 0 and 1,
and it degrades the two deterministic baselines along with the learned ones, so
the cause cannot be estimator-specific. The module therefore also evaluates the
exact rank-128 POD coefficient reconstruction of each held-out window, with no
observation model and no noise: if the representation describes the window, the
extra error belongs to the estimators; if it does not, the window lies outside
the span of a basis trained on the remaining snapshots. The placement-2 window is
additionally evaluated with the basis of the random split, which separates a
basis effect from a state-interval effect, and the leading limit-cycle harmonic
is compared between each training set and its own held-out window.

Output
------
artifacts/statistics/blocked_holdout_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.metrics.sample_metrics import compute_sample_metrics  # noqa: E402
from features.training.estimator_runs import load_run  # noqa: E402


def _load_training_pipeline():
    """The training pipeline owns the blocked split, so it is the reference.

    Its module name starts with a digit and cannot be imported by name.
    """
    path = ROOT / "applications" / "pipelines" / "03_train_estimators.py"
    spec = importlib.util.spec_from_file_location("_train_estimators", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PIPELINE = _load_training_pipeline()

BLOCKED_ROOT = ROOT / "results" / "20260930_blocked_canonical"
#: directory tag -> placement. The convolutional estimator was trained in two
#: halves, one per sensor count, so placement 2 has two output roots.
TAG_TO_PLACEMENT = {"p0": 0, "p1": 1, "p2": 2, "p2_m20": 2, "p2_m30": 2}
SEEDS = ("seed000", "seed101", "seed202")
MODELS = ("mlp", "vcnn", "ridge", "gappy")

REPORTED_M = 20
REPORTED_SIGMA = 0.0
TAU = 0.05
RANK = 128

REFERENCE_MODEL, REFERENCE_M, REFERENCE_SIGMA, REFERENCE_SEED = "mlp", 20, 0.0, 0


# ---------------------------------------------------------------------------
# estimator table
# ---------------------------------------------------------------------------
RUN_PATTERNS = {
    "mlp": "pod_model_sweep_nc/mlp_n{M:04d}/seed*/tests/s0000/test_raw.npz",
    "ridge": "ridge_closed_form_sweep_nc/ridge_n{M:04d}/seed000/tests/s0000/test_raw.npz",
    "gappy": "gappy_closed_form_sweep_nc/gappy_n{M:04d}/seed000/tests/s0000/test_raw.npz",
    "vcnn": "vcnn_results/*/vcnn_n{M:04d}_seed000_custom/tests/s0000/test_raw.npz",
}


def cell(npz: Path, rows: np.ndarray) -> dict:
    """Mean GER and mean ``S_full`` of one run over the rows of the inner block."""
    target, recon = load_run(npz)
    with np.load(npz) as data:
        indices = np.asarray(data["test_indices"], dtype=np.int64)
    keep = np.where((indices >= rows[0]) & (indices < rows[-1] + 1))[0]
    metrics = [compute_sample_metrics(recon, target, int(i), tau=TAU) for i in keep]
    return {
        "n_snapshots": int(len(keep)),
        "ger": float(np.mean([m["GER"] for m in metrics])),
        "s_full": float(np.mean([m["S_full"] for m in metrics])),
    }


def estimator_table(sequence_rows: dict[int, np.ndarray]) -> dict:
    rows: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        rows[model] = {}
        for tag, placement in TAG_TO_PLACEMENT.items():
            pattern = RUN_PATTERNS[model].format(M=REPORTED_M)
            runs = sorted((BLOCKED_ROOT / tag).glob(pattern))
            runs = [r for r in runs if model not in ("mlp", "vcnn")
                    or any(s in str(r) for s in SEEDS)]
            if not runs:
                continue
            cells = [cell(r, sequence_rows[placement]) for r in runs]
            entry = rows[model].setdefault(str(placement), {
                "n_runs": 0, "n_snapshots": 0, "ger": [], "s_full": []})
            entry["ger"].extend(c["ger"] for c in cells)
            entry["s_full"].extend(c["s_full"] for c in cells)
            entry["n_runs"] += len(cells)
            entry["n_snapshots"] = int(sum(c["n_snapshots"] for c in cells) / len(cells))
    for model, per_placement in rows.items():
        for placement, entry in per_placement.items():
            entry["ger"] = round(float(np.mean(entry["ger"])), 5)
            entry["s_full"] = round(float(np.mean(entry["s_full"])), 3)
    return {
        "sensor_count": REPORTED_M,
        "noise_sigma": REPORTED_SIGMA,
        "placement_seeds": list(SEEDS),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# representation audit
# ---------------------------------------------------------------------------
def load_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        mean = np.asarray(data["mean_field"], dtype=np.float64)
        basis = np.asarray(data["pod_basis"], dtype=np.float64)
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
    return mean, basis.reshape(basis.shape[0], -1), coefficients


def exact_reconstruction(mean: np.ndarray, basis: np.ndarray,
                         coefficients: np.ndarray) -> np.ndarray:
    field = mean.reshape(-1)[None, :] + coefficients @ basis
    return field.reshape((-1,) + mean.shape).transpose(0, 3, 1, 2)


def window_reading(sequence: np.ndarray, mean: np.ndarray, basis: np.ndarray,
                   coefficients: np.ndarray, rows: np.ndarray) -> dict:
    recon = exact_reconstruction(mean, basis, coefficients[rows])
    target = np.ascontiguousarray(sequence[rows].transpose(0, 3, 1, 2))
    metrics = [compute_sample_metrics(recon, target, i, tau=TAU)
               for i in range(len(rows))]
    ger = np.array([m["GER"] for m in metrics])
    sfull = np.array([m["S_full"] for m in metrics])
    a4 = np.array([m["E_A4"] for m in metrics])
    w1 = np.array([m["E_W1"] for m in metrics])
    return {
        "n_snapshots": int(len(rows)),
        "ger_mean": round(float(ger.mean()), 8),
        "s_full_mean": round(float(sfull.mean()), 4),
        "s_full_five_pct": round(100.0 * float((sfull == 5).mean()), 1),
        "a4_pass_pct": round(100.0 * float((a4 <= TAU).mean()), 1),
        "e_direct_a4_mean": round(float(a4.mean()), 8),
        "e_direct_w1_mean": round(float(w1.mean()), 8),
        "e_direct_w1_max": round(float(w1.max()), 8),
    }


def harmonic_reading(coefficients: np.ndarray, train_rows: np.ndarray,
                     rows: np.ndarray, n_modes: int = 10) -> dict:
    amplitude_train = np.hypot(coefficients[train_rows, 0], coefficients[train_rows, 1])
    amplitude_test = np.hypot(coefficients[rows, 0], coefficients[rows, 1])
    mean = coefficients[train_rows, :n_modes].mean(axis=0)
    std = coefficients[train_rows, :n_modes].std(axis=0)
    z = np.abs(coefficients[rows, :n_modes] - mean[None, :]) / (std[None, :] + 1e-30)
    return {
        "amplitude_train_mean": round(float(amplitude_train.mean()), 4),
        "amplitude_test_mean": round(float(amplitude_test.mean()), 4),
        "amplitude_ratio": round(float(amplitude_test.mean() / amplitude_train.mean()), 4),
        "z_median": round(float(np.median(z)), 3),
        "z_max": round(float(z.max()), 3),
        "pct_beyond_three_sigma": round(100.0 * float((z > 3.0).mean()), 2),
    }


def random_test_indices() -> np.ndarray:
    """Test rows of the canonical random split, taken from the reference run."""
    matches = sorted((ROOT / "artifacts" / "pod_model_sweep_nc"
                      / f"mlp_n{REFERENCE_M:04d}" / "seed000" / "tests" / "s0000"
                      ).glob("*.npz"))
    if not matches:
        raise FileNotFoundError("the reference random-split run is missing")
    with np.load(matches[0]) as data:
        return np.asarray(data["test_indices"], dtype=np.int64)


def representation_audit() -> tuple[dict, dict, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # level-4 boundary-effects notice
        sequence = np.asarray(np.load(PIPELINE.DATA_ARRAY, mmap_mode="r"),
                              dtype=np.float64)
        n_total = int(sequence.shape[0])

        random_rows = random_test_indices()
        random_train = np.setdiff1d(np.arange(n_total), random_rows)
        mean_r, basis_r, coefficients_r = load_bundle(PIPELINE.POD_BUNDLE)

        arms: dict[str, dict] = {}
        shifts: dict[str, dict] = {}
        random_reading = window_reading(sequence, mean_r, basis_r, coefficients_r,
                                        random_rows)
        random_reading["pod_bundle"] = str(PIPELINE.POD_BUNDLE.relative_to(ROOT))
        arms["random"] = random_reading
        shifts["random"] = harmonic_reading(coefficients_r, random_train, random_rows)

        cross_basis: dict[str, dict] = {}
        for placement in sorted(PIPELINE.BLOCKED_PLACEMENTS):
            tag = f"p{placement}"
            split = PIPELINE.blocked_split(placement, n_total)
            bundle = (BLOCKED_ROOT / tag / "pod_bases" / "cylinder2d_q1"
                      / f"blocked_p{placement}.npz")
            mean, basis, coefficients = load_bundle(bundle)
            rows = np.asarray(split["test_inner"], dtype=np.int64)
            reading = window_reading(sequence, mean, basis, coefficients, rows)
            reading["pod_bundle"] = str(bundle.relative_to(ROOT))
            reading["n_training_snapshots"] = int(len(split["train"]))
            arms[tag] = reading
            shifts[tag] = harmonic_reading(
                coefficients, np.asarray(split["train"], dtype=np.int64), rows)
            cross = window_reading(sequence, mean_r, basis_r, coefficients_r, rows)
            cross["pod_bundle"] = "random-split basis"
            cross["note"] = (
                "the random-split basis was trained on snapshots interleaved with "
                "this window, so the reading bounds the temporal adjacency of the "
                "random protocol rather than the quality of a held-out basis")
            cross_basis[tag] = cross
    return arms, cross_basis, shifts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only write the artifact")
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inner_rows = {p: np.asarray(PIPELINE.blocked_split(p, 1501)["test_inner"],
                                    dtype=np.int64)
                      for p in sorted(PIPELINE.BLOCKED_PLACEMENTS)}
        table = estimator_table(inner_rows)
        arms, cross_basis, shifts = representation_audit()

    result = {
        "schema": "luna.blocked_holdout_audit.v1",
        "task": "contiguous-holdout sensitivity audit",
        "description": (
            "Reported blocked-split table (mean GER and mean S_full over the inner "
            "test block, physical units, both velocity components) together with "
            "the exact rank-128 POD coefficient reconstruction of every held-out "
            "window, which is what separates an estimator effect from a "
            "representation effect."),
        "guard_snapshots": int(PIPELINE.BLOCKED_GUARD),
        "inner_test_snapshots": int(PIPELINE.BLOCKED_N_TEST),
        "tau": TAU,
        "rank": RANK,
        "estimator_table": table,
        "representation": arms,
        "cross_basis": cross_basis,
        "coefficient_shift": shifts,
    }

    path = ROOT / "artifacts" / "statistics" / "blocked_holdout_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"{'model':7s} " + " ".join(f"{'p'+str(p):>16s}"
                                          for p in sorted(PIPELINE.BLOCKED_PLACEMENTS)))
        for model, per_placement in table["rows"].items():
            cells = []
            for placement in sorted(PIPELINE.BLOCKED_PLACEMENTS):
                entry = per_placement.get(str(placement))
                cells.append(f"{entry['ger']:.5f}/{entry['s_full']:.2f}"
                             if entry else "-")
            print(f"{model:7s} " + " ".join(f"{c:>16s}" for c in cells))
        print("  representation (GER / S_full):",
              {k: f"{v['ger_mean']:.2e}/{v['s_full_mean']:.2f}"
               for k, v in arms.items()})
    print(f"[OK] {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

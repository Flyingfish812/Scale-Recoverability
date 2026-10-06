#!/usr/bin/env python3
"""Retained-rank (r) sensitivity: r in {100, 128, 150}.

The main study fixes r = 128.  This sensitivity checks whether the coarse-to-fine hierarchy and the estimator ranking depend on that particular choice, by rebuilding the POD bundle of the numerical wake with 200 modes and re-running the coefficient maps at r = 100, 128, 150.

Estimators: Ridge (closed form), Gappy POD (rank cap r <= 2M) and MLP, all at
the three ranks.  The bundle is written by the canonical builder of the
published bases (``applications/pipelines/02_build_pod_bases.py``), so the
r = 128 arm reproduces the main study and only the retained rank varies.

Outputs (nothing existing is touched):
    <out-root>/r{100,128,150}/...            npz per case
    <out-root>/rank_sensitivity.json          summary

Usage
-----
    PYTHONPATH=<repo root>
    python -u applications/statistics/rank_sensitivity.py \
        [--out-root results/rank_scan]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from features.training.pod_sweep import (  # noqa: E402
    run_gappy_case,
    run_mlp_case,
    run_ridge_closed_form_case,
)

H, W, C = 80, 160, 2
N_BUNDLE = 200                                   # bundle size (>= max scanned r)
RANKS = (100, 128, 150)
DATASET = "nc"                                   # numerical cylinder wake (registry key)
RIDGE_MASKS = (20, 30, 50)
MLP_MASKS = (20, 30)
MLP_RANKS = RANKS
SIGMAS = (0.0, 0.001, 0.01, 0.1)
CANDIDATE_RANKS = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128)
FAMILY = "inc"

DATA = ROOT / "data" / "cylinder2d_q1.npy"
# : Output root; ``--out-root`` overrides it so the published script carries no : date-stamped batch directory.
OUT_ROOT = ROOT / "results" / "rank_scan"
BUNDLE = OUT_ROOT / "pod_bundle_200.npz"

_PRED_KEYS = ("output_nchw", "pred", "prediction", "reconstruction", "recon")
_TGT_KEYS = ("target_nchw", "target", "tgt", "truth", "reference", "ref")


def relative(path: Path) -> str:
    """Path relative to the repository root when possible (keeps the summary
    portable across machines and batch directories)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_mask(mask_num: int) -> np.ndarray:
    mask_path = ROOT / "masks2" / f"cylinder2d_80x160_random_inc_n{mask_num:03d}.csv"
    coords = np.loadtxt(str(mask_path), delimiter=",", dtype=np.int32, skiprows=1)
    mask = np.zeros((H, W), dtype=bool)
    for row, col in coords:
        mask[int(row), int(col)] = True
    return mask


def build_bundle_200(out_root: Path) -> Path:
    """POD bundle with ``N_BUNDLE`` modes, built by the canonical builder.

    ``applications/pipelines/02_build_pod_bases.py`` writes the published
    bundles; it is called here with a larger rank so that the basis, the mean field and the coefficient convention are exactly those of the rank-128 bundle the main study uses.  The scan then varies the retained rank alone,
    and its r = 128 arm reproduces the main study instead of re-deriving it on a
    basis built from a different snapshot sample.
    """
    bundle = out_root / "pod_bundle_200.npz"
    if bundle.exists():
        return bundle
    spec = importlib.util.spec_from_file_location(
        "luna_build_pod_bases",
        ROOT / "applications" / "pipelines" / "02_build_pod_bases.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays = module.build_bundle(DATASET, N_BUNDLE)
    np.savez(
        str(bundle),
        pod_basis=arrays["pod_basis"],
        mean_field=arrays["mean_field"],
        coefficients=arrays["coefficients"],
    )
    return bundle


def ger(npz_path: Path) -> float | None:
    with np.load(str(npz_path)) as z:
        pk = next((k for k in _PRED_KEYS if k in z.files), None)
        tk = next((k for k in _TGT_KEYS if k in z.files), None)
        if pk is None or tk is None:
            return None
        pred = np.asarray(z[pk], dtype=np.float64).reshape(len(z[pk]), -1)
        tgt = np.asarray(z[tk], dtype=np.float64).reshape(len(z[tk]), -1)
    return float(np.mean(np.linalg.norm(pred - tgt, axis=1) / (np.linalg.norm(tgt, axis=1) + 1e-12)))


def main(argv: list[str] | None = None) -> None:
    global OUT_ROOT, BUNDLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT),
                        help="directory for the scan outputs (default: results/rank_scan)")
    args = parser.parse_args(argv)
    OUT_ROOT = Path(args.out_root)
    if not OUT_ROOT.is_absolute():
        OUT_ROOT = ROOT / OUT_ROOT
    BUNDLE = OUT_ROOT / "pod_bundle_200.npz"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("=" * 72)
    print(f"Retained-rank sensitivity r in {RANKS}")
    print("=" * 72, flush=True)
    bundle = build_bundle_200(OUT_ROOT)
    print(f"bundle ready: {bundle} ({N_BUNDLE} modes)", flush=True)

    rows: list[dict] = []
    for rank in RANKS:
        out_root = OUT_ROOT / f"r{rank}"
        for mask_num in RIDGE_MASKS:
            mask_hw = load_mask(mask_num)
            for name, runner in (
                ("ridge", lambda: run_ridge_closed_form_case(
                    family=FAMILY, M=mask_num, data_path=DATA, pod_bundle_path=bundle,
                    mask_hw=mask_hw, out_root=out_root, test_sigmas=SIGMAS,
                    n_modes=rank, verbose=False)),
                ("gappy", lambda: run_gappy_case(
                    family=FAMILY, M=mask_num, data_path=DATA, pod_bundle_path=bundle,
                    mask_hw=mask_hw, out_root=out_root, test_sigmas=SIGMAS,
                    n_modes=rank, candidate_ranks=CANDIDATE_RANKS,
                    rank_cap="scalars", verbose=False)),
            ):
                tag = f"r{rank}/{name}/M{mask_num}"
                try:
                    t0 = time.time()
                    case = runner()
                    dt = time.time() - t0
                    print(f"[{tag}] done in {dt:.1f}s (rank={case.get('rank', '-')})", flush=True)
                    for sigma in SIGMAS:
                        npz = Path(case["npz_paths"][float(sigma)])
                        rows.append({"r": rank, "estimator": name, "M": mask_num,
                                     "m_obs": mask_num * C, "sigma": float(sigma),
                                     "selected_rank": case.get("rank"),
                                     "ger": ger(npz) if npz.exists() else None,
                                     "npz": relative(npz), "seconds": round(dt, 2)})
                except Exception as exc:
                    print(f"[{tag}] FAILED: {exc}", flush=True)
                    traceback.print_exc()
                    rows.append({"r": rank, "estimator": name, "M": mask_num,
                                 "sigma": None, "error": str(exc)})

    for rank in MLP_RANKS:
        out_root = OUT_ROOT / f"r{rank}"
        for mask_num in MLP_MASKS:
            tag = f"r{rank}/mlp/M{mask_num}"
            try:
                t0 = time.time()
                print(f"[{tag}] training MLP (r={rank}) ...", flush=True)
                case = run_mlp_case(
                    family=FAMILY, M=mask_num, training_seed=0, data_path=DATA,
                    pod_bundle_path=bundle, mask_hw=load_mask(mask_num),
                    out_root=out_root, test_sigmas=SIGMAS, n_modes=rank, verbose=False)
                dt = time.time() - t0
                print(f"[{tag}] done in {dt:.1f}s", flush=True)
                for sigma in SIGMAS:
                    npz = Path(case["npz_paths"][float(sigma)])
                    rows.append({"r": rank, "estimator": "mlp", "M": mask_num,
                                 "m_obs": mask_num * C, "sigma": float(sigma),
                                 "selected_rank": rank,
                                 "ger": ger(npz) if npz.exists() else None,
                                 "npz": relative(npz), "seconds": round(dt, 2)})
            except Exception as exc:
                print(f"[{tag}] FAILED: {exc}", flush=True)
                traceback.print_exc()
                rows.append({"r": rank, "estimator": "mlp", "M": mask_num,
                             "sigma": None, "error": str(exc)})

    out_json = OUT_ROOT / "rank_sensitivity.json"
    with out_json.open("w") as handle:
        json.dump({
            "task": "retained_rank_sensitivity",
            "description": ("Ridge / Gappy (r <= 2M) / MLP re-run at retained ranks "
                            "100, 128 and 150 on the standard split, with the canonical "
                            "POD bundle of the main study built at 200 modes; "
                            "r = 128 reproduces the main study."),
            "ranks": list(RANKS),
            "sigma": list(SIGMAS),
            "bundle_modes": N_BUNDLE,
            "total_seconds": round(time.time() - t_start, 1),
            "results": rows,
        }, handle, indent=2)
    print(f"\nwrote {out_json}  (total {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

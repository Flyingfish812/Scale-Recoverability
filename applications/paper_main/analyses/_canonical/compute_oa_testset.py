#!/usr/bin/env python3
"""
P0-1 完整修复: Oracle Audit 使用主实验相同的测试集快照 (2026-07-20)

问题: Oracle audit 从全量数据随机采样 200, 而非使用主实验测试集 (300)
修复: 使用时间顺序最后 300 个快照 (NC test set), 与主实验口径完全一致

用法:
  conda run -n sana python scripts/20260720/fix_p0_oracle_audit.py

输出:
  results/20260720/oracle_audit_testset.json  — 基于测试集的 Oracle audit 结果
  results/20260720/oracle_audit_testset.csv   — 扁平化表格
  results/20260720/oracle_audit_testset.txt   — 可读摘要
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from luna.core.constants import (
    BANDS_CF, TAU_DEFAULT, DEFAULT_WAVELET, DEFAULT_LEVEL, DEFAULT_MODE, EPS,
)
from luna.data.io import load_npy, load_npz, save_json
from luna.pod.oracle import pod_oracle_reconstruct
from luna.wavelet.transform import decompose_field_2d
from luna.wavelet.metrics import band_error

# ══════════════════════════════════════════════════════════════════════
# Dataset Configurations — 与主实验一致的测试集划分
# ══════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "nc": {
        "data": "data/cylinder2d_q1.npy",
        "pod_bundle": "artifacts/pod_bases/cylinder2d_q1/pod_base_bundle.npz",
        "grid": (80, 160),
        "n_channels": 2,
        "ranks": [16, 32, 64, 128, 200],
        "test_ratio": 0.2,
        "val_ratio": 0.1,
        "n_test_samples": 300,  # 与主实验完全一致
    },
    "rdb_h5": {
        "data": "data/rdb_h5.npy",
        "pod_bundle": "artifacts/pod_bases/rdb_h5/pod_base_bundle.npz",
        "grid": (128, 128),
        "n_channels": 1,
        "ranks": [16, 32, 64, 128, 256],
        "test_ratio": 0.2,
        "val_ratio": 0.1,
        "n_test_samples": 300,  # 取前 300 个测试集样本（保持与 NC 一致）
    },
    "sst_weekly": {
        "data": "data/sst_weekly.npy",
        "pod_bundle": "artifacts/pod_bases/sst_weekly/pod_base_bundle.npz",
        "grid": (180, 360),
        "n_channels": 1,
        "ranks": [32, 64, 128, 256, 512, 1024],
        "test_ratio": 0.2,
        "val_ratio": 0.1,
        "n_test_samples": 300,  # 取前 300 个测试集样本
    },
}

NC_NORM_MEAN = np.array([1.0004944, -0.00017817653], dtype=np.float64)
NC_NORM_STD = np.array([0.21863055, 0.19121747], dtype=np.float64)


def get_test_set_indices(n_total: int, test_ratio: float, val_ratio: float,
                         n_max: int | None = None) -> np.ndarray:
    """获取时间顺序测试集索引 (与主实验完全一致)。

    主实验使用时间顺序划分: 前 train_ratio 训练, 中间 val_ratio 验证,
    最后 test_ratio 测试。
    """
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    test_start = n_total - n_test
    indices = np.arange(test_start, n_total, dtype=int)
    if n_max is not None and len(indices) > n_max:
        indices = indices[:n_max]
    return indices


def compute_band_truncation_errors(
    target_fields: np.ndarray,
    basis_full_flat: np.ndarray | None,
    mean_full_flat: np.ndarray | None,
    basis_ch0_flat: np.ndarray,
    mean_ch0_flat: np.ndarray,
    ranks: list[int],
    wavelet: str,
    level: int,
    mode: str,
    n_channels: int = 1,
    grid: tuple[int, int] = (80, 160),
    fields_full: np.ndarray | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute per-sample E_trunc(b) for each rank."""
    H, W = grid
    N = target_fields.shape[0]
    basis_ch0 = np.asarray(basis_ch0_flat, dtype=np.float64)
    mean_ch0 = np.asarray(mean_ch0_flat, dtype=np.float64).ravel()
    result: dict[int, dict[str, np.ndarray]] = {}

    for r in ranks:
        rclamped = min(r, basis_ch0.shape[0])
        per_sample: dict[str, list[float]] = {b: [] for b in BANDS_CF}

        for i in range(N):
            u = target_fields[i]

            if basis_full_flat is not None and mean_full_flat is not None and n_channels > 1 and fields_full is not None:
                basis_full_r = np.asarray(basis_full_flat[:rclamped, :], dtype=np.float64)
                mean_full = np.asarray(mean_full_flat, dtype=np.float64).ravel()
                u_full = fields_full[i].ravel()
                coeff = basis_full_r @ (u_full - mean_full)
                u_oracle_full = basis_full_r.T @ coeff + mean_full
                u_oracle = u_oracle_full.reshape(H, W, n_channels)[:, :, 0]
            else:
                basis_r = basis_ch0[:rclamped, :]
                u_flat = u.ravel()
                coeff = basis_r @ (u_flat - mean_ch0)
                u_oracle_flat = basis_r.T @ coeff + mean_ch0
                u_oracle = u_oracle_flat.reshape(H, W)

            u_bands = decompose_field_2d(u, wavelet, level, mode)
            o_bands = decompose_field_2d(u_oracle, wavelet, level, mode)

            for b in BANDS_CF:
                per_sample[b].append(band_error(u_bands[b], o_bands[b]))

        result[r] = {b: np.array(per_sample[b], dtype=np.float64) for b in BANDS_CF}

    return result


def compute_oracle_audit_testset(dataset_name: str, config: dict, tau: float,
                                  tau_q95: float, wavelet: str, level: int,
                                  mode: str, project_root: Path) -> dict:
    """使用主实验测试集快照运行 oracle audit。"""
    print(f"\n{'='*60}")
    print(f"  数据集: {dataset_name}")
    print(f"{'='*60}")

    data_path = project_root / config["data"]
    pod_path = project_root / config["pod_bundle"]
    H, W = config["grid"]
    C = config["n_channels"]
    ranks = config["ranks"]

    print(f"  加载数据: {data_path}")
    fields = load_npy(str(data_path))
    print(f"  数据形状: {fields.shape}")

    print(f"  加载 POD: {pod_path}")
    pod = load_npz(str(pod_path))
    basis_raw = np.asarray(pod["pod_basis"], dtype=np.float64)
    mean_raw = np.asarray(pod["mean_field"], dtype=np.float64)
    R_avail = basis_raw.shape[0]
    print(f"  POD 可用秩: {R_avail}")

    # ── 使用测试集索引 ────────────────────────────────────────────
    n_total = fields.shape[0]
    test_indices = get_test_set_indices(n_total, config["test_ratio"],
                                        config["val_ratio"], config["n_test_samples"])
    n_actual = len(test_indices)
    print(f"  测试集: {n_actual} 个快照 (索引 {test_indices[0]}–{test_indices[-1]}, "
          f"从全量 {n_total} 中取最后 {config['test_ratio']*100:.0f}%)")

    # ── Prepare fields ─────────────────────────────────────────────
    if fields.ndim == 4:
        test_fields_full = fields[test_indices].astype(np.float64)
        target_ch0 = test_fields_full[:, :, :, 0]
        D_full = H * W * C
        basis_full_flat = basis_raw.reshape(R_avail, D_full)
        mean_full_flat = mean_raw.ravel()
        basis_ch0_flat = basis_raw[:, :, :, 0].reshape(R_avail, H * W)
        mean_ch0_flat = mean_raw[:, :, 0].ravel()
        use_full_pod = True
    else:
        test_fields_full = fields[test_indices].astype(np.float64)
        target_ch0 = test_fields_full
        D_full = H * W
        basis_full_flat = basis_raw.reshape(R_avail, D_full)
        mean_full_flat = mean_raw.ravel()
        basis_ch0_flat = basis_full_flat
        mean_ch0_flat = mean_full_flat
        use_full_pod = False

    print(f"  目标场形状: {target_ch0.shape}")
    print(f"  使用全通道 POD: {use_full_pod}")

    # ── Compute ────────────────────────────────────────────────────
    effective_ranks = sorted(set(r for r in ranks if r <= R_avail))
    if R_avail not in effective_ranks:
        effective_ranks.append(R_avail)
    effective_ranks = sorted(set(effective_ranks))
    print(f"  有效秩: {effective_ranks}")

    fields_full_tensor = None
    if use_full_pod and fields.ndim == 4:
        fields_full_tensor = fields[test_indices].astype(np.float64)

    print(f"  计算逐样本频带截断误差...")
    t0 = time.time()
    per_sample_errors = compute_band_truncation_errors(
        target_fields=target_ch0,
        basis_full_flat=basis_full_flat if use_full_pod else None,
        mean_full_flat=mean_full_flat if use_full_pod else None,
        basis_ch0_flat=basis_ch0_flat,
        mean_ch0_flat=mean_ch0_flat,
        ranks=effective_ranks,
        wavelet=wavelet,
        level=level,
        mode=mode,
        n_channels=C,
        grid=(H, W),
        fields_full=fields_full_tensor,
    )
    elapsed = time.time() - t0
    print(f"  完成 ({elapsed:.1f}s)")

    # ── 统计 ───────────────────────────────────────────────────────
    audit_table: dict[str, dict] = {}
    safe_rank_mean: int | None = None
    safe_rank_q95: int | None = None
    safe_rank_strict: int | None = None

    for r in effective_ranks:
        band_stats = {}
        all_mean_ok = True
        all_q95_ok = True

        for b in BANDS_CF:
            errs = per_sample_errors[r][b]
            mean_val = float(np.mean(errs))
            q95_val = float(np.quantile(errs, 0.95))
            max_val = float(np.max(errs))
            std_val = float(np.std(errs))
            median_val = float(np.median(errs))

            band_stats[b] = {
                "mean": mean_val,
                "median": median_val,
                "q95": q95_val,
                "max": max_val,
                "std": std_val,
                "n_samples": n_actual,
                "mean_ok": mean_val < tau,
                "q95_ok": q95_val < tau_q95,
            }
            if mean_val >= tau:
                all_mean_ok = False
            if q95_val >= tau_q95:
                all_q95_ok = False

        audit_table[str(r)] = {
            "bands": band_stats,
            "mean_ok_all_bands": all_mean_ok,
            "q95_ok_all_bands": all_q95_ok,
            "strict_ok": all_mean_ok and all_q95_ok,
        }

        if all_mean_ok and safe_rank_mean is None:
            safe_rank_mean = r
        if all_q95_ok and safe_rank_q95 is None:
            safe_rank_q95 = r
        if all_mean_ok and all_q95_ok and safe_rank_strict is None:
            safe_rank_strict = r

    summary = {
        "dataset": dataset_name,
        "config": {
            "grid": list(config["grid"]),
            "n_channels": C,
            "n_test_samples": n_actual,
            "test_indices": [int(i) for i in test_indices],
            "test_ratio": config["test_ratio"],
            "tau": tau,
            "tau_q95": tau_q95,
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
            "ranks_tested": effective_ranks,
        },
        "safe_rank_mean_criterion": safe_rank_mean,
        "safe_rank_q95_criterion": safe_rank_q95,
        "safe_rank_strict_criterion": safe_rank_strict,
        "audit_table": audit_table,
    }

    return summary


def print_audit_summary(summary: dict) -> None:
    ds = summary["dataset"]
    cfg = summary["config"]
    print(f"\n  DATASET: {ds}")
    print(f"  Tau: {cfg['tau']}, Tau/5: {cfg['tau_q95']}")
    print(f"  测试样本: {cfg['n_test_samples']}")
    print(f"  测试秩: {cfg['ranks_tested']}")
    print(f"  安全秩 (mean<τ):       {summary['safe_rank_mean_criterion']}")
    print(f"  安全秩 (Q95<τ/5):      {summary['safe_rank_q95_criterion']}")
    print(f"  安全秩 (BOTH):         {summary['safe_rank_strict_criterion']}")
    print()


def main() -> None:
    project_root = _PROJECT_ROOT
    output_dir = project_root / "artifacts" / "derived" / "main" / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    tau = TAU_DEFAULT
    tau_q95 = tau / 5.0
    wavelet = DEFAULT_WAVELET
    level = DEFAULT_LEVEL
    mode = DEFAULT_MODE

    print("=" * 60)
    print("P0-1 修复: Oracle Audit 使用主实验测试集快照")
    print("=" * 60)

    all_summaries = []

    for ds_name in ["nc", "rdb_h5", "sst_weekly"]:
        if ds_name not in DATASET_CONFIGS:
            print(f"  SKIP {ds_name}")
            continue

        summary = compute_oracle_audit_testset(
            dataset_name=ds_name,
            config=DATASET_CONFIGS[ds_name],
            tau=tau,
            tau_q95=tau_q95,
            wavelet=wavelet,
            level=level,
            mode=mode,
            project_root=project_root,
        )
        all_summaries.append(summary)
        print_audit_summary(summary)

    # ── 与旧版 (200 随机采样) 对比 ─────────────────────────────────
    old_path = project_root / "artifacts" / "derived" / "main" / "statistics" / "oracle_audit_refined.json"
    if old_path.exists():
        with open(old_path) as f:
            old = json.load(f)
        print("\n" + "=" * 60)
        print("新旧对比 (测试集 300 vs 随机 200)")
        print("=" * 60)
        old_map = {s["dataset"]: s for s in old["summaries"]}
        for s in all_summaries:
            ds = s["dataset"]
            old_s = old_map.get(ds)
            if not old_s:
                continue
            print(f"\n  {ds}:")
            print(f"    {'':>10} {'旧(随机200)':>12} {'新(测试集)':>12} {'差值':>12}")
            for rank_key in sorted(s["audit_table"].keys(), key=int):
                rank = int(rank_key)
                new_worst_mean = max(s["audit_table"][rank_key]["bands"][b]["mean"] for b in BANDS_CF)
                old_entry = old_s["audit_table"].get(rank_key, {})
                old_worst_mean = max(old_entry.get("bands", {}).get(b, {}).get("mean", 0) for b in BANDS_CF) if old_entry else 0
                diff = new_worst_mean - old_worst_mean
                print(f"    rank {rank:>4}: {old_worst_mean:>10.6f}  {new_worst_mean:>10.6f}  {diff:>+11.6f}")

    # ── 保存 ───────────────────────────────────────────────────────
    output = {
        "method": "test_set_oracle_audit",
        "description": "使用主实验相同测试集快照 (时间顺序最后 20%) 的 Oracle audit",
        "tau": tau,
        "tau_q95": tau_q95,
        "summaries": all_summaries,
    }

    json_path = output_dir / "oracle_audit_testset.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON: {json_path}")

    # CSV
    csv_path = output_dir / "oracle_audit_testset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rank", "band", "mean", "median", "q95", "max", "std",
                         "n_samples", "mean_ok", "q95_ok", "strict_ok"])
        for s in all_summaries:
            for r_str, entry in s["audit_table"].items():
                for b in BANDS_CF:
                    bs = entry["bands"][b]
                    writer.writerow([s["dataset"], int(r_str), b,
                                     f"{bs['mean']:.6f}", f"{bs['median']:.6f}",
                                     f"{bs['q95']:.6f}", f"{bs['max']:.6f}",
                                     f"{bs['std']:.6f}", bs["n_samples"],
                                     bs["mean_ok"], bs["q95_ok"], entry["strict_ok"]])
    print(f"CSV: {csv_path}")

    # TXT
    txt_path = output_dir / "oracle_audit_testset.txt"
    with open(txt_path, "w") as f:
        for s in all_summaries:
            f.write(f"Dataset: {s['dataset']}\n")
            f.write(f"  Tau={s['config']['tau']}, Tau/5={s['config']['tau_q95']}\n")
            f.write(f"  Test samples: {s['config']['n_test_samples']}\n")
            f.write(f"  Test indices: {s['config']['test_indices'][0]}–{s['config']['test_indices'][-1]}\n")
            f.write(f"  Safe rank (mean<τ):       {s['safe_rank_mean_criterion']}\n")
            f.write(f"  Safe rank (Q95<τ/5):      {s['safe_rank_q95_criterion']}\n")
            f.write(f"  Safe rank (BOTH):         {s['safe_rank_strict_criterion']}\n")
            f.write("\n")
    print(f"TXT: {txt_path}")

    print("\n✓ P0-1 修复完成")


if __name__ == "__main__":
    main()

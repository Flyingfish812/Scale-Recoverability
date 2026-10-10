#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-driven contour rendering of the supplied cylinder wake.

Standalone: python graphical_abstract_contours.py --data-dir /path/to/data
In the original project: copy next to graphical_abstract.py and run it there.

Only NumPy, SciPy, Matplotlib and Pillow are required. No style.py or luna
imports are needed. Original project raw arrays and CSV masks are supported.
The supplied no-text, single-panel, 1.2:1 layout is retained for the main image.

Drawing recipe found at:
https://goswami-13.github.io/posts/2024/06/blog-post-23/
The author's snippet defines contourf and signed contour levels but omits the
definition of `cmap`. Our palette is an approximation, not the author's LUT.
Contours here describe vorticity, NOT wavelet-band recoverability.
Cubic interpolation is display-only and does not create measured resolution.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.ndimage import map_coordinates

# Most useful controls: change these before editing individual drawing calls.
CROP = (32, 86, 17, 62)  # c0, c1, r0, r1; width/height = 54/45 = 1.2
WIDE_CROP = (28, 112, 17, 62)
CYLINDER = (39.5, 39.5, 5.0)  # row, column, radius in grid units
SNAPSHOT = 40
COLOUR_PCT = 97.0
FILL_LEVELS = 257
UPSAMPLE = 5
CONTOUR_FRACTIONS = (0.18, 0.45, 0.75)  # magnitudes / colour limit; both signs
CONTOUR_WIDTH = 0.80
SENSORS_MODE = "illustrative"  # "illustrative", "mask", "none"
SENSOR_POINTS = ((30, 40), (49, 40), (29, 57), (50, 57),
                 (35, 68), (46, 70), (37, 84))  # row, column
SENSOR_BAND = 15
SENSOR_SIZE = 4.8


def make_wake_cmap() -> LinearSegmentedColormap:
    """Negative: yellow-red-white; positive: white-blue-cyan.

    A narrow flat white centre suppresses barely visible background tint by
    colour mapping only. No values are thresholded or filtered.
    """
    return LinearSegmentedColormap.from_list("wake_reference_approx", [
        (0.000, "#ffff00"), (0.090, "#ffad00"),
        (0.180, "#ff1600"), (0.300, "#850f1b"),
        (0.400, "#dcc3c6"), (0.480, "#ffffff"),
        (0.520, "#ffffff"), (0.600, "#c6c6dd"),
        (0.700, "#181870"), (0.820, "#0000ef"),
        (0.910, "#009dff"), (1.000, "#00eeee"),
    ], N=1024)


def vorticity(velocity: np.ndarray) -> np.ndarray:
    """Same unit-grid convention as the supplied original script."""
    velocity = np.asarray(velocity)
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError(f"Expected (2,H,W) velocity, got {velocity.shape}")
    return np.gradient(velocity[1], axis=1) - np.gradient(velocity[0], axis=0)


def load_data(data_dir: Path, raw: Path | None, snapshot: int):
    if raw is not None:
        with np.load(raw, allow_pickle=False) as d:
            ref = d["target_nchw"][snapshot].copy()
            pred = d["output_nchw"][snapshot].copy()
        return ref, pred, None
    ref_file = data_dir / "graphical_abstract_ref.npz"
    pred_file = data_dir / "graphical_abstract_pred.npz"
    if not ref_file.is_file():
        raise FileNotFoundError(f"Missing {ref_file}; use --data-dir or --raw")
    with np.load(ref_file, allow_pickle=False) as d:
        ref = d["ref"].copy()
    pred = None
    if pred_file.is_file():
        with np.load(pred_file, allow_pickle=False) as d:
            pred = d["pred"].copy()
        if pred.shape != ref.shape:
            raise ValueError("Reference and prediction shapes differ")
    field_file = data_dir / "graphical_abstract_field.npy"
    field = np.load(field_file, allow_pickle=False) if field_file.is_file() else None
    if not np.all(np.isfinite(ref)) or (pred is not None and not np.all(np.isfinite(pred))):
        raise ValueError("Velocity arrays contain nonfinite values")
    return ref, pred, field


def display_grid(omega: np.ndarray, crop, factor=UPSAMPLE):
    c0, c1, r0, r1 = crop
    if not (0 <= c0 < c1 <= omega.shape[1] and 0 <= r0 < r1 <= omega.shape[0]):
        raise ValueError(f"Crop {crop} outside field {omega.shape}")
    # Include the half-cell margin of the original imshow extent. Interpolate
    # from the full field, so the crop edge is not an interpolation boundary.
    x = np.linspace(c0 - 0.5, c1 - 0.5, (c1-c0)*factor + 1)
    y = np.linspace(r0 - 0.5, r1 - 0.5, (r1-r0)*factor + 1)
    xx, yy = np.meshgrid(x, y)
    z = map_coordinates(omega.astype(float), [yy, xx], order=3, mode="nearest")
    br, bc, radius = CYLINDER
    # Small guard is hidden by the exact-radius foreground cylinder patch.
    z = np.ma.masked_where((xx-bc)**2 + (yy-br)**2 < (radius-0.15)**2, z)
    return x-c0, y-r0, z


def sensor_locations(mode: str, mask: Path | None, crop):
    if mode == "none":
        return np.empty((0, 2))
    if mode == "mask":
        if mask is None or not mask.is_file():
            raise FileNotFoundError("--sensors mask requires a valid --mask CSV")
        pts = []
        with mask.open(newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                try:
                    r, c = float(row[0]), float(row[1])
                except (ValueError, IndexError):
                    continue
                if abs(r-CYLINDER[0]) <= SENSOR_BAND:
                    pts.append((r, c))
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    else:
        pts = np.asarray(SENSOR_POINTS, dtype=float)
    c0, c1, r0, r1 = crop
    keep = ((pts[:, 1] >= c0-.5) & (pts[:, 1] <= c1-.5)
            & (pts[:, 0] >= r0-.5) & (pts[:, 0] <= r1-.5))
    pts = pts[keep]
    return np.column_stack((pts[:, 1]-c0, pts[:, 0]-r0))


def draw_wake(ax, omega, crop, limit, *, palette="wake", dense=False,
              sensors="none", mask=None):
    x, y, z = display_grid(omega, crop)
    cm = make_wake_cmap() if palette == "wake" else plt.get_cmap("RdBu_r")
    fill = ax.contourf(x, y, z, levels=np.linspace(-limit, limit, FILL_LEVELS),
                       cmap=cm, norm=Normalize(-limit, limit), extend="both",
                       antialiased=False, zorder=1)
    # Rasterize only dense filled polygons for compact PDF; lines stay vector.
    ax.set_rasterization_zorder(1.5)
    fractions = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20) if dense else CONTOUR_FRACTIONS
    levels = limit * np.asarray(fractions)
    width = 0.78 if dense else CONTOUR_WIDTH
    # Explicit styles: independent of house-style rcParams.
    ax.contour(x, y, z, levels=-levels[::-1], colors="#22222a",
               linewidths=width, linestyles="dashed", alpha=0.85, zorder=2)
    ax.contour(x, y, z, levels=levels, colors="#22222a",
               linewidths=width, linestyles="solid", alpha=0.85, zorder=2)
    c0, c1, r0, r1 = crop
    br, bc, radius = CYLINDER
    ax.add_patch(plt.Circle((bc-c0, br-r0), radius,
                            facecolor="white", edgecolor="#22222a",
                            linewidth=1.1, zorder=5))
    pts = sensor_locations(sensors, mask, crop)
    if len(pts):
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=SENSOR_SIZE, mfc="#181822",
                mec="white", mew=0.85, ls="none", zorder=6)
    ax.set(xlim=(-.5, c1-c0-.5), ylim=(-.5, r1-r0-.5), aspect="equal")
    ax.set_axis_off()


def save_single(out_dir, name, omega, crop, limit, **kwargs):
    c0, c1, r0, r1 = crop
    fig = plt.figure(figsize=(6, 6*(r1-r0)/(c1-c0)), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    draw_wake(ax, omega, crop, limit, **kwargs)
    for suffix in ("png", "jpg", "pdf"):
        options = {"pil_kwargs": {"quality": 95}} if suffix == "jpg" else {}
        fig.savefig(out_dir/f"{name}.{suffix}", dpi=300, bbox_inches=None,
                    facecolor="white", **options)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--raw", type=Path, help="Original test_raw.npz")
    parser.add_argument("--snapshot", type=int, default=SNAPSHOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sensors", choices=("illustrative", "mask", "none"), default=SENSORS_MODE)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--colour-limit", type=float, help="Shared symmetric limit, overriding percentile")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    project = next((p for p in here.parents if (p/"artifacts/pod_model_sweep_nc").is_dir()), None)
    candidates = (here, here/"data", here.parent/"upload", Path.cwd(), Path.cwd()/"upload")
    data_dir = args.data_dir or next((p for p in candidates if (p/"graphical_abstract_ref.npz").is_file()), here)
    raw = args.raw
    if raw is None and args.data_dir is None and not (data_dir/"graphical_abstract_ref.npz").is_file() and project:
        raw = project/"artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz"
    out = args.out_dir or (project/"artifacts/figures/contour_style" if project else here/"contour_output")
    out.mkdir(parents=True, exist_ok=True)
    mask = args.mask
    if mask is None and project:
        mask = project/"masks_families/family_01/masks/cylinder2d_80x160_random_inc_n020.csv"
    ref, pred, supplied_field = load_data(data_dir, raw, args.snapshot)
    omega = vorticity(ref)
    c0, c1, r0, r1 = CROP
    cropped = omega[r0:r1, c0:c1]
    # Use all supplied inputs and check that the standalone field corresponds
    # to the same snapshot/component/crop before using it for normalisation.
    if supplied_field is not None:
        if supplied_field.shape != cropped.shape or not np.allclose(supplied_field, cropped, rtol=1e-5, atol=1e-6):
            raise ValueError("graphical_abstract_field.npy does not match reference vorticity and CROP")
        cropped = supplied_field
    rr, cc = np.mgrid[r0:r1, c0:c1]
    outside = (rr-CYLINDER[0])**2 + (cc-CYLINDER[1])**2 > CYLINDER[2]**2
    limit = (args.colour_limit if args.colour_limit is not None
             else float(np.percentile(np.abs(cropped[outside]), COLOUR_PCT)))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Colour limit must be finite and positive")
    plt.rcParams.update({"savefig.bbox": None, "font.family": "DejaVu Sans"})
    save_single(out, "graphical_abstract_recommended", omega, CROP, limit,
                sensors=args.sensors, mask=mask)
    save_single(out, "graphical_abstract_reference_style", omega, CROP, limit, dense=True)
    save_single(out, "graphical_abstract_wide", omega, WIDE_CROP, limit, dense=True)
    # Comparison contact sheet, clearly labelled and NOT a submission image.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
    for ax, title, palette, dense, sensors in zip(
        axs, ("A  RdBu + data contours", "B  Approximate source palette", "C  Fewer contours + sensors"),
        ("RdBu", "wake", "wake"), (False, True, False), ("none", "none", args.sensors)):
        draw_wake(ax, omega, CROP, limit, palette=palette, dense=dense, sensors=sensors, mask=mask)
        ax.set_title(title, fontsize=12, pad=12)
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.91, wspace=.04)
    fig.savefig(out/"style_comparison.png", dpi=180, bbox_inches=None)
    plt.close(fig)
    metrics = {"crop": CROP, "colour_limit": limit, "colour_percentile": COLOUR_PCT,
               "field_matches_reference": supplied_field is not None,
               "contour_fractions": CONTOUR_FRACTIONS,
               "sensor_mode": args.sensors, "cmap": "approximation of source, not exact LUT"}
    if pred is not None:
        wp = vorticity(pred)
        diff = (wp-omega)[r0:r1, c0:c1]
        metrics.update(velocity_relative_l2=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),
                       cropped_vorticity_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(cropped)),
                       cropped_vorticity_residual_max=float(np.max(np.abs(diff))))
        save_single(out, "graphical_abstract_prediction", wp, CROP, limit, sensors=args.sensors, mask=mask)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
        for ax, w, title in zip(axs[:2], (omega, wp), ("Reference", "Prediction: same colour scale")):
            draw_wake(ax, w, CROP, limit)
            ax.set_title(title, fontsize=12)
        x, y, dz = display_grid(wp-omega, CROP)
        im = axs[2].pcolormesh(x, y, abs(dz), shading="auto", cmap="magma", vmin=0,
                               vmax=float(abs(diff).max()), rasterized=True)
        axs[2].add_patch(plt.Circle((CYLINDER[1]-c0, CYLINDER[0]-r0), CYLINDER[2], fc="white", ec="#222", zorder=5))
        axs[2].set(aspect="equal", xlim=(-.5,c1-c0-.5), ylim=(-.5,r1-r0-.5))
        axs[2].axis("off")
        axs[2].set_title("|Vorticity residual|: separate scale", fontsize=12)
        fig.colorbar(im, ax=axs[2], shrink=.66, pad=.02)
        fig.subplots_adjust(left=.015, right=.985, bottom=.04, top=.91, wspace=.10)
        fig.savefig(out/"data_comparison.png", dpi=180, bbox_inches=None)
        plt.close(fig)
    (out/"render_parameters.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Figures saved in {out.resolve()}")


if __name__ == "__main__":
    main()
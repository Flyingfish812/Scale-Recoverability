#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphical_abstract.py — single-panel graphical abstract for the JFM submission

JFM uses the graphical abstract as a small table-of-contents thumbnail (about 2.4 cm x 2 cm) and as a cover candidate, so it must be a single panel, 1.2:1, without caption and without text.  The wake is rendered in the approximate palette of the reference vortex-street illustration (drawing recipe: goswami-13.github.io/posts/2024/06/blog-post-23): a yellow-red-white-blue-cyan colour map filled by contour levels, with a few signed vorticity contours on top (solid for positive, dashed for negative), and the cylinder section and the sparse measurement locations drawn in.  The palette is an approximation, not the reference LUT; the contours describe the vorticity of the flow, not the wavelet-band recoverability of the paper.

The measurement locations are illustrative: the nested sensor masks are drawn randomly (`masks_families/`), so the artwork shows a representative set placed where it reads as sparse sampling of the wake; SENSORS_MODE = "mask" plots the family_01 / M = 20 mask inside the field of view instead.  Set SCALE_GUIDES = True to overlay the three hand-drawn guide loops (kept from the earlier layout) on top of the contours.

Data
    artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz
        target_nchw (n, 2, 80, 160)  reference fields
    masks_families/family_01/masks/cylinder2d_80x160_random_inc_n020.csv
        sensor mask of the M = 20 configuration (SENSORS_MODE = "mask")

Output
    artifacts/figures/graphical_abstract.jpg   (1800 x 1500 px, 300 dpi; submit)
    artifacts/figures/graphical_abstract.pdf   (vector copy for the record)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps  # noqa: E402

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.ndimage import map_coordinates

OUT_DIR = _ROOT / "artifacts" / "figures"
PRED = (_ROOT / "artifacts" / "pod_model_sweep_nc" / "mlp_n0020" / "seed000"
        / "tests" / "s0000" / "test_raw.npz")
MASK = (_ROOT / "masks_families" / "family_01" / "masks"
        / "cylinder2d_80x160_random_inc_n020.csv")
SNAPSHOT = 40          # one fully developed wake state
# 54 columns x 45 rows = 1.2:1 exactly: the tightest window that still encloses
# the cylinder and the whole near-wake structure with a narrow margin above and below the street.
CROP = (32, 86, 17, 62)
# The true body is a circle of radius 5 grid units at (39.5, 39.5).  It is drawn slightly smaller because it is only a locator for the reader: the wake has to stay the subject.
CYLINDER = (39.5, 39.5, 5.0)
# The measurement mask is a randomly drawn nested set, so the artwork uses a representative set of measurement locations instead of the raw mask; the layout is chosen to read as "sparse sensors sampling the wake" at thumbnail size (one at each cylinder shoulder, two on the shear layers, two inside the
# wake and one in the far wake).  Set SENSORS_MODE = "mask" to plot the
# family_01 / M = 20 mask instead.
SENSORS_MODE = "illustrative"
SENSOR_POINTS = ((30, 40), (49, 40), (29, 57), (50, 57),
                 (35, 68), (46, 70), (37, 84))   # (row, column) in field units
SENSOR_BAND = 15               # half-width of the mask band used in "mask" mode
COLOUR_PCT = 97.0              # symmetric colour limit, in percentiles of |omega|; body interior excluded
FILL_LEVELS = 257              # contour-fill levels over [-limit, limit]
UPSAMPLE = 5                   # cubic upsampling factor of the crop window for the display grid
# Signed vorticity contours drawn on top of the fill as fractions of the colour limit
# (solid positive, dashed negative): the sparse linework that stays readable at thumbnail size.
CONTOUR_FRACTIONS = (0.18, 0.45, 0.75)
CONTOUR_WIDTH = 0.80
CONTOUR_COLOUR = "#22222a"
# Hand-drawn guide loops kept from the earlier layout (crop coordinates, x right / y up);
# drawn only when SCALE_GUIDES is True.
SCALE_GUIDES = False
GUIDE_STYLE = ((1.45, 0.86), (1.10, 0.80), (0.92, 0.76))  # outer / middle / inner: (lw, alpha)
outer_pts = [        # outer guide loop
    (1.0, 24.0),
    (8.0, 31.0),
    (24.0, 30.5),
    (41.0, 37.0),
    (50.0, 31.0),
    (52.5, 18.0),
    (51.0, 8.0),
    (40.0, 12.5),
    (20.0, 11.5),
    (5.0, 15.0),
]
middle_pts = [       # middle guide loop
    (6.0, 17.0),
    (6.0, 27.0),
    (15.0, 28.5),
    (24.0, 27.0),
    (38.0, 31.5),
    (39.0, 22.0),
    (40.0, 16.0),
    (33.0, 16.0),
    (18.0, 14.0),
]
inner_pts = [        # inner guide loop
    (10.0, 19.0),
    (9.0, 24.0),
    (14.0, 25.5),
    (24.0, 24.0),
    (35.0, 28.0),
    (31.0, 21.0),
    (23.0, 17.5),
    (14.5, 18.5),
]


def vorticity(field: np.ndarray) -> np.ndarray:
    """Signed vorticity of a two-component velocity field (u, v)."""
    u, v = field[0], field[1]
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    return np.asarray(dvdx - dudy, dtype=np.float32)


def make_wake_cmap() -> LinearSegmentedColormap:
    """Negative: yellow-red-white; positive: white-blue-cyan.

    A narrow flat white centre suppresses barely visible background tint by
    colour mapping only.  No values are thresholded or filtered.
    """
    return LinearSegmentedColormap.from_list("wake_reference_approx", [
        (0.000, "#ffff00"), (0.090, "#ffad00"),
        (0.180, "#ff1600"), (0.300, "#850f1b"),
        (0.400, "#dcc3c6"), (0.480, "#ffffff"),
        (0.520, "#ffffff"), (0.600, "#c6c6dd"),
        (0.700, "#181870"), (0.820, "#0000ef"),
        (0.910, "#009dff"), (1.000, "#00eeee"),
    ], N=1024)


def display_grid(omega: np.ndarray, crop, factor: int = UPSAMPLE):
    """Cubic-upsampled crop window for the fill, with the cylinder body masked out."""
    c0, c1, r0, r1 = crop
    if not (0 <= c0 < c1 <= omega.shape[1] and 0 <= r0 < r1 <= omega.shape[0]):
        raise ValueError(f"Crop {crop} outside field {omega.shape}")
    # Include the half-cell margin of the display extent and interpolate from the
    # full field, so the crop edge is not an interpolation boundary.
    x = np.linspace(c0 - 0.5, c1 - 0.5, (c1 - c0) * factor + 1)
    y = np.linspace(r0 - 0.5, r1 - 0.5, (r1 - r0) * factor + 1)
    xx, yy = np.meshgrid(x, y)
    z = map_coordinates(omega.astype(float), [yy, xx], order=3, mode="nearest")
    br, bc, radius = CYLINDER
    # Small guard is hidden by the exact-radius foreground cylinder patch.
    z = np.ma.masked_where((xx - bc) ** 2 + (yy - br) ** 2 < (radius - 0.15) ** 2, z)
    return x - c0, y - r0, z


def sensors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = [], []
    with path.open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                r, c = int(float(row[0])), int(float(row[1]))
            except (ValueError, IndexError):
                continue
            rows.append(r)
            cols.append(c)
    return np.asarray(rows), np.asarray(cols)


def bezier_path(start, segments, closed=True):
    verts = [start]
    codes = [MplPath.MOVETO]

    for c1, c2, end in segments:
        verts.extend([c1, c2, end])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    if closed:
        verts.append((0, 0))
        codes.append(MplPath.CLOSEPOLY)

    return MplPath(verts, codes)


def add_scale_guide(ax, path, lw, alpha):
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor="0.18",
            lw=lw,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
    )


def catmull_rom_closed_path(points):
    """
    Build a smooth closed curve that passes through all given anchor points.
    points: sequence of (x, y), do NOT repeat the first point at the end.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points for a closed spline.")

    segments = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        # Catmull-Rom -> cubic Bézier
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        segments.append((tuple(c1), tuple(c2), tuple(p2)))

    return bezier_path(tuple(pts[0]), segments, closed=True)

def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    data = np.load(PRED)
    ref = data["target_nchw"][SNAPSHOT]
    c0, c1, r0, r1 = CROP
    height, width = r1 - r0, c1 - c0

    omega = vorticity(ref)
    # Symmetric colour limit on the cropped vorticity, body interior excluded.
    cropped = omega[r0:r1, c0:c1]
    rr, cc = np.mgrid[r0:r1, c0:c1]
    outside = (rr - CYLINDER[0]) ** 2 + (cc - CYLINDER[1]) ** 2 > CYLINDER[2] ** 2
    lim = float(np.percentile(np.abs(cropped[outside]), COLOUR_PCT))

    x, y, z = display_grid(omega, CROP)

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.contourf(x, y, z, levels=np.linspace(-lim, lim, FILL_LEVELS),
                cmap=make_wake_cmap(), norm=Normalize(-lim, lim),
                extend="both", antialiased=False, zorder=1)
    # Only the dense fill is rasterised in the vector copy; the linework stays vector.
    ax.set_rasterization_zorder(1.5)

    levels = lim * np.asarray(CONTOUR_FRACTIONS)
    ax.contour(x, y, z, levels=-levels[::-1], colors=CONTOUR_COLOUR,
               linewidths=CONTOUR_WIDTH, linestyles="dashed", alpha=0.85,
               zorder=2)
    ax.contour(x, y, z, levels=levels, colors=CONTOUR_COLOUR,
               linewidths=CONTOUR_WIDTH, linestyles="solid", alpha=0.85,
               zorder=2)
    if SCALE_GUIDES:
        for pts_g, style in zip((outer_pts, middle_pts, inner_pts), GUIDE_STYLE):
            add_scale_guide(ax, catmull_rom_closed_path(pts_g), *style)

    # The cylinder section is drawn explicitly: without it the picture reads as an arbitrary jet, with it as a cylinder wake.
    body_r, body_c, body_radius = CYLINDER
    ax.add_patch(plt.Circle((body_c - c0, body_r - r0), body_radius,
                            facecolor="white", edgecolor="#22222a", lw=1.1,
                            zorder=5))

    if SENSORS_MODE == "mask":
        r, c = sensors(MASK)
        keep = ((c >= c0) & (c < c1) & (r >= r0) & (r < r1)
                & (np.abs(r - CYLINDER[0]) <= SENSOR_BAND))
        pts = np.column_stack([c[keep] - c0, r[keep] - r0])
    else:
        pts = np.asarray([(col - c0, row - r0) for row, col in SENSOR_POINTS],
                         dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=4.8, mfc="#181822", mec="white",
            mew=0.85, ls="none", zorder=6)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)

    # The house style sets a tight bounding box; the graphical abstract has to keep the exact 1.2:1 canvas that JFM asks for, so save the full figure.
    plt.rcParams["savefig.bbox"] = "standard"
    pdf = OUT_DIR / "graphical_abstract.pdf"
    fig.savefig(OUT_DIR / "graphical_abstract.jpg", format="jpg", dpi=300,
                pil_kwargs={"quality": 95})
    fig.savefig(pdf, format="pdf", dpi=300, metadata={"CreationDate": None})
    fig.savefig("/tmp/ga_preview.png", format="png", dpi=170)
    plt.close(fig)
    print(f"  [jpg] {OUT_DIR / 'graphical_abstract.jpg'}")
    print(f"  [pdf] {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

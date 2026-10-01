#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphical_abstract.py — single-panel graphical abstract for the JFM submission

JFM uses the graphical abstract as a small table-of-contents thumbnail (about
2.4 cm x 2 cm) and as a cover candidate, so it must be a single panel, 1.2:1,
without caption and without text.  The image is therefore kept deliberately
simple and self-explanatory at that size: one reference cylinder-wake state with
the cylinder section drawn in, the vortex street filling most of the frame, the
sparse measurement locations marked, and two iso-lines that suggest the
large-scale wake envelope and the finer shear-layer structures.

The measurement locations are illustrative: the nested sensor masks are drawn
randomly (`masks_families/`), so the artwork shows a representative set placed
where it reads as sparse sampling of the wake; SENSORS_MODE = "mask" plots the
family_01 / M = 20 mask inside the field of view instead.

Data
    artifacts/pod_model_sweep_nc/mlp_n0020/seed000/tests/s0000/test_raw.npz
        target_nchw (n, 2, 80, 160)  reference fields
        output_nchw (n, 2, 80, 160)  MLP reconstruction at (M, sigma) = (20, 0)
        (used only for FINE_SOURCE = "lost")
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

from luna.wavelet.transform import decompose_field_2d  # noqa: E402

OUT_DIR = _ROOT / "artifacts" / "figures"
PRED = (_ROOT / "artifacts" / "pod_model_sweep_nc" / "mlp_n0020" / "seed000"
        / "tests" / "s0000" / "test_raw.npz")
MASK = (_ROOT / "masks_families" / "family_01" / "masks"
        / "cylinder2d_80x160_random_inc_n020.csv")
SNAPSHOT = 40          # one fully developed wake state
FINE_BANDS = ("W1",)   # finest band of the streamwise velocity
# 72 columns x 60 rows = 1.2:1 exactly: the tightest window that still encloses
# the cylinder and the whole near-wake structure with a narrow margin above and
# below the street.
CROP = (20, 92, 10, 70)
# The true body is a circle of radius 5 grid units at (39.5, 39.5).  It is drawn
# slightly smaller because it is only a locator for the reader: the wake has to
# stay the subject.
CYLINDER = (39.5, 39.5, 3.9)
# The measurement mask is a randomly drawn nested set, so the artwork uses a
# representative set of measurement locations instead of the raw mask; the
# layout is chosen to read as "sparse sensors sampling the wake" at thumbnail
# size (one at each cylinder shoulder, two on the shear layers, two inside the
# wake and one in the far wake).  Set SENSORS_MODE = "mask" to plot the
# family_01 / M = 20 mask instead.
SENSORS_MODE = "illustrative"
SENSOR_POINTS = ((30, 40), (49, 40), (29, 57), (50, 57),
                 (35, 68), (46, 70), (37, 84))   # (row, column) in field units
SENSOR_BAND = 15               # half-width of the mask band used in "mask" mode
# The iso-lines are drawn as a contour map over the wake, which is what makes
# the multiscale structure readable at thumbnail size: one outer envelope plus
# two inner levels.  One or two levels degenerate into a few unexplained closed
# loops, and four or more give the outer lines a "tree ring" look.
# FINE_SOURCE = "lost" maps the fine-scale content the reconstruction misses.
FINE_SOURCE = "reference"      # "lost" (reconstruction) or "reference" (truth)
CONTOUR_PCTS = (70.0, 88.0, 96.0)
CONTOUR_SIGMA = 3.0
CONTOUR_WIDTHS = (0.55, 0.8, 1.05)
BACKGROUND = "vorticity"   # "vorticity" (vortex street) or "u" (deficit band)
COLOUR_PCT = 98.0              # symmetric colour limit, in percentiles of |omega|


def vorticity(field: np.ndarray) -> np.ndarray:
    """Signed vorticity of a two-component velocity field (u, v)."""
    u, v = field[0], field[1]
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    return np.asarray(dvdx - dudy, dtype=np.float32)


def fine_scale(field: np.ndarray) -> np.ndarray:
    """Finest wavelet band of the streamwise velocity."""
    bands = decompose_field_2d(np.asarray(field[0], dtype=np.float64))
    return np.asarray(sum(bands[b] for b in FINE_BANDS), dtype=np.float32)


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


def main() -> int:
    ps.apply()
    import matplotlib.pyplot as plt

    data = np.load(PRED)
    ref = data["target_nchw"][SNAPSHOT]
    pred = data["output_nchw"][SNAPSHOT]
    c0, c1, r0, r1 = CROP
    height, width = r1 - r0, c1 - c0

    omega = vorticity(ref)
    if BACKGROUND == "vorticity":
        field = omega[r0:r1, c0:c1]
        lim = float(np.percentile(np.abs(field), COLOUR_PCT))
        kwargs = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        field = np.asarray(ref[0], dtype=np.float32)[r0:r1, c0:c1]
        # colour scale centred on the free stream, so that the wake reads as a
        # deficit band on a neutral background
        kwargs = dict(cmap="RdBu_r", vmin=-0.2, vmax=2.2)

    extent = (-0.5, width - 0.5, -0.5, height - 0.5)
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.imshow(field, origin="lower", interpolation="bilinear", **kwargs)

    from scipy.ndimage import gaussian_filter

    # A contour map of the fine-scale content that the sparse reconstruction
    # misses: nested thin lines over the shear layers and the vortex cores,
    # which is the multiscale cue that survives thumbnailing.
    fine_field = (fine_scale(ref) - fine_scale(pred)
                  if FINE_SOURCE == "lost" else fine_scale(ref))
    fine = np.abs(fine_field)[r0:r1, c0:c1]
    fine_smooth = gaussian_filter(fine, sigma=CONTOUR_SIGMA, mode="nearest")
    levels = [float(np.percentile(fine_smooth, p)) for p in CONTOUR_PCTS]
    ax.contour(fine_smooth, levels=levels, colors="0.15", alpha=0.85,
               linewidths=CONTOUR_WIDTHS, origin="lower", extent=extent,zorder=2)

    # The cylinder section is drawn explicitly: without it the picture reads as
    # an arbitrary jet, with it as a cylinder wake.  The body is grey rather
    # than white because the free stream upstream of the cylinder is white in
    # the vorticity colour scale.
    body_r, body_c, body_radius = CYLINDER
    ax.add_patch(plt.Circle((body_c - c0, body_r - r0), body_radius,
                            facecolor="0.78", edgecolor="0.25", lw=0.6,
                            zorder=3))

    if SENSORS_MODE == "mask":
        r, c = sensors(MASK)
        keep = ((c >= c0) & (c < c1) & (r >= r0) & (r < r1)
                & (np.abs(r - CYLINDER[0]) <= SENSOR_BAND))
        pts = np.column_stack([c[keep] - c0, r[keep] - r0])
    else:
        pts = np.asarray([(col - c0, row - r0) for row, col in SENSOR_POINTS],
                         dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=5.0, mfc="0.05", mec="white",
            mew=0.7, ls="none", zorder=4)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)

    # The house style sets a tight bounding box; the graphical abstract has to
    # keep the exact 1.2:1 canvas that JFM asks for, so save the full figure.
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

"""Draw every figure of the paper.

The figures are drawn by the scripts under ``applications/figures``, one module per figure or group of figures, from the artifacts written by the statistics layer. ``applications.figures.make_all_figures`` runs them in the order of the paper and writes a vector PDF per figure into ``artifacts/figures``; this module is the entry point of the pipeline step, so that the sequence of the repository is read from ``applications/pipelines`` alone.

Prerequisites
    the statistics under artifacts/statistics/
             (see applications/pipelines/04_compute_statistics.py)

Outputs
    artifacts/figures/*.pdf

Pipeline
    previous  applications/pipelines/04_compute_statistics.py this step writes artifacts/figures/
    next      applications/figures/publish_figures.py (copy into the paper tree)

Usage
    python applications/pipelines/05_make_figures.py
    python -m applications.figures.make_all_figures --only fig03_sensor_noise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.figures.make_all_figures import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

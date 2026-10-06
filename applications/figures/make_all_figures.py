"""Regenerate every figure of the paper.

Each module in this directory draws one figure, or one group of figures, from the artifacts written by the statistics layer, and writes a vector PDF into ``artifacts/figures``. The registry below lists the scripts in the order of the paper, main text first and supplementary material after it. ``style.py`` fixes the shared publication style and ``records.py`` loads the artifact records.

Publishing the PDFs into the paper tree is a separate, explicit step (``applications/figures/publish_figures.py``), so the repository itself never contains generated output.

Usage
    python applications/figures/make_all_figures.py
    python applications/figures/make_all_figures.py --only fig03_sensor_noise.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    # main text, in figure order
    "fig01_known_scale.py",
    "fig02_global_vs_scale.py",
    "fig03_sensor_noise.py",
    "fig05_modal_hierarchy.py",
    "fig09_cross_model_bands.py",
    "fig06_robustness.py",
    # supplementary material, in figure order
    "fig01_method_framework.py",
    "fig02_analytical_benchmark.py",
    "figS01_oracle.py",
    "figS04_three_layer.py",
    "fig03_counterexample.py",
    "fig05_wavelet_vs_fourier.py",
    "figS02_phase.py",
    "figS05_mode_scale_energy.py",
    "figS03_diagnostics.py",
    "figS03c_pod_dominant.py",
    "figS07_sensor_family_ger.py",
    "figS08_sensor_family_paired.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only the named scripts")
    args = parser.parse_args()

    scripts = args.only or SCRIPTS
    ok = True
    for name in scripts:
        path = HERE / name
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        if subprocess.run([sys.executable, str(path)]).returncode != 0:
            ok = False
            print(f"  [!] {name} failed")
    print("\n[done] all scripts finished" + ("" if ok else " (some failed)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

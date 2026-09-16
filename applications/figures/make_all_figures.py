"""Regenerate every figure of the paper.

Each script in this directory draws one figure or one group of figures from the
artifacts written by the statistics layer, and writes a vector PDF into
``artifacts/figures``. Publishing those PDFs into the paper tree is a separate,
explicit step (``applications/figures/publish_figures.py``), so that the
repository itself never contains generated output.

Usage
    python applications/figures/make_all_figures.py
    python applications/figures/make_all_figures.py --only fig06_results.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "fig01_method_framework.py",
    "fig02_analytical_benchmark.py",
    "fig03_counterexample.py",
    "fig04_equal_ger.py",
    "fig05_wavelet_vs_fourier.py",
    "fig06_results.py",
    "fig08_recoverability_chain.py",
    "fig09_cross_model_bands.py",
    "fig10_energy_vs_nrmse.py",
    "fig11_wavelet_sensitivity.py",
    "figS01_oracle.py",
    "figS02_phase.py",
    "figS03_diagnostics.py",
    "figS03c_coherent_only.py",
    "figS04_three_layer.py",
    "figS05_mode_scale_energy.py",
    "figS06_tau_sensitivity.py",
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

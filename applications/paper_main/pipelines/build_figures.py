#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_figures.py — regenerate all paper figures from derived statistics.

Runs every canonical figure script (output under
``applications/paper_main/build/figures_raw`` and
``applications/paper_main/build/figures``) and the supplementary
figS3/S4 script.  Figure scripts that would additionally publish a copy
into a private manuscript tree (``thesis_src/``) automatically skip that
step when the tree is absent.

Data source: ``artifacts/derived/main/statistics`` (guaranteed by the
statistics builders run beforehand).

Usage:
    conda run -n luna python -m applications.paper_main.pipelines.build_figures
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG = "applications.paper_main.figures._canonical."

# Canonical figure scripts, in dependency order.
STEPS = [
    (FIG + "fig03_equal_ger",            "fig03 Equal-GER"),
    (FIG + "fig04_wavelet_vs_fourier",   "fig04 wavelet vs Fourier"),
    (FIG + "fig05_ger_sfull_vs_M",       "fig05 GER/S_full vs M (+fig09)"),
    (FIG + "fig06_recoverability_chain", "fig06 recovery chain"),
    (FIG + "fig07_cross_model_bands",    "fig07 cross-model bands"),
    (FIG + "fig08_three_layer",          "fig08 three-layer errors"),
    (FIG + "fig09_energy_vs_nrmse",      "fig09 energy vs NRMSE"),
    (FIG + "fig11_tau_sensitivity",      "fig11 tau sensitivity"),
    (FIG + "figS1_S2_oracle",            "figS1/S2 oracle"),
    (FIG + "figS5_S6_S7_supp",           "figS5/S6/S7 supplementary"),
    (FIG + "figS8_oracle_audit",         "figS8 oracle audit"),
    ("applications.paper_main.figures.build_figS3_S4", "figS3/S4 phase"),
]


def main() -> int:
    ok = True
    for mod, desc in STEPS:
        print(f"\n########## {desc} ({mod}) ##########")
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        if r.returncode != 0:
            ok = False
            print(f"[build_figures] FAILED {mod}")
    print("\n[build_figures] " + ("all figures regenerated" if ok else "some steps failed"))
    print("outputs: applications/paper_main/build/figures*")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

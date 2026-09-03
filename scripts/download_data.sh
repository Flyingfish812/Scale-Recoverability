#!/usr/bin/env bash
# ============================================================================
# Scale-Recoverability — data acquisition helper
#
# This repository intentionally does NOT ship raw data, derived arrays,
# trained models, or masks.  This script explains what is needed to
# reproduce the main-paper experiments and where to get the raw public
# sources.  Files land in git-ignored directories (data/, masks*/, ...).
#
# The main-paper experiments run on the NC dataset; RDB and SST belong to
# experiments that are not part of this public pipeline.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "Scale-Recoverability data setup"
echo "==============================="
echo "Repo root : $ROOT"
echo ""
echo "Required for the main paper (NC only):"
echo "  - data/cylinder2d_q1.npy   (1501, 80, 160, 2)  float32 snapshots, channels = (u, v)"
echo ""
echo "1) NC — 2-D unsteady cylinder wake (Re=160)"
echo "   Raw source : ETH Zürich CGL '2D Unsteady Cylinder Flow' (Gerris)"
echo "                https://cgl.ethz.ch/research/visualization/data.php"
echo "                (cite: Günther, Gross & Theisel, ACM TOG/SIGGRAPH 2017)"
echo "   The experimental array is the CGL NetCDF simulation cropped to the"
echo "   sensor-study region (80 x 160) and split into (u, v) channels."
echo "   Steps: download cylinder2d.nc from the page above, crop the spatial"
echo "   domain to the 80x160 region used in the paper, and export it as"
echo "   data/cylinder2d_q1.npy with shape (1501, 80, 160, 2)."
echo ""
echo "2) RDB — 2-D shallow-water radial dam break (NOT needed for main paper)"
echo "   Raw source : PDEBench on DaRUS, DOI 10.18419/DARUS-2986"
echo "   https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
echo "   File: 2D/shallow-water/2D_rdb_NA_NA.h5 (128 x 128)."
echo ""
echo "3) SST — weekly sea-surface temperature (NOT needed for main paper)"
echo "   Raw source : NOAA OISST, packaged in 'The Senseiver Dataset'"
echo "   https://zenodo.org/records/8290040  (DOI 10.5281/zenodo.8290040)"
echo "   File: sst_weekly.mat (180 x 360 weekly fields)."
echo ""
echo "After placing the arrays, run:"
echo "  conda run -n luna python -m applications.paper_main.pipelines.build_all"
echo ""
echo "NOTE: POD bases, sensor masks and all trained models are produced by"
echo "the training code under features/training/ and stored under artifacts/;"
echo "they are regenerated locally and never committed."

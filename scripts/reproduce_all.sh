#!/usr/bin/env bash
# ============================================================================
# Scale-Recoverability — full main-paper reproduction
#
# Usage:  bash scripts/reproduce_all.sh [env-name]
#
# Requires:
#   - conda environment with the dependencies in environment/environment.yml
#   - data/cylinder2d_q1.npy (see scripts/download_data.sh)
#   - trained artifacts under artifacts/ (produced by features/training/)
#
# The pipeline regenerates statistics, data pools, tables and figures.
# Manuscript-table generation and thesis-figure publishing are skipped
# automatically when the private thesis_src/ tree is absent.
# ============================================================================
set -euo pipefail
ENV_NAME="${1:-luna}"
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "[1/4] checking raw data"
if [[ ! -f "$ROOT/data/cylinder2d_q1.npy" ]]; then
    echo "ERROR: data/cylinder2d_q1.npy not found."
    echo "       See scripts/download_data.sh for the raw source and layout."
    exit 1
fi
echo "      data/cylinder2d_q1.npy OK"

echo "[2/4] running the main-paper pipeline (statistics -> pools -> tables -> figures)"
conda run -n "$ENV_NAME" python -m applications.paper_main.pipelines.build_all

echo "[3/4] running unit tests"
conda run -n "$ENV_NAME" python -m pytest tests/unit -q

echo "[4/4] done"
echo "      statistics : artifacts/derived/main/statistics/"
echo "      figures    : applications/paper_main/build/figures*/"

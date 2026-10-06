#!/usr/bin/env bash
# ============================================================================
# Scale-Recoverability — full main-paper reproduction

# Usage:
# bash scripts/reproduce_all.sh [env-name]

# Requires:
# - conda environment with the dependencies in environment/environment.yml
# - data/cylinder2d_q1.npy (see scripts/download_data.sh)
# - trained runs under artifacts/ (applications/pipelines/03_train_estimators.py)

# The pipeline regenerates the statistics and the figures of the paper. Figures
# are written to artifacts/figures; copying them into the directory that holds the
# paper sources is a separate step (applications/figures/publish_figures.py).
# ============================================================================
set -euo pipefail
ENV_NAME="${1:-luna}"
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "[1/5] checking raw data"
if [[ ! -f "$ROOT/data/cylinder2d_q1.npy" ]]; then
    echo "ERROR: data/cylinder2d_q1.npy not found."
    echo "       See scripts/download_data.sh for the raw source and layout."
    exit 1
fi
echo "      data/cylinder2d_q1.npy OK"

echo "[2/5] checking the trained runs"
conda run -n "$ENV_NAME" python applications/pipelines/03_train_estimators.py --check \
    --models mlp ridge vcnn

echo "[3/5] computing the statistics"
conda run -n "$ENV_NAME" python applications/pipelines/04_compute_statistics.py

echo "[4/5] drawing the figures"
conda run -n "$ENV_NAME" python applications/figures/make_all_figures.py

echo "[5/5] running the unit tests"
conda run -n "$ENV_NAME" python -m pytest tests/unit -q

echo "done"
echo "      statistics : artifacts/statistics/"
echo "      figures    : artifacts/figures/"

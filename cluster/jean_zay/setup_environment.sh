#!/usr/bin/env bash
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.3.0
module load gurobi/13.0.1

VGE_REPO_DIR="${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}"
VGE_GED_ENV_DIR="${VGE_GED_ENV_DIR:-$WORK/environments/vascular-graph-embedding-ged}"

python -m venv --system-site-packages "$VGE_GED_ENV_DIR"
"$VGE_GED_ENV_DIR/bin/python" -m pip install "gurobipy==13.0.1"
"$VGE_GED_ENV_DIR/bin/python" -m pip install -e "$VGE_REPO_DIR"
"$VGE_GED_ENV_DIR/bin/python" -c 'import gurobipy, matplotlib, numpy, pandas, scipy, torch, yaml; print("GED imports verified")'

echo "GED environment ready: $VGE_GED_ENV_DIR"
echo "Run the Slurm preflight before submitting any solve stage."

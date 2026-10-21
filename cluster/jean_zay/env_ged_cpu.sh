#!/usr/bin/env bash
module purge
module load pytorch-gpu/py3/2.3.0
module load gurobi/13.0.1

VGE_GED_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGE_REPO_CANDIDATE="${SLURM_SUBMIT_DIR:-$VGE_GED_SCRIPT_DIR/../..}"
export VGE_REPO_DIR="$(cd "$VGE_REPO_CANDIDATE" && pwd -P)"
export VGE_GED_ENV_DIR="${VGE_GED_ENV_DIR:-$WORK/environments/vascular-graph-embedding-ged}"
export VGE_GED_OUTPUT_ROOT="${VGE_GED_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding/discrete-ged-go-no-go}"
export VGE_GED_LOG_ROOT="${VGE_GED_LOG_ROOT:-$WORK/logs/vascular-graph-embedding}"
if [[ ! -x "$VGE_GED_ENV_DIR/bin/python" ]]; then
  echo "GED environment is missing at $VGE_GED_ENV_DIR; run cluster/jean_zay/setup_environment.sh first" >&2
  return 1
fi
export PATH="$VGE_GED_ENV_DIR/bin:$PATH"
export PYTHONPATH="$VGE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$VGE_GED_OUTPUT_ROOT" "$VGE_GED_LOG_ROOT"
cd "$VGE_REPO_DIR"

#!/usr/bin/env bash
set -euo pipefail
module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
export VGE_REPO_DIR="${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}"
export VGE_OUTPUT_ROOT="${VGE_OUTPUT_ROOT:-$WORK/analysis/vascular-graph-embedding}"
export VGE_ENV_DIR="${VGE_ENV_DIR:-$WORK/environments/vascular-graph-embedding-cpu}"
export PATH="$VGE_ENV_DIR/bin:$PATH"
export PYTHONPATH="$VGE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"


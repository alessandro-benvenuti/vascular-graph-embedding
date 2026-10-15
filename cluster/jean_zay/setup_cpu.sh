#!/usr/bin/env bash
set -euo pipefail
module purge
module load pytorch-gpu/py3/2.3.0
VGE_REPO_DIR="${VGE_REPO_DIR:-$WORK/projects/vascular-graph-embedding}"
VGE_ENV_DIR="${VGE_ENV_DIR:-$WORK/environments/vascular-graph-embedding-cpu}"
python -m venv --system-site-packages "$VGE_ENV_DIR"
"$VGE_ENV_DIR/bin/python" -m pip install --upgrade pip
"$VGE_ENV_DIR/bin/python" -m pip install -e "$VGE_REPO_DIR"

